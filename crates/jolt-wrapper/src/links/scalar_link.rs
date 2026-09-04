use jolt_field::{Fr, One, Ring, Zero};
#[cfg(test)]
use jolt_hyperkzg::{NoopVerifierObserver, VerifierObserver};
use jolt_poly::{BindingOrder, Polynomial, UnivariatePoly};
use jolt_sumcheck::prover::ProveRounds;
use jolt_sumcheck::SumcheckError;

use crate::limb_table::lookup::{link_weights, link_weights_with};
use crate::limb_table::schedule::Layout;
use crate::stream::TermObserver;

pub struct DoryScalarLink<'a> {
    rows: usize,
    positions: &'a [usize],
    rho: Fr,
    layout: &'a Layout,
}

impl<'a> DoryScalarLink<'a> {
    /// Both the public weight and wire MLEs vary in a round; their product is quadratic.
    pub const DEGREE: usize = 2;

    pub fn new(rows: usize, positions: &'a [usize], layout: &'a Layout, rho: Fr) -> Self {
        assert!(rows.is_power_of_two());
        assert!(positions.iter().all(|position| *position < rows));
        assert_eq!(positions.len(), layout.digit_bases as usize - 2);
        Self {
            rows,
            positions,
            rho,
            layout,
        }
    }

    pub(super) fn rows(&self) -> usize {
        self.rows
    }

    pub fn prover(&self, wire: &[Fr]) -> DoryScalarLinkProver {
        let weights = self.weights();
        let input_claim = weights.iter().zip(wire).map(|(&a, &b)| a * b).sum();
        DoryScalarLinkProver {
            wire: Polynomial::new(wire.to_vec()),
            weight: Polynomial::new(weights),
            rounds: self.rows.trailing_zeros() as usize,
            input_claim,
        }
    }

    #[cfg(test)]
    pub fn final_claim(&self, point: &[Fr], wire: Fr) -> Fr {
        self.final_claim_observed(point, wire, &mut NoopVerifierObserver)
    }

    #[cfg(test)]
    pub fn final_claim_observed<O: VerifierObserver>(
        &self,
        point: &[Fr],
        wire: Fr,
        observer: &mut O,
    ) -> Fr {
        assert_eq!(point.len(), self.rows.trailing_zeros() as usize);
        let weight = self.weight_at_observed(point, observer);
        VerifierObserver::fr_mul(observer, weight, wire)
    }

    fn weights(&self) -> Vec<Fr> {
        let mut weights = vec![Fr::zero(); self.rows];
        let link = link_weights(self.layout, self.rho);
        for (&position, &weight) in self.positions.iter().zip(&link) {
            weights[position] += weight;
        }
        weights
    }

    pub(super) fn weight_at_observed<O: TermObserver + ?Sized>(
        &self,
        point: &[Fr],
        observer: &mut O,
    ) -> Fr {
        let weights = link_weights_with(self.layout, self.rho, &mut |a, b| observer.fr_mul(a, b));
        self.positions
            .iter()
            .zip(weights)
            .fold(Fr::zero(), |sum, (&position, weight)| {
                let eq = point
                    .iter()
                    .enumerate()
                    .fold(Fr::one(), |value, (index, &coordinate)| {
                        let bit = (position >> (point.len() - index - 1)) & 1;
                        observer.fr_mul(
                            value,
                            if bit == 1 {
                                coordinate
                            } else {
                                Fr::one() - coordinate
                            },
                        )
                    });
                sum + observer.fr_mul(weight, eq)
            })
    }
}

pub struct DoryScalarLinkProver {
    wire: Polynomial<Fr>,
    weight: Polynomial<Fr>,
    rounds: usize,
    input_claim: Fr,
}

impl DoryScalarLinkProver {
    pub fn input_claim(&self) -> Fr {
        self.input_claim
    }

    pub fn wire_claim(&self) -> Fr {
        self.wire.evals()[0]
    }

    fn bind(&mut self, challenge: Fr) {
        self.wire
            .bind_with_order(challenge, BindingOrder::HighToLow);
        self.weight
            .bind_with_order(challenge, BindingOrder::HighToLow);
    }
}

impl ProveRounds<Fr> for DoryScalarLinkProver {
    fn num_rounds(&self) -> usize {
        self.rounds
    }

    fn prove_round(
        &mut self,
        bind: Option<Fr>,
        round: usize,
        previous_claim: Fr,
    ) -> Result<UnivariatePoly<Fr>, SumcheckError<Fr>> {
        if let Some(challenge) = bind {
            self.bind(challenge);
        }
        let evaluations = [Fr::zero(), Fr::one(), Fr::from_u64(2)].map(|x| {
            (0..self.wire.len() / 2)
                .map(|index| {
                    self.wire
                        .sumcheck_round_eval_with_order(index, x, BindingOrder::HighToLow)
                        * self.weight.sumcheck_round_eval_with_order(
                            index,
                            x,
                            BindingOrder::HighToLow,
                        )
                })
                .sum()
        });
        if evaluations[0] + evaluations[1] != previous_claim {
            return Err(SumcheckError::RoundCheckFailed {
                round,
                expected: previous_claim,
                actual: evaluations[0] + evaluations[1],
            });
        }
        Ok(UnivariatePoly::from_evals(&evaluations))
    }

    fn finish_rounds(&mut self, bind: Fr) -> Result<(), SumcheckError<Fr>> {
        self.bind(bind);
        Ok(())
    }
}
