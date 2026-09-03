use jolt_field::{Fr, One, Ring, Zero};
#[cfg(test)]
use jolt_hyperkzg::{NoopVerifierObserver, VerifierObserver};
use jolt_poly::{BindingOrder, Polynomial, UnivariatePoly};
use jolt_sumcheck::prover::ProveRounds;
use jolt_sumcheck::SumcheckError;

use crate::limb_table::lookup::{link_weights, link_weights_with};
use crate::limb_table::schedule::Layout;
use crate::stream::TermObserver;

use super::{RelationCellLayout, RelationTableWitness};

pub struct DoryScalarLink<'a> {
    rows: usize,
    base: usize,
    capacity: usize,
    rho: Fr,
    layout: &'a Layout,
}

impl<'a> DoryScalarLink<'a> {
    /// Both the public weight and wire MLEs vary in a round; their product is quadratic.
    pub const DEGREE: usize = 2;

    pub fn new(rows: usize, cells: RelationCellLayout, layout: &'a Layout, rho: Fr) -> Self {
        assert!(rows.is_power_of_two());
        assert!(cells.dory_scalar_capacity.is_power_of_two());
        assert_eq!(cells.dory_scalar_base % cells.dory_scalar_capacity, 0);
        Self {
            rows,
            base: cells.dory_scalar_base,
            capacity: cells.dory_scalar_capacity,
            rho,
            layout,
        }
    }

    pub(super) fn rows(&self) -> usize {
        self.rows
    }

    pub fn prover(&self, witness: &RelationTableWitness) -> DoryScalarLinkProver {
        let weights = self.weights();
        let wire = witness.columns[0].clone();
        let input_claim = weights.iter().zip(&wire).map(|(&a, &b)| a * b).sum();
        DoryScalarLinkProver {
            wire: Polynomial::new(wire),
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
        let named = self.layout.digit_bases as usize - 2;
        weights[self.base..self.base + named].copy_from_slice(&link[..named]);
        weights
    }

    pub(super) fn weight_at_observed<O: TermObserver + ?Sized>(
        &self,
        point: &[Fr],
        observer: &mut O,
    ) -> Fr {
        let suffix = self.capacity.trailing_zeros() as usize;
        let prefix = point.len() - suffix;
        let prefix_index = self.base >> suffix;
        let mut value = Fr::one();
        for (index, &coordinate) in point[..prefix].iter().enumerate() {
            let bit = (prefix_index >> (prefix - index - 1)) & 1;
            value = observer.fr_mul(
                value,
                if bit == 1 {
                    coordinate
                } else {
                    Fr::one() - coordinate
                },
            );
        }
        let named = self.layout.digit_bases as usize - 2;
        let mut layer = link_weights_with(self.layout, self.rho, &mut |a, b| observer.fr_mul(a, b));
        layer.truncate(named);
        layer.resize(self.capacity, Fr::zero());
        for &coordinate in &point[prefix..] {
            let half = layer.len() / 2;
            for index in 0..half {
                let low = layer[index];
                let difference = layer[index + half] - low;
                layer[index] = low + observer.fr_mul(coordinate, difference);
            }
            layer.truncate(half);
        }
        observer.fr_mul(value, layer[0])
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
