//! Eq-weighted claim transport into the common row point.

use jolt_field::{Fr, Ring, Zero};
use jolt_poly::{BindingOrder, EqPolynomial, MultilinearPoly, Polynomial, UnivariatePoly};
use jolt_sumcheck::prover::ProveRounds;
use jolt_sumcheck::SumcheckError;
use rayon::prelude::*;
use thiserror::Error;

const DEGREE: usize = 2;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum CarryError {
    #[error("carry polynomial length must be a nonzero power of two")]
    InvalidLength,
    #[error("carry point has {actual} coordinates, expected {expected}")]
    PointDimension { expected: usize, actual: usize },
    #[error("carry input claim does not match the source evaluation")]
    ClaimMismatch,
}

/// Proves `claim = sum_x eq(source, x) * polynomial(x)` while binding the
/// polynomial to the enclosing stage's row point.
pub struct CarryProver {
    polynomial: Polynomial<Fr>,
    eq_source: Polynomial<Fr>,
    rounds: usize,
}

impl CarryProver {
    pub fn new(evaluations: &[Fr], source_point: &[Fr], claim: Fr) -> Result<Self, CarryError> {
        if evaluations.is_empty() || !evaluations.len().is_power_of_two() {
            return Err(CarryError::InvalidLength);
        }
        let rounds = evaluations.len().trailing_zeros() as usize;
        if source_point.len() != rounds {
            return Err(CarryError::PointDimension {
                expected: rounds,
                actual: source_point.len(),
            });
        }
        if evaluations.evaluate(source_point) != claim {
            return Err(CarryError::ClaimMismatch);
        }
        Ok(Self {
            polynomial: Polynomial::new(evaluations.to_vec()),
            eq_source: Polynomial::new(EqPolynomial::<Fr>::evals(source_point, None)),
            rounds,
        })
    }

    pub fn degree(&self) -> usize {
        DEGREE
    }

    pub fn final_evaluation(&self) -> Fr {
        self.polynomial.evals()[0]
    }

    fn bind(&mut self, challenge: Fr) {
        self.polynomial
            .bind_with_order(challenge, BindingOrder::HighToLow);
        self.eq_source
            .bind_with_order(challenge, BindingOrder::HighToLow);
    }
}

impl ProveRounds<Fr> for CarryProver {
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
        let half = self.polynomial.len() / 2;
        let evaluations = (0..half)
            .into_par_iter()
            .map(|index| {
                let mut local = [Fr::zero(); DEGREE + 1];
                for (x, evaluation) in local.iter_mut().enumerate() {
                    let x = Fr::from_u64(x as u64);
                    *evaluation = self.polynomial.sumcheck_round_eval(index, x)
                        * self.eq_source.sumcheck_round_eval(index, x);
                }
                local
            })
            .reduce(
                || [Fr::zero(); DEGREE + 1],
                |mut sum, local| {
                    for (sum, value) in sum.iter_mut().zip(local) {
                        *sum += value;
                    }
                    sum
                },
            );
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

pub fn carried_final(
    source_point: &[Fr],
    target_point: &[Fr],
    target_evaluation: Fr,
) -> Result<Fr, CarryError> {
    if source_point.len() != target_point.len() {
        return Err(CarryError::PointDimension {
            expected: source_point.len(),
            actual: target_point.len(),
        });
    }
    Ok(EqPolynomial::<Fr>::mle(source_point, target_point) * target_evaluation)
}
