use jolt_field::{Fr, One, Ring, Zero};
use jolt_hyperkzg::{NoopVerifierObserver, VerifierObserver};
use jolt_poly::{BindingOrder, EqPolynomial, Polynomial, UnivariatePoly};
use jolt_sumcheck::prover::ProveRounds;
use jolt_sumcheck::SumcheckError;

use super::StreamError;

pub struct ColumnReduction {
    polynomial: Polynomial<Fr>,
    eq: Polynomial<Fr>,
    rounds: usize,
    claim: Fr,
}

impl ColumnReduction {
    pub fn new(values: Vec<Fr>, column: usize) -> Result<Self, StreamError> {
        if values.is_empty() || !values.len().is_power_of_two() {
            return Err(StreamError::NoColumns);
        }
        let claim = *values.get(column).ok_or(StreamError::ColumnOutOfRange {
            column,
            columns: values.len(),
        })?;
        let rounds = values.len().trailing_zeros() as usize;
        Ok(Self {
            polynomial: Polynomial::new(values),
            eq: Polynomial::new(EqPolynomial::<Fr>::evals(
                &boolean_point(column, rounds),
                None,
            )),
            rounds,
            claim,
        })
    }

    pub fn input_claim(&self) -> Fr {
        self.claim
    }

    pub fn final_evaluation(&self) -> Fr {
        self.polynomial.evals()[0]
    }

    pub fn expected_final(
        column_count: usize,
        column: usize,
        point: &[Fr],
        evaluation: Fr,
    ) -> Result<Fr, StreamError> {
        Self::expected_final_observed(
            column_count,
            column,
            point,
            evaluation,
            &mut NoopVerifierObserver,
        )
    }

    pub fn expected_final_observed<O: VerifierObserver>(
        column_count: usize,
        column: usize,
        point: &[Fr],
        evaluation: Fr,
        observer: &mut O,
    ) -> Result<Fr, StreamError> {
        if column_count == 0 || !column_count.is_power_of_two() {
            return Err(StreamError::NoColumns);
        }
        let log_columns = column_count.trailing_zeros() as usize;
        if point.len() != log_columns {
            return Err(StreamError::PointDimension {
                expected: log_columns,
                actual: point.len(),
            });
        }
        if column >= column_count {
            return Err(StreamError::ColumnOutOfRange {
                column,
                columns: column_count,
            });
        }
        let selector = eq_mle_observed(&boolean_point(column, log_columns), point, observer);
        Ok(observer.fr_mul(selector, evaluation))
    }
}

impl ProveRounds<Fr> for ColumnReduction {
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
            self.polynomial
                .bind_with_order(challenge, BindingOrder::HighToLow);
            self.eq.bind_with_order(challenge, BindingOrder::HighToLow);
        }
        let half = self.polynomial.len() / 2;
        let evaluations = [Fr::zero(), Fr::one(), Fr::from_u64(2)].map(|x| {
            (0..half)
                .map(|index| {
                    self.polynomial.sumcheck_round_eval_with_order(
                        index,
                        x,
                        BindingOrder::HighToLow,
                    ) * self
                        .eq
                        .sumcheck_round_eval_with_order(index, x, BindingOrder::HighToLow)
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
        self.polynomial
            .bind_with_order(bind, BindingOrder::HighToLow);
        self.eq.bind_with_order(bind, BindingOrder::HighToLow);
        Ok(())
    }
}

fn boolean_point(index: usize, variables: usize) -> Vec<Fr> {
    (0..variables)
        .map(|bit| Fr::from_u64(((index >> (variables - bit - 1)) & 1) as u64))
        .collect()
}

fn eq_mle_observed<O: VerifierObserver>(left: &[Fr], right: &[Fr], observer: &mut O) -> Fr {
    let mut result = Fr::one();
    for (&left, &right) in left.iter().zip(right) {
        let both = observer.fr_mul(left, right);
        let neither = observer.fr_mul(Fr::one() - left, Fr::one() - right);
        result = observer.fr_mul(result, both + neither);
    }
    result
}
