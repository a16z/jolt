use jolt_field::{Fr, One, Ring, Zero};
use jolt_hyperkzg::{NoopVerifierObserver, VerifierObserver};
use jolt_poly::{BindingOrder, Polynomial, UnivariatePoly};
use jolt_sumcheck::prover::ProveRounds;
use jolt_sumcheck::SumcheckError;

use super::{AffineForm, StreamError, Term};

pub struct TermStageProver {
    coefficient: Polynomial<Fr>,
    factors: Vec<Polynomial<Fr>>,
    rounds: usize,
    claim: Fr,
}

impl TermStageProver {
    pub fn new(terms: &[Term], columns: &[Fr], packing: usize) -> Result<Self, StreamError> {
        if terms.is_empty() {
            return Err(StreamError::EmptyTensor);
        }
        let factor_count = terms
            .iter()
            .map(|term| term.factors.len())
            .max()
            .ok_or(StreamError::EmptyTensor)?;
        if factor_count == 0 || factor_count > 5 {
            return Err(StreamError::TensorArity {
                term: 0,
                expected: 5,
                actual: factor_count,
            });
        }
        let table_size = terms.len().next_power_of_two().max(2);
        let mut coefficients = vec![Fr::zero(); table_size];
        let mut factors = vec![vec![Fr::one(); table_size]; factor_count];
        let mut claim = Fr::zero();
        for (index, term) in terms.iter().enumerate() {
            coefficients[index] = term.coefficient;
            let mut value = term.coefficient;
            for (factor_index, values) in factors.iter_mut().enumerate() {
                let factor = term
                    .factors
                    .get(factor_index)
                    .map_or(Ok(Fr::one()), |form| evaluate_form(form, columns, packing))?;
                values[index] = factor;
                value *= factor;
            }
            claim += value;
        }
        Ok(Self {
            coefficient: Polynomial::new(coefficients),
            factors: factors.into_iter().map(Polynomial::new).collect(),
            rounds: table_size.trailing_zeros() as usize,
            claim,
        })
    }

    pub fn input_claim(&self) -> Fr {
        self.claim
    }

    pub fn factor_evaluations(&self) -> Result<Vec<Fr>, StreamError> {
        self.factors
            .iter()
            .map(|factor| {
                factor
                    .evals()
                    .first()
                    .copied()
                    .ok_or(StreamError::StageCount)
            })
            .collect()
    }

    pub fn coefficient_evaluation(&self) -> Option<Fr> {
        self.coefficient.evals().first().copied()
    }

    fn bind(&mut self, challenge: Fr) {
        self.coefficient
            .bind_with_order(challenge, BindingOrder::HighToLow);
        for factor in &mut self.factors {
            factor.bind_with_order(challenge, BindingOrder::HighToLow);
        }
    }
}

impl ProveRounds<Fr> for TermStageProver {
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
        let half = self.coefficient.len() / 2;
        let evaluations = (0..=6)
            .map(|value| Fr::from_u64(value as u64))
            .map(|x| {
                (0..half)
                    .map(|index| {
                        self.factors.iter().fold(
                            self.coefficient.sumcheck_round_eval_with_order(
                                index,
                                x,
                                BindingOrder::HighToLow,
                            ),
                            |product, factor| {
                                product
                                    * factor.sumcheck_round_eval_with_order(
                                        index,
                                        x,
                                        BindingOrder::HighToLow,
                                    )
                            },
                        )
                    })
                    .sum()
            })
            .collect::<Vec<_>>();
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

pub struct WeightedColumnReduction {
    values: Polynomial<Fr>,
    weights: Polynomial<Fr>,
    rounds: usize,
    claim: Fr,
}

impl WeightedColumnReduction {
    pub fn new(values: Vec<Fr>, weights: Vec<Fr>) -> Result<Self, StreamError> {
        if values.is_empty() || !values.len().is_power_of_two() || values.len() != weights.len() {
            return Err(StreamError::NoColumns);
        }
        let claim = values.iter().zip(&weights).map(|(&v, &w)| v * w).sum();
        let rounds = values.len().trailing_zeros() as usize;
        Ok(Self {
            values: Polynomial::new(values),
            weights: Polynomial::new(weights),
            rounds,
            claim,
        })
    }

    pub fn input_claim(&self) -> Fr {
        self.claim
    }

    pub fn value_evaluation(&self) -> Option<Fr> {
        self.values.evals().first().copied()
    }

    pub fn weight_evaluation(&self) -> Option<Fr> {
        self.weights.evals().first().copied()
    }

    fn bind(&mut self, challenge: Fr) {
        self.values
            .bind_with_order(challenge, BindingOrder::HighToLow);
        self.weights
            .bind_with_order(challenge, BindingOrder::HighToLow);
    }
}

impl ProveRounds<Fr> for WeightedColumnReduction {
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
        let half = self.values.len() / 2;
        let evaluations = [Fr::zero(), Fr::one(), Fr::from_u64(2)].map(|x| {
            (0..half)
                .map(|index| {
                    self.values
                        .sumcheck_round_eval_with_order(index, x, BindingOrder::HighToLow)
                        * self.weights.sumcheck_round_eval_with_order(
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

pub struct TermReduction {
    pub weights: Vec<Fr>,
    pub constants: Vec<Fr>,
}

impl TermReduction {
    pub fn claim(&self, evaluations: &[Fr], lambdas: &[Fr]) -> Result<Fr, StreamError> {
        self.claim_observed(evaluations, lambdas, &mut NoopVerifierObserver)
    }

    pub fn claim_observed<O: VerifierObserver>(
        &self,
        evaluations: &[Fr],
        lambdas: &[Fr],
        observer: &mut O,
    ) -> Result<Fr, StreamError> {
        if evaluations.len() != self.constants.len() || evaluations.len() != lambdas.len() {
            return Err(StreamError::StageMemberCount);
        }
        Ok(evaluations
            .iter()
            .zip(&self.constants)
            .zip(lambdas)
            .map(|((&evaluation, &constant), &lambda)| {
                observer.fr_mul(lambda, evaluation - constant)
            })
            .sum())
    }
}

pub fn coefficient_evaluation(terms: &[Term], point: &[Fr]) -> Result<Fr, StreamError> {
    coefficient_evaluation_observed(terms, point, &mut NoopVerifierObserver)
}

pub fn coefficient_evaluation_observed<O: VerifierObserver>(
    terms: &[Term],
    point: &[Fr],
    observer: &mut O,
) -> Result<Fr, StreamError> {
    let weights = eq_evaluations_observed(point, observer);
    if terms.len() > weights.len() {
        return Err(StreamError::StageMemberCount);
    }
    Ok(terms
        .iter()
        .zip(weights)
        .map(|(term, weight)| observer.fr_mul(term.coefficient, weight))
        .sum())
}

pub fn term_reduction(
    terms: &[Term],
    point: &[Fr],
    lambdas: &[Fr],
    columns: usize,
    packing: usize,
) -> Result<TermReduction, StreamError> {
    term_reduction_observed(
        terms,
        point,
        lambdas,
        columns,
        packing,
        &mut NoopVerifierObserver,
    )
}

pub fn term_reduction_observed<O: VerifierObserver>(
    terms: &[Term],
    point: &[Fr],
    lambdas: &[Fr],
    columns: usize,
    packing: usize,
    observer: &mut O,
) -> Result<TermReduction, StreamError> {
    let factor_count = terms
        .iter()
        .map(|term| term.factors.len())
        .max()
        .ok_or(StreamError::EmptyTensor)?;
    if factor_count != lambdas.len() {
        return Err(StreamError::StageMemberCount);
    }
    let term_weights = eq_evaluations_observed(point, observer);
    if terms.len() > term_weights.len() {
        return Err(StreamError::StageMemberCount);
    }
    let padding_constant: Fr = term_weights.iter().skip(terms.len()).copied().sum();
    let mut factor_constants = vec![Fr::zero(); factor_count];
    let mut factor_weights = vec![vec![Fr::zero(); columns]; factor_count];
    for (term, term_weight) in terms.iter().zip(term_weights) {
        for (factor, (factor_constant, factor_weights)) in factor_constants
            .iter_mut()
            .zip(&mut factor_weights)
            .enumerate()
        {
            let form = term.factors.get(factor);
            let constant = form.map_or(Fr::one(), |form| form.constant);
            *factor_constant += observer.fr_mul(term_weight, constant);
            if let Some(form) = form {
                for &(column, weight) in &form.weights {
                    let index = column.index(packing)?;
                    let target =
                        factor_weights
                            .get_mut(index)
                            .ok_or(StreamError::ColumnOutOfRange {
                                column: index,
                                columns,
                            })?;
                    *target += observer.fr_mul(term_weight, weight);
                }
            }
        }
    }
    for constant in &mut factor_constants {
        *constant += padding_constant;
    }
    let mut weights = vec![Fr::zero(); columns];
    for (factor_weights, &lambda) in factor_weights.iter().zip(lambdas) {
        for (weight, &factor_weight) in weights.iter_mut().zip(factor_weights) {
            *weight += observer.fr_mul(lambda, factor_weight);
        }
    }
    Ok(TermReduction {
        weights,
        constants: factor_constants,
    })
}

pub fn multilinear_evaluation_observed<O: VerifierObserver>(
    values: &[Fr],
    point: &[Fr],
    observer: &mut O,
) -> Result<Fr, StreamError> {
    let eq = eq_evaluations_observed(point, observer);
    if values.len() != eq.len() {
        return Err(StreamError::PointDimension {
            expected: values.len().trailing_zeros() as usize,
            actual: point.len(),
        });
    }
    Ok(values
        .iter()
        .zip(eq)
        .map(|(&value, weight)| observer.fr_mul(value, weight))
        .sum())
}

fn evaluate_form(form: &AffineForm, columns: &[Fr], packing: usize) -> Result<Fr, StreamError> {
    form.weights
        .iter()
        .try_fold(form.constant, |value, &(column, weight)| {
            let index = column.index(packing)?;
            let column = columns
                .get(index)
                .copied()
                .ok_or(StreamError::ColumnOutOfRange {
                    column: index,
                    columns: columns.len(),
                })?;
            Ok(value + weight * column)
        })
}

fn eq_evaluations_observed<O: VerifierObserver>(point: &[Fr], observer: &mut O) -> Vec<Fr> {
    let mut evaluations = vec![Fr::one()];
    for &challenge in point {
        let mut next = Vec::with_capacity(2 * evaluations.len());
        for value in evaluations {
            next.push(observer.fr_mul(value, Fr::one() - challenge));
            next.push(observer.fr_mul(value, challenge));
        }
        evaluations = next;
    }
    evaluations
}
