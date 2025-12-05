use std::{
    collections::HashMap,
    ops::{Add, Mul, Sub},
};

use crate::{
    field::JoltField,
    poly::{
        eq_plus_one_poly::EqPlusOnePolynomial,
        eq_poly::EqPolynomial,
        opening_proof::{OpeningAccumulator, OpeningPoint, SumcheckId, BIG_ENDIAN},
    },
    zkvm::witness::{CommittedPolynomial, VirtualPolynomial},
};

#[derive(Debug, Clone)]
pub enum ClaimExpr<F> {
    Val(F),
    Add(Box<ClaimExpr<F>>, Box<ClaimExpr<F>>),
    Mul(Box<ClaimExpr<F>>, Box<ClaimExpr<F>>),
    Sub(Box<ClaimExpr<F>>, Box<ClaimExpr<F>>),
    /// Corresponds to an "opening"
    CommittedVar(CommittedPolynomial),
    VirtualVar(VirtualPolynomial),
}

impl<F: JoltField> ClaimExpr<F> {
    fn evaluate(&self, acc: &impl OpeningAccumulator<F>, sumcheck_id: SumcheckId) -> F {
        match self {
            ClaimExpr::Val(f) => *f,
            ClaimExpr::Add(e1, e2) => {
                F::add(e1.evaluate(acc, sumcheck_id), e2.evaluate(acc, sumcheck_id))
            }
            ClaimExpr::Mul(e1, e2) => {
                F::mul(e1.evaluate(acc, sumcheck_id), e2.evaluate(acc, sumcheck_id))
            }
            ClaimExpr::Sub(e1, e2) => {
                F::sub(e1.evaluate(acc, sumcheck_id), e2.evaluate(acc, sumcheck_id))
            }
            ClaimExpr::CommittedVar(committed_polynomial) => {
                acc.get_committed_polynomial_opening(*committed_polynomial, sumcheck_id)
                    .1
            }
            ClaimExpr::VirtualVar(virtual_polynomial) => {
                acc.get_virtual_polynomial_opening(*virtual_polynomial, sumcheck_id)
                    .1
            }
        }
    }

    /// We use this to get the `r_cycle_stage` opening point from an input claim to use in the
    /// corresponding output claim. We need this because we need to know the polynomial ID and
    /// whether to call `get_committed_polynomial_opening` or `get_virtual_polynomial_opening`.
    // TODO: Better way to do this?
    fn get_first_opening_point(
        &self,
        acc: &impl OpeningAccumulator<F>,
        sumcheck_id: SumcheckId,
    ) -> Option<OpeningPoint<BIG_ENDIAN, F>> {
        match self {
            ClaimExpr::Val(_) => None,
            ClaimExpr::Add(e1, e2) | ClaimExpr::Mul(e1, e2) | ClaimExpr::Sub(e1, e2) => e1
                .get_first_opening_point(acc, sumcheck_id)
                .or(e2.get_first_opening_point(acc, sumcheck_id)),
            ClaimExpr::CommittedVar(committed_polynomial) => Some(
                acc.get_committed_polynomial_opening(*committed_polynomial, sumcheck_id)
                    .0,
            ),
            ClaimExpr::VirtualVar(virtual_polynomial) => Some(
                acc.get_virtual_polynomial_opening(*virtual_polynomial, sumcheck_id)
                    .0,
            ),
        }
    }
}

impl<F: JoltField> From<F> for ClaimExpr<F> {
    fn from(value: F) -> Self {
        Self::Val(value)
    }
}

impl<F> From<VirtualPolynomial> for ClaimExpr<F> {
    fn from(value: VirtualPolynomial) -> Self {
        Self::VirtualVar(value)
    }
}

impl<F> From<CommittedPolynomial> for ClaimExpr<F> {
    fn from(value: CommittedPolynomial) -> Self {
        Self::CommittedVar(value)
    }
}

impl<F> Add for ClaimExpr<F> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::Add(Box::new(self), Box::new(rhs))
    }
}

impl<F> Mul for ClaimExpr<F> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self::Mul(Box::new(self), Box::new(rhs))
    }
}

impl<F> Sub for ClaimExpr<F> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::Sub(Box::new(self), Box::new(rhs))
    }
}

/// Interpreted as `input_claim - output_claim = 0`.
pub struct Claim<F: JoltField> {
    pub input_sumcheck_id: SumcheckId,
    pub input_claim_expr: ClaimExpr<F>,
    pub expected_output_claim_expr: ClaimExpr<F>,
    pub is_offset: bool,
}

pub struct InputOutputClaims<F: JoltField> {
    pub claims: Vec<Claim<F>>,
    pub output_sumcheck_id: SumcheckId,
    pub gamma_pows: Vec<F>,
}

impl<F: JoltField> InputOutputClaims<F> {
    pub fn new_from_gamma(claims: Vec<Claim<F>>, output_sumcheck_id: SumcheckId, gamma: F) -> Self {
        let gamma_pows: Vec<F> = std::iter::successors(Some(F::one()), |prev| Some(*prev * gamma))
            .take(claims.len())
            .collect();

        Self {
            claims,
            output_sumcheck_id,
            gamma_pows,
        }
    }

    pub fn input_claim(&self, acc: &impl OpeningAccumulator<F>) -> F {
        self.claims
            .iter()
            .zip(&self.gamma_pows)
            .map(|(claim, gamma_pow)| {
                let claim_eval = claim
                    .input_claim_expr
                    .evaluate(acc, claim.input_sumcheck_id);
                *gamma_pow * claim_eval
            })
            .sum()
    }

    pub fn expected_output_claim(
        &self,
        r: &OpeningPoint<BIG_ENDIAN, F>,
        acc: &impl OpeningAccumulator<F>,
    ) -> F {
        let mut eq_eval_cache: HashMap<SumcheckId, F> = HashMap::new();
        let mut eq_plus_one_eval_cache: HashMap<SumcheckId, F> = HashMap::new();

        self.claims
            .iter()
            .zip(&self.gamma_pows)
            .map(|(claim, gamma_pow)| {
                let eq_eval = if claim.is_offset {
                    eq_plus_one_eval_cache
                        .entry(claim.input_sumcheck_id)
                        .or_insert_with(|| {
                            let opening_point = claim
                                .input_claim_expr
                                .get_first_opening_point(acc, claim.input_sumcheck_id)
                                .unwrap();
                            let r_cycle_stage = if r.len() < opening_point.len() {
                                opening_point.split_at(opening_point.len() - 1 - r.len()).1
                            } else {
                                opening_point
                            };
                            EqPlusOnePolynomial::new(r_cycle_stage.r).evaluate(&r.r)
                        })
                } else {
                    eq_eval_cache
                        .entry(claim.input_sumcheck_id)
                        .or_insert_with(|| {
                            let opening_point = claim
                                .input_claim_expr
                                .get_first_opening_point(acc, claim.input_sumcheck_id)
                                .unwrap();
                            let r_cycle_stage = if r.len() < opening_point.len() {
                                opening_point.split_at(opening_point.len() - 1 - r.len()).1
                            } else {
                                opening_point
                            };
                            EqPolynomial::mle_endian(&r_cycle_stage, r)
                        })
                };
                let claim_eval = claim
                    .expected_output_claim_expr
                    .evaluate(acc, self.output_sumcheck_id);
                *gamma_pow * *eq_eval * claim_eval
            })
            .sum()
    }
}

pub trait SumcheckFrontend<F: JoltField> {
    fn input_output_claims(&self) -> InputOutputClaims<F>;
}
