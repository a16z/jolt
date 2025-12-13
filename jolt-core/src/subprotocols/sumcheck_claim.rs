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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OpeningRef {
    Committed(CommittedPolynomial),
    Virtual(VirtualPolynomial),
}

impl OpeningRef {
    fn get_cached_opening<F: JoltField>(&self, acc: &impl OpeningAccumulator<F>, sumcheck_id: SumcheckId) -> F {
        match self {
            Self::Committed(poly) => acc.get_committed_polynomial_opening(*poly, sumcheck_id),
            Self::Virtual(poly) => acc.get_virtual_polynomial_opening(*poly, sumcheck_id),
        }.1
    }

    fn get_point<F: JoltField>(&self, acc: &impl OpeningAccumulator<F>, sumcheck_id: SumcheckId) -> OpeningPoint<BIG_ENDIAN, F> {
        match self {
            Self::Committed(poly) => acc.get_committed_polynomial_opening(*poly, sumcheck_id),
            Self::Virtual(poly) => acc.get_virtual_polynomial_opening(*poly, sumcheck_id),
        }.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BatchingPolynomial {
    Cycle(OpeningRef),
    NextCycle(OpeningRef),
    LtCycle(OpeningRef),
    Address(OpeningRef),
    Identity,
}

#[derive(Debug, Clone)]
pub enum ClaimExpr<F> {
    Val(F),
    Var(OpeningRef),
    Add(Box<ClaimExpr<F>>, Box<ClaimExpr<F>>),
    Mul(Box<ClaimExpr<F>>, Box<ClaimExpr<F>>),
    Sub(Box<ClaimExpr<F>>, Box<ClaimExpr<F>>),
}

impl<F: JoltField> ClaimExpr<F> {
    pub fn committed_var(poly: CommittedPolynomial) -> Self {
        Self::Var(OpeningRef::Committed(poly))
    }

    pub fn virtual_var(poly: VirtualPolynomial) -> Self {
        Self::Var(OpeningRef::Virtual(poly))
    }

    fn evaluate(&self, acc: &impl OpeningAccumulator<F>, sumcheck_id: SumcheckId) -> F {
        match self {
            ClaimExpr::Val(f) => *f,
            ClaimExpr::Var(opening_ref) => opening_ref.get_cached_opening(acc, sumcheck_id),
            ClaimExpr::Add(e1, e2) => {
                F::add(e1.evaluate(acc, sumcheck_id), e2.evaluate(acc, sumcheck_id))
            }
            ClaimExpr::Mul(e1, e2) => {
                F::mul(e1.evaluate(acc, sumcheck_id), e2.evaluate(acc, sumcheck_id))
            }
            ClaimExpr::Sub(e1, e2) => {
                F::sub(e1.evaluate(acc, sumcheck_id), e2.evaluate(acc, sumcheck_id))
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
            ClaimExpr::Var(opening_ref) => Some(opening_ref.get_point(acc, sumcheck_id)),
            ClaimExpr::Add(e1, e2) | ClaimExpr::Mul(e1, e2) | ClaimExpr::Sub(e1, e2) => e1
                .get_first_opening_point(acc, sumcheck_id)
                .or(e2.get_first_opening_point(acc, sumcheck_id)),
        }
    }
}

impl<F: JoltField> From<F> for ClaimExpr<F> {
    fn from(value: F) -> Self {
        Self::Val(value)
    }
}

impl<F: JoltField> From<VirtualPolynomial> for ClaimExpr<F> {
    fn from(value: VirtualPolynomial) -> Self {
        Self::virtual_var(value)
    }
}

impl<F: JoltField> From<CommittedPolynomial> for ClaimExpr<F> {
    fn from(value: CommittedPolynomial) -> Self {
        Self::committed_var(value)
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
#[derive(Debug, Clone)]
pub struct Claim<F: JoltField> {
    pub input_sumcheck_id: SumcheckId,
    pub input_claim_expr: ClaimExpr<F>,
    pub expected_output_claim_expr: ClaimExpr<F>,
    pub is_offset: bool,
}

#[derive(Debug, Clone)]
pub struct InputOutputClaims<F: JoltField> {
    pub claims: Vec<Claim<F>>,
    pub output_sumcheck_id: SumcheckId,
}

impl<F: JoltField> InputOutputClaims<F> {
    pub fn input_claim(&self, gamma_pows: &[F], acc: &impl OpeningAccumulator<F>) -> F {
        self.claims
            .iter()
            .zip(gamma_pows)
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
        gamma_pows: &[F],
        acc: &impl OpeningAccumulator<F>,
    ) -> F {
        let mut eq_eval_cache: HashMap<SumcheckId, F> = HashMap::new();
        let mut eq_plus_one_eval_cache: HashMap<SumcheckId, F> = HashMap::new();

        self.claims
            .iter()
            .zip(gamma_pows)
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
    fn input_output_claims() -> InputOutputClaims<F>;
}
