use std::{
    collections::HashMap,
    ops::{Add, Mul, Sub},
};

use crate::{
    field::JoltField,
    poly::{
        eq_plus_one_poly::EqPlusOnePolynomial,
        eq_poly::EqPolynomial,
        identity_poly::UnmapRamAddressPolynomial,
        multilinear_polynomial::PolynomialEvaluation as _,
        opening_proof::{OpeningAccumulator, OpeningPoint, SumcheckId, BIG_ENDIAN},
    },
    zkvm::witness::{CommittedPolynomial, VirtualPolynomial},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpeningRef {
    Committed(CommittedPolynomial),
    Virtual(VirtualPolynomial),
}

impl OpeningRef {
    fn get_cached_opening<F: JoltField>(
        &self,
        sumcheck_id: SumcheckId,
        acc: &impl OpeningAccumulator<F>,
    ) -> F {
        match self {
            Self::Committed(poly) => acc.get_committed_polynomial_opening(*poly, sumcheck_id),
            Self::Virtual(poly) => acc.get_virtual_polynomial_opening(*poly, sumcheck_id),
        }
        .1
    }

    fn get_point<F: JoltField>(
        &self,
        sumcheck_id: SumcheckId,
        acc: &impl OpeningAccumulator<F>,
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        match self {
            Self::Committed(poly) => acc.get_committed_polynomial_opening(*poly, sumcheck_id),
            Self::Virtual(poly) => acc.get_virtual_polynomial_opening(*poly, sumcheck_id),
        }
        .0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChallengePart {
    Address,
    Cycle,
    Full,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CachedPointRef {
    pub opening: OpeningRef,
    pub sumcheck: SumcheckId,
    pub part: ChallengePart,
    pub reverse: bool,
}

impl CachedPointRef {
    fn get_point<F: JoltField>(
        &self,
        n_vars: usize,
        acc: &impl OpeningAccumulator<F>,
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        let point = self.opening.get_point(self.sumcheck, acc);
        let point_part = match self.part {
            // Take address part to be initial elements of challenge point
            ChallengePart::Address if point.len() > n_vars => point.split_at(n_vars).0,
            // Take cycle part to be final elements of challenge point
            ChallengePart::Cycle if point.len() > n_vars => {
                point.split_at(point.len() - n_vars).1
            }
            _ => point,
        };
        if self.reverse {
            OpeningPoint::<BIG_ENDIAN, F> {
                r: point_part.r.into_iter().rev().collect(),
            }
        } else {
            point_part
        }
    }
}

/// These are parameters needed to evaluate some batching polynomials, but not all. They're
/// available in only some sumcheck verifier instances, so we allow them to be options.
// TODO: Find a better way to do this
#[derive(Debug, Clone)]
pub struct BatchingEvaluationParams {
    ram_k: Option<usize>,
    ram_start_address: Option<u64>,
}

impl BatchingEvaluationParams {
    pub fn new(ram_k: usize, ram_start_address: u64) -> Self {
        BatchingEvaluationParams {
            ram_k: Some(ram_k),
            ram_start_address: Some(ram_start_address),
        }
    }

    pub fn empty() -> Self {
        BatchingEvaluationParams {
            ram_k: None,
            ram_start_address: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BatchingPolynomial {
    Eq(CachedPointRef),
    EqPlusOne(CachedPointRef),
    Lt(CachedPointRef),
    Identity,
    // TODO: We could handle this using Identity, but we would need to add binary ops and constants
    // within this struct.
    UnmapRamAddress,
    NoBatching,
}

impl BatchingPolynomial {
    fn evaluate<F: JoltField>(
        &self,
        eval_params: &BatchingEvaluationParams,
        r: &OpeningPoint<BIG_ENDIAN, F>,
        acc: &impl OpeningAccumulator<F>,
    ) -> F {
        match self {
            BatchingPolynomial::Eq(point_ref) => {
                let tau = point_ref.get_point(r.len(), acc);
                EqPolynomial::mle_endian(&tau, r)
            }
            BatchingPolynomial::EqPlusOne(point_ref) => {
                let tau = point_ref.get_point(r.len(), acc);
                EqPlusOnePolynomial::new(tau.r).evaluate(&r.r)
            }
            BatchingPolynomial::Lt(_opening_ref) => todo!(),
            BatchingPolynomial::Identity => todo!(),
            BatchingPolynomial::UnmapRamAddress => UnmapRamAddressPolynomial::<F>::new(
                eval_params.ram_k.unwrap(),
                eval_params.ram_start_address.unwrap(),
            )
            .evaluate(&r.r),
            BatchingPolynomial::NoBatching => F::one(),
        }
    }
}

#[derive(Debug, Clone)]
pub enum ClaimExpr<F> {
    Constant(F),
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

    fn evaluate(&self, sumcheck_id: SumcheckId, acc: &impl OpeningAccumulator<F>) -> F {
        match self {
            ClaimExpr::Constant(f) => *f,
            ClaimExpr::Var(opening_ref) => opening_ref.get_cached_opening(sumcheck_id, acc),
            ClaimExpr::Add(e1, e2) => {
                F::add(e1.evaluate(sumcheck_id, acc), e2.evaluate(sumcheck_id, acc))
            }
            ClaimExpr::Mul(e1, e2) => {
                F::mul(e1.evaluate(sumcheck_id, acc), e2.evaluate(sumcheck_id, acc))
            }
            ClaimExpr::Sub(e1, e2) => {
                F::sub(e1.evaluate(sumcheck_id, acc), e2.evaluate(sumcheck_id, acc))
            }
        }
    }
}

impl<F: JoltField> From<F> for ClaimExpr<F> {
    fn from(value: F) -> Self {
        Self::Constant(value)
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
    pub batching_poly: BatchingPolynomial,
    pub expected_output_claim_expr: ClaimExpr<F>,
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
                    .evaluate(claim.input_sumcheck_id, acc);
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
        let eval_params = BatchingEvaluationParams::empty();
        self.expected_output_claim_with_batching_parameters(&eval_params, r, gamma_pows, acc)
    }

    pub fn expected_output_claim_with_batching_parameters(
        &self,
        eval_params: &BatchingEvaluationParams,
        r: &OpeningPoint<BIG_ENDIAN, F>,
        gamma_pows: &[F],
        acc: &impl OpeningAccumulator<F>,
    ) -> F {
        let mut batching_poly_eval_cache: HashMap<BatchingPolynomial, F> = HashMap::new();

        self.claims
            .iter()
            .zip(gamma_pows)
            .map(|(claim, gamma_pow)| {
                let batching_poly_eval = batching_poly_eval_cache
                    .entry(claim.batching_poly)
                    .or_insert_with(|| claim.batching_poly.evaluate(eval_params, r, acc));
                let claim_eval = claim
                    .expected_output_claim_expr
                    .evaluate(self.output_sumcheck_id, acc);
                *gamma_pow * *batching_poly_eval * claim_eval
            })
            .sum()
    }
}

pub trait SumcheckFrontend<F: JoltField> {
    fn input_output_claims() -> InputOutputClaims<F>;
}
