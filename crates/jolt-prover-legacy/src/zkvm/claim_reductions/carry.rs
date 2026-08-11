//! Reduces the committed `Carry` polynomial's openings to a single final
//! opening for the stage-8 batch.
//!
//! `Carry` is opened at two sumcheck points (the product-virtualization cycle
//! point, as the right factor of `CarryUsed`, and the shift point, as the
//! shifted output of the `NextCarry` term). This instance batches them with a
//! third, public pair: the all-zeros point with claim 0, which enforces the
//! `carry_init` invariant `Carry(0) = 0` (a fresh CPU starts with zero carry).
//!
//! The relation proved is
//!   sum_t [eq(r_product, t) + gamma * eq(r_shift, t) + gamma^2 * eq(0, t)] * Carry(t)
//!     = claim_product + gamma * claim_shift + gamma^2 * 0
//! reducing all three to one opening of `Carry` at a fresh point.

use allocative::Allocative;
use rayon::prelude::*;
use std::sync::Arc;

use crate::field::JoltField;
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding};
#[cfg(feature = "zk")]
use crate::poly::opening_proof::OpeningId;
use crate::poly::opening_proof::{
    AbstractVerifierOpeningAccumulator, OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator,
    SumcheckId, BIG_ENDIAN, LITTLE_ENDIAN,
};
use crate::poly::unipoly::UniPoly;
#[cfg(feature = "zk")]
use crate::subprotocols::blindfold::{InputClaimConstraint, OutputClaimConstraint};
use crate::subprotocols::sumcheck_prover::SumcheckInstanceProver;
use crate::subprotocols::sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier};
use crate::transcripts::Transcript;
use crate::utils::math::Math;
use crate::zkvm::witness::CommittedPolynomial;
use tracer::instruction::Cycle;

const DEGREE_BOUND: usize = 2;

#[derive(Allocative, Clone)]
pub struct CarryClaimReductionSumcheckParams<F: JoltField> {
    /// gamma, gamma^2 for batching the three claims.
    pub gamma_powers: [F; 2],
    pub n_cycle_vars: usize,
    pub r_product: OpeningPoint<BIG_ENDIAN, F>,
    pub r_shift: OpeningPoint<BIG_ENDIAN, F>,
}

impl<F: JoltField> CarryClaimReductionSumcheckParams<F> {
    pub fn new(
        trace_len: usize,
        accumulator: &dyn OpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        // `challenge_scalar_powers` returns `[1, gamma, gamma^2, ...]`; the
        // batch uses gamma and gamma^2.
        let powers = transcript.challenge_scalar_powers(3);
        let gamma_powers = [powers[1], powers[2]];
        let (r_product, _) = accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::Carry,
            SumcheckId::SpartanProductVirtualization,
        );
        let (r_shift, _) = accumulator
            .get_committed_polynomial_opening(CommittedPolynomial::Carry, SumcheckId::SpartanShift);

        Self {
            gamma_powers,
            n_cycle_vars: trace_len.log_2(),
            r_product,
            r_shift,
        }
    }

    /// eq(r_product, r) + gamma * eq(r_shift, r) + gamma^2 * eq(0, r)
    fn combined_eq_at(&self, r: &[F::Challenge]) -> F {
        let [gamma, gamma_sqr] = self.gamma_powers;
        let eq_product: F = EqPolynomial::mle(&self.r_product.r, r);
        let eq_shift: F = EqPolynomial::mle(&self.r_shift.r, r);
        let eq_zero: F = EqPolynomial::zero_selector(r);
        eq_product + gamma * eq_shift + gamma_sqr * eq_zero
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for CarryClaimReductionSumcheckParams<F> {
    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        let [gamma, _] = self.gamma_powers;
        let (_, claim_product) = accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::Carry,
            SumcheckId::SpartanProductVirtualization,
        );
        let (_, claim_shift) = accumulator
            .get_committed_polynomial_opening(CommittedPolynomial::Carry, SumcheckId::SpartanShift);
        // The carry_init pair contributes gamma^2 * 0.
        claim_product + gamma * claim_shift
    }

    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn num_rounds(&self) -> usize {
        self.n_cycle_vars
    }

    fn normalize_opening_point(
        &self,
        challenges: &[<F as JoltField>::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
    }

    #[cfg(feature = "zk")]
    fn input_claim_constraint(&self) -> InputClaimConstraint {
        InputClaimConstraint::weighted_openings(&[
            OpeningId::committed(
                CommittedPolynomial::Carry,
                SumcheckId::SpartanProductVirtualization,
            ),
            OpeningId::committed(CommittedPolynomial::Carry, SumcheckId::SpartanShift),
        ])
    }

    #[cfg(feature = "zk")]
    fn input_constraint_challenge_values(&self, _: &dyn OpeningAccumulator<F>) -> Vec<F> {
        vec![self.gamma_powers[0]]
    }

    #[cfg(feature = "zk")]
    fn output_claim_constraint(&self) -> Option<OutputClaimConstraint> {
        Some(OutputClaimConstraint::all_weighted_openings(&[
            OpeningId::committed(CommittedPolynomial::Carry, SumcheckId::CarryClaimReduction),
        ]))
    }

    #[cfg(feature = "zk")]
    fn output_constraint_challenge_values(&self, sumcheck_challenges: &[F::Challenge]) -> Vec<F> {
        let opening_point = self.normalize_opening_point(sumcheck_challenges);
        vec![self.combined_eq_at(&opening_point.r)]
    }
}

#[derive(Allocative)]
pub struct CarryClaimReductionSumcheckProver<F: JoltField> {
    carry_poly: MultilinearPolynomial<F>,
    eq_poly: MultilinearPolynomial<F>,
    pub params: CarryClaimReductionSumcheckParams<F>,
}

impl<F: JoltField> CarryClaimReductionSumcheckProver<F> {
    #[tracing::instrument(skip_all, name = "CarryClaimReductionSumcheckProver::initialize")]
    pub fn initialize(
        params: CarryClaimReductionSumcheckParams<F>,
        trace: Arc<Vec<Cycle>>,
    ) -> Self {
        let [gamma, gamma_sqr] = params.gamma_powers;
        let t = 1 << params.n_cycle_vars;

        let carry_coeffs: Vec<u64> = trace
            .par_iter()
            .map(|cycle| cycle.carry())
            .chain(rayon::iter::repeatn(0u64, t - trace.len()))
            .collect();

        let (eq_product, eq_shift) = rayon::join(
            || EqPolynomial::evals(&params.r_product.r),
            || EqPolynomial::evals(&params.r_shift.r),
        );
        let mut eq_combined: Vec<F> = eq_product
            .into_par_iter()
            .zip(eq_shift.into_par_iter())
            .map(|(p, s)| p + gamma * s)
            .collect();
        // eq(0, t) is 1 only at t = 0.
        eq_combined[0] += gamma_sqr;

        Self {
            carry_poly: MultilinearPolynomial::from(carry_coeffs),
            eq_poly: MultilinearPolynomial::from(eq_combined),
            params,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T>
    for CarryClaimReductionSumcheckProver<F>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    #[tracing::instrument(skip_all, name = "CarryClaimReductionSumcheckProver::compute_message")]
    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        let half_n = self.carry_poly.len() / 2;
        let mut evals = [F::zero(); DEGREE_BOUND];
        for j in 0..half_n {
            let carry_evals = self
                .carry_poly
                .sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);
            let eq_evals = self
                .eq_poly
                .sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);
            for i in 0..DEGREE_BOUND {
                evals[i] += eq_evals[i] * carry_evals[i];
            }
        }
        UniPoly::from_evals_and_hint(previous_claim, &evals)
    }

    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        self.carry_poly.bind(r_j, BindingOrder::LowToHigh);
        self.eq_poly.bind(r_j, BindingOrder::LowToHigh);
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        sumcheck_challenges: &[<F as JoltField>::Challenge],
    ) {
        let opening_point = self.params.normalize_opening_point(sumcheck_challenges);
        accumulator.append_dense(
            CommittedPolynomial::Carry,
            SumcheckId::CarryClaimReduction,
            opening_point.r,
            self.carry_poly.final_sumcheck_claim(),
        );
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut allocative::FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

pub struct CarryClaimReductionSumcheckVerifier<F: JoltField> {
    params: CarryClaimReductionSumcheckParams<F>,
}

impl<F: JoltField> CarryClaimReductionSumcheckVerifier<F> {
    pub fn new<A: AbstractVerifierOpeningAccumulator<F>>(
        trace_len: usize,
        accumulator: &A,
        transcript: &mut impl Transcript,
    ) -> Self
    where
        A: OpeningAccumulator<F>,
    {
        Self {
            params: CarryClaimReductionSumcheckParams::new(trace_len, accumulator, transcript),
        }
    }
}

impl<F, T, A> SumcheckInstanceVerifier<F, T, A> for CarryClaimReductionSumcheckVerifier<F>
where
    F: JoltField,
    T: Transcript,
    A: AbstractVerifierOpeningAccumulator<F> + OpeningAccumulator<F>,
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(&self, accumulator: &A, sumcheck_challenges: &[F::Challenge]) -> F {
        let opening_point = self.params.normalize_opening_point(sumcheck_challenges);
        let (_, carry_claim) = accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::Carry,
            SumcheckId::CarryClaimReduction,
        );
        self.params.combined_eq_at(&opening_point.r) * carry_claim
    }

    fn cache_openings(
        &self,
        accumulator: &mut A,
        sumcheck_challenges: &[<F as JoltField>::Challenge],
    ) {
        let opening_point = self.params.normalize_opening_point(sumcheck_challenges);
        accumulator.append_dense(
            CommittedPolynomial::Carry,
            SumcheckId::CarryClaimReduction,
            opening_point.r,
        );
    }
}
