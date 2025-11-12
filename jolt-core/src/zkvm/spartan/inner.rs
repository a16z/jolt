use std::sync::Arc;

use allocative::Allocative;
use tracing::{span, Level};

use crate::field::JoltField;
use crate::poly::dense_mlpoly::DensePolynomial;
use crate::poly::multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding};
use crate::poly::opening_proof::{
    OpeningAccumulator, OpeningId, ProverOpeningAccumulator, SumcheckId, VerifierOpeningAccumulator,
};
use crate::subprotocols::sumcheck_prover::SumcheckInstanceProver;
use crate::subprotocols::sumcheck_verifier::SumcheckInstanceVerifier;
use crate::transcripts::Transcript;
use crate::utils::math::Math;
use crate::zkvm::r1cs::inputs::{JoltR1CSInputs, ALL_R1CS_INPUTS};
use crate::zkvm::r1cs::key::UniformSpartanKey;
use crate::zkvm::witness::VirtualPolynomial;

use rayon::prelude::*;

/// Degree bound of the sumcheck round polynomials in [`InnerSumcheckVerifier`].
const DEGREE_BOUND: usize = 2;

/// Sumcheck prover for [`InnerSumcheckVerifier`].
#[derive(Allocative)]
pub struct InnerSumcheckProver<F: JoltField> {
    poly_abc_small: MultilinearPolynomial<F>,
    poly_z: MultilinearPolynomial<F>,
    #[allocative(skip)]
    params: InnerSumcheckParams<F>,
}

impl<F: JoltField> InnerSumcheckProver<F> {
    #[tracing::instrument(skip_all, name = "InnerSumcheckProver::gen")]
    pub fn gen(
        opening_accumulator: &ProverOpeningAccumulator<F>,
        key: Arc<UniformSpartanKey<F>>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let num_vars_uniform = key.num_vars_uniform_padded();
        let num_cycles = key.num_steps;
        let num_cycles_bits = num_cycles.ilog2() as usize;

        let params = InnerSumcheckParams::new(transcript);

        // Get opening_point and claims from accumulator (Az, Bz all have the same point)
        let (outer_sumcheck_r, _) = opening_accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::SpartanAz, SumcheckId::SpartanOuter);

        let (_r_cycle, rx_var) = outer_sumcheck_r.r.split_at(num_cycles_bits);

        // Evaluate A_small, B_small combined with RLC at point rx_var
        let poly_abc_small =
            DensePolynomial::new(key.evaluate_small_matrix_rlc(rx_var, params.gamma));

        let span = span!(Level::INFO, "binding_z_second_sumcheck");
        let _guard = span.enter();

        // Bind witness polynomials z at point r_cycle
        let mut bind_z = vec![F::zero(); num_vars_uniform];

        ALL_R1CS_INPUTS
            .into_iter()
            .zip(bind_z.iter_mut())
            .for_each(|(r1cs_input, dest)| {
                let key = OpeningId::from(&r1cs_input);
                let (_, claim) = opening_accumulator
                    .openings
                    .get(&key)
                    .expect("Missing opening claim for expected OpeningId in bind_z");
                *dest = *claim;
            });

        // Set the constant value at the appropriate position
        let const_col = JoltR1CSInputs::num_inputs();
        if const_col < num_vars_uniform {
            bind_z[const_col] = F::one();
        }

        drop(_guard);
        drop(span);

        let poly_z = DensePolynomial::new(bind_z);
        assert_eq!(poly_z.len(), poly_abc_small.len());

        Self {
            poly_abc_small: MultilinearPolynomial::LargeScalars(poly_abc_small),
            poly_z: MultilinearPolynomial::LargeScalars(poly_z),
            params,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for InnerSumcheckProver<F> {
    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn num_rounds(&self) -> usize {
        self.poly_abc_small.original_len().log_2()
    }

    fn input_claim(&self, accumulator: &ProverOpeningAccumulator<F>) -> F {
        self.params.input_claim(accumulator)
    }

    #[tracing::instrument(skip_all, name = "InnerSumcheckProver::compute_prover_message")]
    fn compute_prover_message(&mut self, _round: usize, _previous_claim: F) -> Vec<F> {
        let univariate_poly_evals: [F; DEGREE_BOUND] = (0..self.poly_abc_small.len() / 2)
            .into_par_iter()
            .map(|i| {
                let abc_evals = self
                    .poly_abc_small
                    .sumcheck_evals_array::<DEGREE_BOUND>(i, BindingOrder::LowToHigh);
                let z_evals = self
                    .poly_z
                    .sumcheck_evals_array::<DEGREE_BOUND>(i, BindingOrder::LowToHigh);

                [
                    abc_evals[0] * z_evals[0], // eval at 0
                    abc_evals[1] * z_evals[1], // eval at 2
                ]
            })
            .reduce(
                || [F::zero(); DEGREE_BOUND],
                |mut running, new| {
                    for i in 0..DEGREE_BOUND {
                        running[i] += new[i];
                    }
                    running
                },
            );

        univariate_poly_evals.into()
    }

    #[tracing::instrument(skip_all, name = "InnerSumcheckProver::bind")]
    fn bind(&mut self, r_j: F::Challenge, _round: usize) {
        // Bind both polynomials in parallel
        self.poly_abc_small
            .bind_parallel(r_j, BindingOrder::LowToHigh);
        self.poly_z.bind_parallel(r_j, BindingOrder::LowToHigh);
    }

    fn cache_openings(
        &self,
        _accumulator: &mut ProverOpeningAccumulator<F>,
        _transcript: &mut T,
        _sumcheck_challenges: &[F::Challenge],
    ) {
        // Nothing to cache
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut allocative::FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

/// Inner Spartan sumcheck
///
/// This instance proves, for a fixed row opening point `rx_constr = [r_stream, r0]` supplied by the
/// outer stage, the column product identity over the uniform variables
/// `u` of length `log2(num_vars_uniform_padded)`.
///
/// Initial claim (start of sumcheck):
///
///   Σ_u ( A_small(rx_constr, u) + γ·B_small(rx_constr, u) ) · z(u)
///   = Az_claim + γ·Bz_claim.
///
/// where z(u) = u-th-R1CS-input-MLE(r_cycle_stage_1) (with a possible final u for constant term).
/// After `m = log2(num_vars_uniform_padded)` rounds with bound point `r ∈ F^m`, the
/// sumcheck output claim must equal
///
///   ( A_small(rx_constr, r) + γ·B_small(rx_constr, r) ) · z(r).
///
/// Final check (verifier): compute
///   - eval_a = evaluate_uniform_a_at_point(rx_constr, r),
///   - eval_b = evaluate_uniform_b_at_point(rx_constr, r),
///   - eval_z = evaluate_z_mle_with_segment_evals(segment_evals, r, true),
///     where `segment_evals` are the cached witness openings at `r_cycle` from the outer stage.
///
/// Then `expected = (eval_a + γ·eval_b) · eval_z`, and accept iff output_claim == expected.
pub struct InnerSumcheckVerifier<'a, F: JoltField> {
    params: InnerSumcheckParams<F>,
    key: &'a UniformSpartanKey<F>,
}

impl<'a, F: JoltField> InnerSumcheckVerifier<'a, F> {
    pub fn new(key: &'a UniformSpartanKey<F>, transcript: &mut impl Transcript) -> Self {
        let params = InnerSumcheckParams::new(transcript);
        Self { params, key }
    }
}

impl<'a, F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T>
    for InnerSumcheckVerifier<'a, F>
{
    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn num_rounds(&self) -> usize {
        self.key.num_vars_uniform_padded().log_2()
    }

    fn input_claim(&self, accumulator: &VerifierOpeningAccumulator<F>) -> F {
        self.params.input_claim(accumulator)
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let r = sumcheck_challenges
            .iter()
            .cloned()
            .rev()
            .collect::<Vec<_>>();

        // Get rx_var from the outer sumcheck opening point in accumulator
        let (outer_sumcheck_opening, _) = accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::SpartanAz, SumcheckId::SpartanOuter);
        let num_cycles_bits = self.key.num_steps.log_2();
        let (_r_cycle, rx_var) = outer_sumcheck_opening.r.split_at(num_cycles_bits);

        // assert rx var is of length 2
        assert_eq!(rx_var.len(), 2);

        // Pull claimed witness evaluations from the accumulator
        let claimed_witness_evals: Vec<F> = ALL_R1CS_INPUTS
            .into_iter()
            .map(|r1cs_input| {
                let (_, claim) = accumulator
                    .openings
                    .get(&OpeningId::from(&r1cs_input))
                    .unwrap();
                *claim
            })
            .collect();

        // The verifier needs to compute:
        // (A_small(rx_var, r) + gamma * B_small(rx_var, r)) * z(r)

        // Evaluate uniform matrices A_small and B_small at point (rx_var, r)
        let eval_a = self.key.evaluate_uniform_a_at_point(rx_var, &r);
        let eval_b = self.key.evaluate_uniform_b_at_point(rx_var, &r);

        let left_expected = eval_a + self.params.gamma * eval_b;

        // Evaluate z(ry)
        let eval_z = self
            .key
            .evaluate_z_mle_with_segment_evals(&claimed_witness_evals, &r, true);
        left_expected * eval_z
    }

    fn cache_openings(
        &self,
        _accumulator: &mut VerifierOpeningAccumulator<F>,
        _transcript: &mut T,
        _sumcheck_challenges: &[<F as JoltField>::Challenge],
    ) {
        // Nothing to cache
    }
}

struct InnerSumcheckParams<F: JoltField> {
    gamma: F::Challenge,
}

impl<F: JoltField> InnerSumcheckParams<F> {
    fn new(transcript: &mut impl Transcript) -> Self {
        let gamma = transcript.challenge_scalar_optimized::<F>();
        Self { gamma }
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        let (_, claim_Az) = accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::SpartanAz, SumcheckId::SpartanOuter);
        let (_, claim_Bz) = accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::SpartanBz, SumcheckId::SpartanOuter);
        claim_Az + self.gamma * claim_Bz
    }
}
