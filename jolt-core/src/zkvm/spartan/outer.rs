use allocative::Allocative;
#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use ark_ff::biginteger::S160;
use ark_std::Zero;
use rayon::prelude::*;
use tracer::instruction::Cycle;
use std::sync::Arc;

use crate::field::{JoltField, OptimizedMul};
use crate::poly::lagrange_poly::{LagrangeHelper, LagrangePolynomial};
use crate::poly::multilinear_polynomial::BindingOrder;
use crate::poly::split_eq_poly::GruenSplitEqPolynomial;
use crate::poly::unipoly::UniPoly;
use crate::subprotocols::sumcheck::{SingleSumcheck, SumcheckInstance, UniSkipSumcheckProof};
use crate::poly::eq_poly::EqPolynomial;
use crate::transcripts::{AppendToTranscript, Transcript};
use crate::utils::math::Math;
#[cfg(feature = "allocative")]
use crate::utils::profiling::print_data_structure_heap_usage;
// use crate::utils::thread::drop_in_background_thread;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::opening_proof::BIG_ENDIAN;
use crate::poly::opening_proof::{
    OpeningPoint, ProverOpeningAccumulator, SumcheckId, VerifierOpeningAccumulator,
};
use crate::utils::univariate_skip::accum::s160_to_field;
use crate::utils::univariate_skip::accum::{accs160_fmadd_s160, accs160_new, accs160_reduce};
use crate::utils::univariate_skip::{
    compute_az_r_group0, compute_az_r_group1, compute_bz_r_group0, compute_bz_r_group1,
    compute_cz_r_group1,
};
use crate::zkvm::dag::state_manager::{ProofData, ProofKeys, StateManager};
use crate::zkvm::r1cs::inputs::{
    compute_claimed_witness_evals, ALL_R1CS_INPUTS, COMMITTED_R1CS_INPUTS,
};
use crate::zkvm::r1cs::{
    constraints::{
        eval_az_first_group, eval_az_second_group, eval_bz_first_group, eval_bz_second_group,
        eval_cz_second_group, FIRST_ROUND_POLY_NUM_COEFFS, UNIVARIATE_SKIP_DEGREE,
        UNIVARIATE_SKIP_DOMAIN_SIZE, UNIVARIATE_SKIP_EXTENDED_DOMAIN_SIZE,
    },
    inputs::R1CSCycleInputs,
};
use crate::zkvm::witness::{CommittedPolynomial, VirtualPolynomial};
use crate::zkvm::JoltSharedPreprocessing;

#[derive(Default, Debug, Clone, Copy, PartialEq)]
pub struct SparseCoefficient<T> {
    pub(crate) index: usize,
    pub(crate) value: T,
}

impl<T> Allocative for SparseCoefficient<T> {
    fn visit<'a, 'b: 'a>(&self, _visitor: &'a mut allocative::Visitor<'b>) {}
}

impl<T> From<(usize, T)> for SparseCoefficient<T> {
    fn from(x: (usize, T)) -> Self {
        Self {
            index: x.0,
            value: x.1,
        }
    }
}

#[derive(Clone, Debug, Allocative, Default)]
pub struct SpartanInterleavedPoly<F: JoltField> {
    /// The bound coefficients for the Az, Bz, Cz polynomials.
    /// Will be populated in the streaming round (after SVO rounds)
    pub(crate) bound_coeffs: Vec<SparseCoefficient<F>>,

    binding_scratch_space: Vec<SparseCoefficient<F>>,
}

impl<F: JoltField> SpartanInterleavedPoly<F> {
    pub fn new() -> Self {
        Self {
            bound_coeffs: vec![],
            binding_scratch_space: vec![],
        }
    }
}

#[derive(Allocative)]
pub struct OuterSumcheck<F: JoltField> {
    // The claim to be used through the sumcheck rounds, initialized to zero
    claim: F,
    // The interleaved polynomial of Az/Bz/Cz bound evaluations, to be used through the sumcheck rounds
    interleaved_poly: SpartanInterleavedPoly<F>,
    // The split-eq polynomial to be used through the sumcheck rounds
    split_eq_poly: GruenSplitEqPolynomial<F>,
    // The last of tau vector, used for univariate skip
    tau_high: F::Challenge,
}

impl<F: JoltField> OuterSumcheck<F> {
    /// Initialize a Spartan outer sumcheck instance given the tau vector
    pub fn initialize(tau: &[F::Challenge]) -> Self {
        let tau_low = &tau[0..tau.len() - 1];
        Self {
            claim: F::zero(),
            interleaved_poly: SpartanInterleavedPoly::new(),
            split_eq_poly: GruenSplitEqPolynomial::new(tau_low, BindingOrder::LowToHigh),
            tau_high: tau[tau.len() - 1],
        }
    }

    /// Perform only the univariate skip round and return the state needed for remaining rounds
    #[tracing::instrument(skip_all, name = "OuterSumcheck::prove_univariate_skip_round")]
    pub fn prove_univariate_skip_round<ProofTranscript: Transcript>(
        preprocessing: &JoltSharedPreprocessing,
        trace: &[Cycle],
        tau: &[F::Challenge],
        transcript: &mut ProofTranscript,
    ) -> (
        UniPoly<F>,                // first polynomial
        F::Challenge,              // r[0]
        F,                         // claim after first round
        GruenSplitEqPolynomial<F>, // split_eq_poly
        SpartanInterleavedPoly<F>, // interleaved_poly
    ) {
        let mut outer_sumcheck = Self::initialize(tau);
        let extended_evals = outer_sumcheck.compute_univariate_skip_evals(preprocessing, trace);

        let mut r = Vec::new();
        let first_poly = outer_sumcheck.process_first_round_from_extended_evals(
            &extended_evals,
            transcript,
            &mut r,
        );

        (
            first_poly,
            r[0],
            outer_sumcheck.claim,
            outer_sumcheck.split_eq_poly,
            outer_sumcheck.interleaved_poly,
        )
    }

    /// Create a new prover instance from StateManager, extracting necessary data and challenges
    #[tracing::instrument(skip_all, name = "OuterSumcheck::new_prover")]
    pub fn new_prover<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
        num_rounds_x: usize,
    ) -> Self {
        let tau: Vec<F::Challenge> = state_manager
            .transcript
            .borrow_mut()
            .challenge_vector_optimized::<F>(num_rounds_x);
        Self::initialize(&tau)
    }

    /// Create a new verifier instance from StateManager, extracting necessary challenges  
    #[tracing::instrument(skip_all, name = "OuterSumcheck::new_verifier")]
    pub fn new_verifier<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
        num_rounds_x: usize,
    ) -> (Self, Vec<F::Challenge>) {
        let tau: Vec<F::Challenge> = state_manager
            .transcript
            .borrow_mut()
            .challenge_vector_optimized::<F>(num_rounds_x);
        (Self::initialize(&tau), tau)
    }

    #[tracing::instrument(skip_all, name = "OuterSumcheck::prove")]
    pub fn prove<ProofTranscript: Transcript>(
        preprocessing: &JoltSharedPreprocessing,
        trace: &[Cycle],
        num_rounds: usize,
        tau: &[F::Challenge],
        transcript: &mut ProofTranscript,
    ) -> (
        UniSkipSumcheckProof<F, ProofTranscript>,
        Vec<F::Challenge>,
        [F; 3],
    ) {
        // Assert that the number of rounds is equal to the number of cycle variables plus two
        // (one for univariate skip of degree ~13-15, and one for the streaming round)
        debug_assert_eq!(num_rounds, trace.len().next_power_of_two().log_2() + 2);

        let mut r = Vec::new();

        let mut outer_sumcheck = Self::initialize(tau);

        let extended_evals = outer_sumcheck.compute_univariate_skip_evals(preprocessing, trace);

        #[cfg(feature = "allocative")]
        {
            print_data_structure_heap_usage("OuterSumcheck", &outer_sumcheck);
            let mut flamegraph = FlameGraphBuilder::default();
            flamegraph.visit_root(&outer_sumcheck);
            write_flamegraph_svg(flamegraph, "outer_sumcheck_flamechart.svg");
        }

        // First round (univariate skip): build s1(Z) = lagrange_poly(Z) * t1(Z)
        let first_poly = outer_sumcheck.process_first_round_from_extended_evals(
            &extended_evals,
            transcript,
            &mut r,
        );

        // Use SumcheckInstance for the remaining rounds: 1 (streaming) + remaining cycle vars
        let remaining_rounds = num_rounds - 1;
        let mut instance = OuterRemainingSumcheck::new_prover(
            outer_sumcheck.claim,
            preprocessing,
            trace,
            outer_sumcheck.split_eq_poly,
            outer_sumcheck.interleaved_poly,
            r[0],
            remaining_rounds,
        );
        let (remaining_proof, r_rest) =
            SingleSumcheck::prove::<F, ProofTranscript>(&mut instance, None, transcript);
        r.extend(r_rest.into_iter());

        (
            UniSkipSumcheckProof::new(first_poly, remaining_proof.compressed_polys),
            r,
            instance.final_sumcheck_evals(),
        )
    }

    /// Prove the outer sumcheck using StateManager, handling proof storage and accumulator updates
    #[tracing::instrument(skip_all, name = "OuterSumcheck::prove_with_state_manager")]
    pub fn prove_with_state_manager<
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F>,
    >(
        self,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
        num_rounds_x: usize,
        tau: &[F::Challenge],
    ) {
        let (preprocessing, trace, _program_io, _final_memory_state) =
            state_manager.get_prover_data();

        let transcript = &mut *state_manager.transcript.borrow_mut();
        let (outer_sumcheck_proof, outer_sumcheck_r, outer_sumcheck_claims) =
            Self::prove::<ProofTranscript>(
                &preprocessing.shared,
                trace,
                num_rounds_x,
                tau,
                transcript,
            );

        let outer_sumcheck_r: Vec<F::Challenge> = outer_sumcheck_r.into_iter().rev().collect();

        ProofTranscript::append_scalars(
            &mut *state_manager.transcript.borrow_mut(),
            &outer_sumcheck_claims,
        );

        // Store Az, Bz, Cz claims with the outer sumcheck point
        let accumulator = state_manager.get_prover_accumulator();
        accumulator.borrow_mut().append_virtual(
            transcript,
            VirtualPolynomial::SpartanAz,
            SumcheckId::SpartanOuter,
            OpeningPoint::new(outer_sumcheck_r.clone()),
            outer_sumcheck_claims[0],
        );
        accumulator.borrow_mut().append_virtual(
            transcript,
            VirtualPolynomial::SpartanBz,
            SumcheckId::SpartanOuter,
            OpeningPoint::new(outer_sumcheck_r.clone()),
            outer_sumcheck_claims[1],
        );
        accumulator.borrow_mut().append_virtual(
            transcript,
            VirtualPolynomial::SpartanCz,
            SumcheckId::SpartanOuter,
            OpeningPoint::new(outer_sumcheck_r.clone()),
            outer_sumcheck_claims[2],
        );

        // Append the outer sumcheck proof to the state manager
        state_manager.proofs.borrow_mut().insert(
            ProofKeys::Stage1Sumcheck,
            ProofData::UniSkipSumcheckProof(outer_sumcheck_proof),
        );

        // Extract num_cycles from the key (we need a way to get this - for now use trace length)
        let num_cycles = trace.len().next_power_of_two();
        let num_cycles_bits = num_cycles.ilog2() as usize;

        let (r_cycle, _rx_var) = outer_sumcheck_r.split_at(num_cycles_bits);

        // Compute claimed witness evals at r_cycle via streaming (fast)
        let claimed_witness_evals = {
            use tracing::{span, Level};
            let _guard = span!(Level::INFO, "claimed_witness_evals_fast").entered();
            compute_claimed_witness_evals::<F>(&preprocessing.shared, trace, r_cycle)
        };

        // Only non-virtual (i.e. committed) polynomials' openings are
        // proven using the PCS opening proof, which we add for future opening proof here
        let committed_polys: Vec<_> = COMMITTED_R1CS_INPUTS
            .iter()
            .map(|input| CommittedPolynomial::try_from(input).ok().unwrap())
            .collect();
        let committed_poly_claims: Vec<_> = COMMITTED_R1CS_INPUTS
            .iter()
            .map(|input| claimed_witness_evals[input.to_index()])
            .collect();

        let accumulator = state_manager.get_prover_accumulator();
        accumulator.borrow_mut().append_dense(
            transcript,
            committed_polys,
            SumcheckId::SpartanOuter,
            r_cycle.to_vec(),
            &committed_poly_claims,
        );

        // Add virtual polynomial evaluations to the accumulator
        // These are needed by the verifier for future sumchecks and are not part of the PCS opening proof
        for (input, eval) in ALL_R1CS_INPUTS.iter().zip(claimed_witness_evals.iter()) {
            // Skip if it's a committed input (already added above)
            if !COMMITTED_R1CS_INPUTS.contains(input) {
                accumulator.borrow_mut().append_virtual(
                    transcript,
                    VirtualPolynomial::try_from(input).ok().unwrap(),
                    SumcheckId::SpartanOuter,
                    OpeningPoint::new(r_cycle.to_vec()),
                    *eval,
                );
            }
        }
    }

    /// Verify only the univariate skip round and return state for remaining rounds
    #[tracing::instrument(skip_all, name = "OuterSumcheck::verify_univariate_skip_round")]
    pub fn verify_univariate_skip_round<ProofTranscript: Transcript>(
        first_poly: &UniPoly<F>,
        _tau_high: &F::Challenge,
        transcript: &mut ProofTranscript,
    ) -> Result<(F::Challenge, F), anyhow::Error> {
        // Verify the first polynomial and derive r0
        first_poly.append_to_transcript(transcript);
        let r0: F::Challenge = transcript.challenge_scalar_optimized::<F>();

        // Compute the claim after first round
        let claim = first_poly.evaluate(&r0);

        Ok((r0, claim))
    }

    /// Verify the outer sumcheck using StateManager, handling accumulator updates
    #[tracing::instrument(skip_all, name = "OuterSumcheck::verify_with_state_manager")]
    pub fn verify_with_state_manager<
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F>,
    >(
        self,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
        num_rounds_x: usize,
        tau: Vec<F::Challenge>,
    ) -> Result<(), anyhow::Error> {
        use crate::poly::eq_poly::EqPolynomial;
        use crate::zkvm::r1cs::constraints::{
            FIRST_ROUND_POLY_NUM_COEFFS, UNIVARIATE_SKIP_DOMAIN_SIZE,
        };

        // Get the outer sumcheck proof
        let proofs = state_manager.proofs.borrow();
        let proof_data = {
            proofs
                .get(&ProofKeys::Stage1Sumcheck)
                .expect("Outer sumcheck proof not found")
        };

        let outer_sumcheck_proof = match proof_data {
            ProofData::UniSkipSumcheckProof(proof) => proof,
            _ => panic!("Invalid proof data type"),
        };

        // Get the claims:
        let accumulator = state_manager.get_verifier_accumulator();
        let accumulator_ref = accumulator.borrow();
        let (_, claim_Az) = accumulator_ref
            .get_virtual_polynomial_opening(VirtualPolynomial::SpartanAz, SumcheckId::SpartanOuter);
        let (_, claim_Bz) = accumulator_ref
            .get_virtual_polynomial_opening(VirtualPolynomial::SpartanBz, SumcheckId::SpartanOuter);
        let (_, claim_Cz) = accumulator_ref
            .get_virtual_polynomial_opening(VirtualPolynomial::SpartanCz, SumcheckId::SpartanOuter);
        drop(accumulator_ref);
        let outer_sumcheck_claims = [claim_Az, claim_Bz, claim_Cz];

        let transcript: &mut ProofTranscript = &mut state_manager.transcript.borrow_mut();

        // Run the main sumcheck verifier:
        let (claim_outer_final, outer_sumcheck_r_original) = {
            let transcript = &mut state_manager.transcript.borrow_mut();
            // The outer sumcheck has to be verified with an altered verifier, which takes
            // into account the univariate skip and does a high-degree interpolation in the first round
            match outer_sumcheck_proof
                .verify::<UNIVARIATE_SKIP_DOMAIN_SIZE, FIRST_ROUND_POLY_NUM_COEFFS>(
                    num_rounds_x,
                    FIRST_ROUND_POLY_NUM_COEFFS - 1,
                    3,
                    transcript,
                ) {
                Ok(result) => result,
                Err(_) => return Err(anyhow::anyhow!("Outer sumcheck verification failed")),
            }
        };

        // Outer sumcheck is bound from the top, reverse the challenge
        let outer_sumcheck_r_reversed: Vec<F::Challenge> =
            outer_sumcheck_r_original.iter().rev().cloned().collect();
        let opening_point = OpeningPoint::new(outer_sumcheck_r_reversed.clone());

        // Populate the opening points for Az, Bz, Cz claims now that we have outer_sumcheck_r
        // Note that the inner sumcheck will handle this opening point differently than other sumchecks,
        // due to univariate skip (first degree is higher)
        // There is NO other sumcheck instance that requires the high-degree part of the opening point
        // (since it is confined to the R1CS constraint part, not the cycle part)
        accumulator.borrow_mut().append_virtual(
            transcript,
            VirtualPolynomial::SpartanAz,
            SumcheckId::SpartanOuter,
            opening_point.clone(),
        );
        accumulator.borrow_mut().append_virtual(
            transcript,
            VirtualPolynomial::SpartanBz,
            SumcheckId::SpartanOuter,
            opening_point.clone(),
        );
        accumulator.borrow_mut().append_virtual(
            transcript,
            VirtualPolynomial::SpartanCz,
            SumcheckId::SpartanOuter,
            opening_point.clone(),
        );

        let tau_bound_rx = EqPolynomial::mle(&tau, &outer_sumcheck_r_reversed);
        let claim_outer_final_expected = tau_bound_rx * (claim_Az * claim_Bz - claim_Cz);
        if claim_outer_final != claim_outer_final_expected {
            return Err(anyhow::anyhow!("Invalid outer sumcheck claim"));
        }

        ProofTranscript::append_scalars(
            &mut state_manager.transcript.borrow_mut(),
            &outer_sumcheck_claims[..],
        );

        // Add the commitments to verifier accumulator
        // Extract num_cycles from trace_length
        let (_, _, trace_length) = state_manager.get_verifier_data();
        let num_cycles = trace_length.next_power_of_two();
        let num_cycles_bits = num_cycles.ilog2() as usize;

        let (r_cycle, _rx_var) = outer_sumcheck_r_reversed.split_at(num_cycles_bits);

        let accumulator = state_manager.get_verifier_accumulator();

        // Only non-virtual (i.e. committed) polynomials' openings are
        // proven using the PCS opening proof, which we add for future opening proof here
        let committed_polys: Vec<_> = COMMITTED_R1CS_INPUTS
            .iter()
            .map(|input| CommittedPolynomial::try_from(input).ok().unwrap())
            .collect();
        accumulator.borrow_mut().append_dense(
            transcript,
            committed_polys,
            SumcheckId::SpartanOuter,
            r_cycle.to_vec(),
        );

        ALL_R1CS_INPUTS.iter().for_each(|input| {
            // Skip if it's a committed input (already added above)
            if !COMMITTED_R1CS_INPUTS.contains(input) {
                accumulator.borrow_mut().append_virtual(
                    transcript,
                    VirtualPolynomial::try_from(input).ok().unwrap(),
                    SumcheckId::SpartanOuter,
                    OpeningPoint::new(r_cycle.to_vec()),
                );
            }
        });

        Ok(())
    }
}

impl<F: JoltField> OuterSumcheck<F> {
    /// Compute the list of Z targets in the extended domain that are outside the base window.
    /// The extended domain is [-D, ..., D] where D = UNIVARIATE_SKIP_DEGREE.
    /// The base window has size UNIVARIATE_SKIP_DOMAIN_SIZE and spans [base_left, base_right].
    /// We return the complement in an interleaved order starting just outside the window:
    /// left, right, left, right, ... then any remaining on the longer side.
    #[inline]
    fn univariate_skip_targets() -> [i64; UNIVARIATE_SKIP_DEGREE] {
        let d: i64 = UNIVARIATE_SKIP_DEGREE as i64;
        let ext_left: i64 = -d;
        let ext_right: i64 = d;
        let base_left: i64 = -((UNIVARIATE_SKIP_DOMAIN_SIZE as i64 - 1) / 2);
        let base_right: i64 = base_left + (UNIVARIATE_SKIP_DOMAIN_SIZE as i64) - 1;

        let mut targets: [i64; UNIVARIATE_SKIP_DEGREE] = [0; UNIVARIATE_SKIP_DEGREE];
        let mut idx = 0usize;
        let mut n = base_left - 1; // first index just left of the base window
        let mut p = base_right + 1; // first index just right of the base window

        // Interleave negatives and positives while both sides have items
        while n >= ext_left && p <= ext_right && idx < UNIVARIATE_SKIP_DEGREE {
            targets[idx] = n;
            idx += 1;
            if idx >= UNIVARIATE_SKIP_DEGREE {
                break;
            }
            targets[idx] = p;
            idx += 1;
            n -= 1;
            p += 1;
        }

        // Append any remaining on the left side
        while idx < UNIVARIATE_SKIP_DEGREE && n >= ext_left {
            targets[idx] = n;
            idx += 1;
            n -= 1;
        }

        // Append any remaining on the right side
        while idx < UNIVARIATE_SKIP_DEGREE && p <= ext_right {
            targets[idx] = p;
            idx += 1;
            p += 1;
        }

        debug_assert_eq!(idx, UNIVARIATE_SKIP_DEGREE);
        targets
    }

    /// Small-value optimization with univariate skip instead of round batching / compression
    ///
    /// Returns the UNIVARIATE_SKIP_DEGREE accumulators:
    /// t_1(z) = \sum_{x_out} E_out[x_out] \sum_{x_in} E_in[x_in] *
    ///             \sum_{y = 0,1} E[y] * (Az(x_out, x_in, y, z) * Bz(x_out, x_in, y, z) - Cz(x_out, x_in, y, z))
    ///
    /// for all z in the extended domain [-D..D] excluding the base window of size D+1,
    /// where D = UNIVARIATE_SKIP_DEGREE.
    #[tracing::instrument(skip_all, name = "OuterSumcheck::compute_univariate_skip_evals")]
    pub fn compute_univariate_skip_evals(
        &mut self,
        preprocess: &JoltSharedPreprocessing,
        trace: &[Cycle],
    ) -> [F; UNIVARIATE_SKIP_DEGREE] {
        // Precompute Lagrange coefficient vectors for target Z values outside the base window
        let base_left: i64 = -((UNIVARIATE_SKIP_DOMAIN_SIZE as i64 - 1) / 2);
        let targets: [i64; UNIVARIATE_SKIP_DEGREE] = Self::univariate_skip_targets();
        let target_shifts: [i64; UNIVARIATE_SKIP_DEGREE] =
            core::array::from_fn(|j| targets[j] - base_left);
        let coeffs_per_j: [[i32; UNIVARIATE_SKIP_DOMAIN_SIZE]; UNIVARIATE_SKIP_DEGREE] =
            core::array::from_fn(|j| {
                LagrangeHelper::shift_coeffs_i32::<UNIVARIATE_SKIP_DOMAIN_SIZE>(target_shifts[j])
            });

        let num_x_out_vals = self.split_eq_poly.E_out_current_len();
        let num_x_in_vals = self.split_eq_poly.E_in_current_len();
        let num_parallel_chunks = if num_x_out_vals > 0 {
            std::cmp::min(
                num_x_out_vals,
                rayon::current_num_threads().next_power_of_two() * 8,
            )
        } else {
            1
        };
        let x_out_chunk_size = if num_x_out_vals > 0 {
            std::cmp::max(1, num_x_out_vals.div_ceil(num_parallel_chunks))
        } else {
            0
        };
        let iter_num_x_in_vars = if num_x_in_vals > 0 {
            num_x_in_vals.log_2()
        } else {
            0
        };

        (0..num_parallel_chunks)
            .into_par_iter()
            .map(|chunk_idx| {
                let x_out_start = chunk_idx * x_out_chunk_size;
                let x_out_end = std::cmp::min((chunk_idx + 1) * x_out_chunk_size, num_x_out_vals);
                let mut acc_field: [F; UNIVARIATE_SKIP_DEGREE] =
                    [F::zero(); UNIVARIATE_SKIP_DEGREE];

                for x_out_val in x_out_start..x_out_end {
                    let mut inner_acc: [<F as JoltField>::Unreduced<9>; UNIVARIATE_SKIP_DEGREE] =
                        [<F as JoltField>::Unreduced::<9>::zero(); UNIVARIATE_SKIP_DEGREE];

                    for x_in_val in 0..num_x_in_vals {
                        let current_step_idx = (x_out_val << iter_num_x_in_vars) | x_in_val;
                        let row_inputs =
                            R1CSCycleInputs::from_trace::<F>(preprocess, trace, current_step_idx);

                        // e_in per x_in
                        let e_in = if num_x_in_vals == 0 {
                            F::one()
                        } else {
                            self.split_eq_poly.E_in_current()[x_in_val]
                        };

                        // First group (UNIVARIATE_SKIP_DOMAIN_SIZE eq-conditional, Cz=0): Az bool, Bz S160
                        let az1_bool = eval_az_first_group(&row_inputs);
                        let bz1_s160 = eval_bz_first_group(&row_inputs);

                        // Second group (NUM_REMAINING_R1CS_CONSTRAINTS + pad): Az i128, Bz/Cz S160
                        let az2_i96 = eval_az_second_group(&row_inputs);
                        let bz2 = eval_bz_second_group(&row_inputs);
                        let cz2 = eval_cz_second_group(&row_inputs);

                        let mut az2_i128_padded: [i128; UNIVARIATE_SKIP_DOMAIN_SIZE] =
                            [0; UNIVARIATE_SKIP_DOMAIN_SIZE];
                        let mut bz2_s160_padded: [S160; UNIVARIATE_SKIP_DOMAIN_SIZE] =
                            [S160::from(0i128); UNIVARIATE_SKIP_DOMAIN_SIZE];
                        let mut cz2_s160_padded: [S160; UNIVARIATE_SKIP_DOMAIN_SIZE] =
                            [S160::from(0i128); UNIVARIATE_SKIP_DOMAIN_SIZE];
                        for i in 0..(UNIVARIATE_SKIP_DOMAIN_SIZE - 1) {
                            az2_i128_padded[i] = az2_i96[i].to_i128();
                            bz2_s160_padded[i] = bz2[i];
                            cz2_s160_padded[i] = cz2[i];
                        }

                        // Accumulate per target j using product-of-sums form
                        for j in 0..UNIVARIATE_SKIP_DEGREE {
                            let coeffs = &coeffs_per_j[j];

                            // Group 1: (Σ c_i * Az1[i]) * (Σ c_i * Bz1[i])
                            // Optimize: accumulate az1 as i64 sum; accumulate bz1 via 7-limb fmadd and reduce once
                            let mut az1_csum: i64 = 0;
                            let mut bz1_acc = accs160_new::<F>();
                            for i in 0..UNIVARIATE_SKIP_DOMAIN_SIZE {
                                let c = coeffs[i] as i64;
                                if c == 0 {
                                    continue;
                                }
                                if az1_bool[i] { az1_csum += c; }
                                accs160_fmadd_s160(&mut bz1_acc, &F::from_i64(c), bz1_s160[i]);
                            }
                            let az1_ext = F::from_i64(az1_csum);
                            let bz1_ext = accs160_reduce(&bz1_acc);
                            inner_acc[j] += e_in.mul_unreduced::<9>(az1_ext * bz1_ext);

                            // Group 2: (Σ c_i * Az2[i])*(Σ c_i * Bz2[i]) - (Σ c_i * Cz2[i])
                            // Optimize: accumulate az2 as i128 sum and convert once; bz2/cz2 via 7-limb fmadd
                            let mut az2_sum: i128 = 0;
                            let mut bz2_acc = accs160_new::<F>();
                            let mut cz2_acc = accs160_new::<F>();
                            for i in 0..UNIVARIATE_SKIP_DOMAIN_SIZE {
                                let c = coeffs[i] as i64;
                                if c == 0 {
                                    continue;
                                }
                                az2_sum += az2_i128_padded[i] * (c as i128);
                                accs160_fmadd_s160(&mut bz2_acc, &F::from_i64(c), bz2_s160_padded[i]);
                                accs160_fmadd_s160(&mut cz2_acc, &F::from_i64(c), cz2_s160_padded[i]);
                            }
                            let az2_ext = F::from_i128(az2_sum);
                            let bz2_ext = accs160_reduce(&bz2_acc);
                            let cz2_ext = accs160_reduce(&cz2_acc);
                            inner_acc[j] += e_in.mul_unreduced::<9>(az2_ext * bz2_ext - cz2_ext);
                        }
                    }
                    let e_out = self.split_eq_poly.E_out_current()[x_out_val];

                    // Apply e_out once per x_out after reducing
                    for j in 0..UNIVARIATE_SKIP_DEGREE {
                        let reduced = F::from_montgomery_reduce::<9>(inner_acc[j]);
                        acc_field[j] += e_out * reduced;
                    }
                }

                acc_field
            })
            .reduce(
                || [F::zero(); UNIVARIATE_SKIP_DEGREE],
                |mut a, b| {
                    for j in 0..UNIVARIATE_SKIP_DEGREE {
                        a[j] += b[j];
                    }
                    a
                },
            )
    }

    /// Builds the first-round polynomial `s1(Z)` from the univariate-skip extended evaluations.
    ///
    /// Steps:
    /// - Places the `UNIVARIATE_SKIP_DEGREE` extended evaluations into the symmetric domain
    ///   for `t1(Z)` on Z ∈ [-13..13], then interpolates the degree-26 coefficients of `t1`.
    /// - Evaluates the length-14 Lagrange basis at `tau_high` over the base window [-6..7] and
    ///   interpolates the degree-13 coefficients of the window polynomial `lagrange_poly(Z)`.
    /// - Convolves `lagrange_poly(Z)` with `t1(Z)` to obtain `s1(Z) = lagrange_poly(Z) * t1(Z)`
    ///   (degree 39).
    /// - Appends the full `s1` coefficients to the transcript, derives `r0`, sets the current
    ///   claim to `s1(r0)`, pushes `r0` into `r`, and returns `s1`.
    ///
    /// This round sends the full high-degree polynomial once; subsequent rounds use compressed
    /// cubic polynomials.
    #[inline]
    fn process_first_round_from_extended_evals<ProofTranscript: Transcript>(
        &mut self,
        extended_evals: &[F; UNIVARIATE_SKIP_DEGREE],
        transcript: &mut ProofTranscript,
        r: &mut Vec<F::Challenge>,
    ) -> UniPoly<F> {
        // Map the UNIVARIATE_SKIP_DEGREE interleaved extended evals into the full symmetric domain of size (2D+1): Z in [-D..D]
        let mut t1_vals: [F; 2 * UNIVARIATE_SKIP_DEGREE + 1] =
            [F::zero(); 2 * UNIVARIATE_SKIP_DEGREE + 1];
        let targets: [i64; UNIVARIATE_SKIP_DEGREE] = Self::univariate_skip_targets();
        for (idx, &val) in extended_evals.iter().enumerate() {
            let z = targets[idx];
            let pos = (z + (UNIVARIATE_SKIP_DEGREE as i64)) as usize;
            t1_vals[pos] = val;
        }

        // Interpolate degree-26 coefficients of t1 from its 27 values
        // TODO: do this faster relying on the fact that t1 evals on original domain are zero
        let t1_coeffs = LagrangePolynomial::<F>::interpolate_coeffs::<
            UNIVARIATE_SKIP_EXTENDED_DOMAIN_SIZE,
        >(&t1_vals);

        // Build lagrange_poly(Z) coefficients of degree 13 from basis values at tau_high over base window [-6..7]
        // TODO: can build this directly
        let lagrange_poly_values = LagrangePolynomial::<F>::evals::<
            F::Challenge,
            UNIVARIATE_SKIP_DOMAIN_SIZE,
        >(&self.tau_high);
        let lagrange_poly_coeffs = LagrangePolynomial::interpolate_coeffs::<
            UNIVARIATE_SKIP_DOMAIN_SIZE,
        >(&lagrange_poly_values);

        // Convolve lagrange_poly (len 14) with t1 (len 27) to get s1 (len 40), degree 39
        let mut s1_coeffs: [F; FIRST_ROUND_POLY_NUM_COEFFS] =
            [F::zero(); FIRST_ROUND_POLY_NUM_COEFFS];
        for (i, &a) in lagrange_poly_coeffs.iter().enumerate() {
            for (j, &b) in t1_coeffs.iter().enumerate() {
                s1_coeffs[i + j] += a * b;
            }
        }

        // Append full s1 poly (send all coeffs), derive r0, set claim (do NOT bind eq_poly yet)
        let s1_poly = UniPoly::from_coeff(s1_coeffs.to_vec());
        s1_poly.append_to_transcript(transcript);
        let r0: F::Challenge = transcript.challenge_scalar_optimized::<F>();
        r.push(r0);
        self.claim = UniPoly::eval_with_coeffs(&s1_coeffs, &r0);
        s1_poly
    }

    /// Computes contiguous index ranges that emulate `par_chunk_by` grouping by `index / block_size`.
    /// Each range [start, end) contains coefficients belonging to the same block bucket.
    fn compute_block_ranges<T>(
        coeffs: &[SparseCoefficient<T>],
        block_size: usize,
    ) -> Vec<(usize, usize)> {
        if coeffs.is_empty() {
            return Vec::new();
        }
        // Safety/net: block_size must be a multiple of 6 to respect Az/Bz/Cz block layout
        debug_assert_eq!(block_size % 6, 0);

        let mut ranges = Vec::new();
        let mut start = 0usize;
        let mut current_bucket = coeffs[0].index / block_size;
        for (i, c) in coeffs.iter().enumerate().skip(1) {
            let bucket = c.index / block_size;
            if bucket != current_bucket {
                ranges.push((start, i));
                start = i;
                current_bucket = bucket;
            }
        }
        ranges.push((start, coeffs.len()));
        ranges
    }

    /// Computes the number of non-zero coefficients that would result from
    /// binding the given slice of coefficients. Only invoked on `bound_coeffs` which holds
    /// Az/Bz/Cz bound evaluations.
    fn binding_output_length<T>(coeffs: &[SparseCoefficient<T>]) -> usize {
        let mut output_size = 0;
        for block in coeffs.chunk_by(|x, y| x.index / 6 == y.index / 6) {
            let mut Az_coeff_found = false;
            let mut Bz_coeff_found = false;
            let mut Cz_coeff_found = false;
            for coeff in block {
                match coeff.index % 3 {
                    0 => {
                        if !Az_coeff_found {
                            Az_coeff_found = true;
                            output_size += 1;
                        }
                    }
                    1 => {
                        if !Bz_coeff_found {
                            Bz_coeff_found = true;
                            output_size += 1;
                        }
                    }
                    2 => {
                        if !Cz_coeff_found {
                            Cz_coeff_found = true;
                            output_size += 1;
                        }
                    }
                    _ => unreachable!(),
                }
            }
        }
        output_size
    }
}

/// SumcheckInstance for Spartan outer rounds after the univariate-skip first round.
/// Round 0 in this instance corresponds to the "streaming" round; subsequent rounds
/// use the remaining linear-time algorithm over cycle variables.
pub struct OuterRemainingSumcheck<F: JoltField> {
    pub input_claim: F,
    pub interleaved_poly: SpartanInterleavedPoly<F>,
    pub split_eq_poly: GruenSplitEqPolynomial<F>,
    pub preprocess: Arc<JoltSharedPreprocessing>,
    pub trace: Arc<Vec<Cycle>>,
    pub r0_uniskip: F::Challenge,
    pub total_rounds: usize,
    /// Only used by verifier to compute expected_output_claim
    pub tau: Option<Vec<F::Challenge>>, 
}

impl<F: JoltField> OuterRemainingSumcheck<F> {
    pub fn new_prover(
        input_claim: F,
        preprocess: &JoltSharedPreprocessing,
        trace: &[Cycle],
        split_eq_poly: GruenSplitEqPolynomial<F>,
        interleaved_poly: SpartanInterleavedPoly<F>,
        r0_uniskip: F::Challenge,
        total_rounds: usize,
    ) -> Self {
        Self {
            input_claim,
            interleaved_poly,
            split_eq_poly,
            preprocess: Arc::new(preprocess.clone()),
            trace: Arc::new(trace.to_vec()),
            r0_uniskip,
            total_rounds,
            tau: None,
        }
    }

    pub fn new_verifier(
        input_claim: F,
        preprocess: &crate::zkvm::JoltSharedPreprocessing,
        trace: &[Cycle],
        split_eq_poly: GruenSplitEqPolynomial<F>,
        interleaved_poly: SpartanInterleavedPoly<F>,
        r0_uniskip: F::Challenge,
        total_rounds: usize,
        tau: Vec<F::Challenge>,
    ) -> Self {
        Self {
            input_claim,
            interleaved_poly,
            split_eq_poly,
            preprocess: Arc::new(preprocess.clone()),
            trace: Arc::new(trace.to_vec()),
            r0_uniskip,
            total_rounds,
            tau: Some(tau),
        }
    }

    #[inline]
    fn build_cubic_from_quadratic(
        eq_poly: &GruenSplitEqPolynomial<F>,
        t0: F,
        t_inf: F,
        previous_claim: F,
    ) -> UniPoly<F> {
        let scalar_times_w_i = eq_poly.current_scalar * eq_poly.w[eq_poly.current_index - 1];
        UniPoly::from_linear_times_quadratic_with_hint(
            [
                eq_poly.current_scalar - scalar_times_w_i,
                scalar_times_w_i + scalar_times_w_i - eq_poly.current_scalar,
            ],
            t0,
            t_inf,
            previous_claim,
        )
    }

    /// Compute the quadratic evaluations for the streaming round (right after univariate skip).
    ///
    /// This uses the streaming algorithm to compute the sum-check polynomial for the round
    /// right after the univariate skip round.
    ///
    /// Recall that we need to compute
    ///
    /// `t_i(0) = \sum_{x_out} E_out[x_out] \sum_{x_in} E_in[x_in] * (unbound_coeffs_a(x_out, x_in,
    /// 0, r) * unbound_coeffs_b(x_out, x_in, 0, r) - unbound_coeffs_c(x_out, x_in, 0, r))`
    ///
    /// and
    ///
    /// `t_i(∞) = \sum_{x_out} E_out[x_out] \sum_{x_in} E_in[x_in] * (unbound_coeffs_a(x_out,
    /// x_in, ∞, r) * unbound_coeffs_b(x_out, x_in, ∞, r))`
    ///
    /// Here the "_a,b,c" subscript indicates the coefficients of `unbound_coeffs` corresponding to
    /// Az, Bz, Cz respectively. Note that we index with x_out being the MSB here.
    ///
    /// Importantly, since the eval at `r` is not cached, we will need to recompute it via another
    /// sum
    ///
    /// `unbound_coeffs_{a,b,c}(x_out, x_in, {0,∞}, r) = \sum_{y in D} eq(r, y) *
    /// unbound_coeffs_{a,b,c}(x_out, x_in, {0,∞}, y)`
    ///
    /// (and the eval at ∞ is computed as (eval at 1) - (eval at 0))
    #[inline]
    fn streaming_quadratic_evals(&self) -> (F, F) {
        // Lagrange basis over the univariate-skip domain at r0
        let lagrange_evals_r = LagrangePolynomial::<F>::evals::<
            F::Challenge,
            UNIVARIATE_SKIP_DOMAIN_SIZE,
        >(&self.r0_uniskip);

        let eq_poly = &self.split_eq_poly;
        let num_x_out_vals = eq_poly.E_out_current_len();
        let num_x_in_vals = eq_poly.E_in_current_len();

        let num_parallel_chunks = if num_x_out_vals > 0 {
            core::cmp::min(
                num_x_out_vals,
                rayon::current_num_threads().next_power_of_two() * 8,
            )
        } else {
            1
        };
        let x_out_chunk_size = if num_x_out_vals > 0 {
            core::cmp::max(1, num_x_out_vals.div_ceil(num_parallel_chunks))
        } else {
            0
        };
        let iter_num_x_in_vars = if num_x_in_vals > 0 {
            num_x_in_vals.log_2()
        } else {
            0
        };

        let results: (F, F) = (0..num_parallel_chunks)
            .into_par_iter()
            .map(|chunk_idx| {
                let x_out_start = chunk_idx * x_out_chunk_size;
                let x_out_end = core::cmp::min((chunk_idx + 1) * x_out_chunk_size, num_x_out_vals);
                let mut task_sum0 = F::zero();
                let mut task_sumInf = F::zero();

                for x_out_val in x_out_start..x_out_end {
                    let mut inner_sum0 = F::Unreduced::<9>::zero();
                    let mut inner_sumInf = F::Unreduced::<9>::zero();

                    for x_in_val in 0..num_x_in_vals {
                        let current_step_idx = (x_out_val << iter_num_x_in_vars) | x_in_val;

                        let row_inputs = R1CSCycleInputs::from_trace::<F>(
                            self.preprocess.as_ref(),
                            self.trace.as_slice(),
                            current_step_idx,
                        );

                        // reduce to field values at y=r for both x_next
                        let az0 = compute_az_r_group0(&row_inputs, &lagrange_evals_r[..]);
                        let bz0 = compute_bz_r_group0(&row_inputs, &lagrange_evals_r[..]);
                        let az1 = compute_az_r_group1(&row_inputs, &lagrange_evals_r[..]);
                        let bz1 = compute_bz_r_group1(&row_inputs, &lagrange_evals_r[..]);
                        let cz1 = compute_cz_r_group1(&row_inputs, &lagrange_evals_r[..]);

                        // sumcheck contributions
                        let p0 = az0 * bz0;
                        let slope = (az1 - az0) * (bz1 - bz0);

                        // e_in per x_in
                        let e_in = if num_x_in_vals == 0 {
                            F::one()
                        } else if num_x_in_vals == 1 {
                            eq_poly.E_in_current()[0]
                        } else {
                            eq_poly.E_in_current()[x_in_val]
                        };

                        inner_sum0 += e_in.mul_unreduced::<9>(p0);
                        inner_sumInf += e_in.mul_unreduced::<9>(slope);
                        // cz1 only affects evaluation used later during binding, not t(0)/t(∞)
                        let _ = cz1;
                    }

                    let e_out = if num_x_out_vals > 0 {
                        eq_poly.E_out_current()[x_out_val]
                    } else {
                        F::zero()
                    };

                    let reduced0 = F::from_montgomery_reduce::<9>(inner_sum0);
                    let reducedInf = F::from_montgomery_reduce::<9>(inner_sumInf);
                    task_sum0 += e_out * reduced0;
                    task_sumInf += e_out * reducedInf;
                }

                (task_sum0, task_sumInf)
            })
            .reduce(|| (F::zero(), F::zero()), |a, b| (a.0 + b.0, a.1 + b.1));

        results
    }

    /// Compute the polynomial for each of the remaining rounds, using the
    /// linear-time algorithm with split-eq optimizations.
    ///
    /// At this point, we have computed the `bound_coeffs` for the current round.
    /// We need to compute:
    ///
    /// `t_i(0) = \sum_{x_out} E_out[x_out] \sum_{x_in} E_in[x_in] *
    /// (az_bound[x_out, x_in, 0] * bz_bound[x_out, x_in, 0] - cz_bound[x_out, x_in, 0])`
    ///
    /// and
    ///
    /// `t_i(∞) = \sum_{x_out} E_out[x_out] \sum_{x_in} E_in[x_in] *
    /// az_bound[x_out, x_in, ∞] * bz_bound[x_out, x_in, ∞]`
    ///
    /// (ordering of indices is MSB to LSB, so x_out is the MSB and x_in is the LSB)
    #[inline]
    fn remaining_quadratic_evals(&self) -> (F, F) {
        let eq_poly = &self.split_eq_poly;

        let chunk_ranges = {
            let block_size = self
                .interleaved_poly
                .bound_coeffs
                .len()
                .div_ceil(rayon::current_num_threads())
                .next_multiple_of(6);
            OuterSumcheck::<F>::compute_block_ranges(
                &self.interleaved_poly.bound_coeffs,
                block_size,
            )
        };

        if eq_poly.E_in_current_len() == 1 {
            chunk_ranges
                .par_iter()
                .flat_map_iter(|&(start, end)| {
                    let chunk = &self.interleaved_poly.bound_coeffs[start..end];
                    chunk
                        .chunk_by(|x, y| x.index / 6 == y.index / 6)
                        .map(|sparse_block| {
                            let block_index = sparse_block[0].index / 6;
                            let mut block = [F::zero(); 6];
                            for coeff in sparse_block {
                                block[coeff.index % 6] = coeff.value;
                            }

                            let az = (block[0], block[3]);
                            let bz = (block[1], block[4]);
                            let cz0 = block[2];

                            let az_eval_infty = az.1 - az.0;
                            let bz_eval_infty = bz.1 - bz.0;

                            let eq_evals = eq_poly.E_out_current()[block_index];

                            (
                                eq_evals.mul_0_optimized(az.0.mul_0_optimized(bz.0) - cz0),
                                eq_evals
                                    .mul_0_optimized(az_eval_infty.mul_0_optimized(bz_eval_infty)),
                            )
                        })
                })
                .reduce(
                    || (F::zero(), F::zero()),
                    |sum, evals| (sum.0 + evals.0, sum.1 + evals.1),
                )
        } else {
            let num_x1_bits = eq_poly.E_in_current_len().log_2();
            let x1_bitmask = (1 << num_x1_bits) - 1;

            chunk_ranges
                .par_iter()
                .map(|&(start, end)| {
                    let chunk = &self.interleaved_poly.bound_coeffs[start..end];
                    let mut eval_point_0 = F::zero();
                    let mut eval_point_infty = F::zero();

                    let mut inner_sums = (F::Unreduced::<9>::zero(), F::Unreduced::<9>::zero());
                    let mut prev_x2 = 0;

                    for sparse_block in chunk.chunk_by(|x, y| x.index / 6 == y.index / 6) {
                        let block_index = sparse_block[0].index / 6;
                        let x1 = block_index & x1_bitmask;
                        let E_in_evals = eq_poly.E_in_current()[x1];
                        let x2 = block_index >> num_x1_bits;

                        if x2 != prev_x2 {
                            let reduced0 = F::from_montgomery_reduce::<9>(inner_sums.0);
                            let reducedInf = F::from_montgomery_reduce::<9>(inner_sums.1);
                            eval_point_0 += eq_poly.E_out_current()[prev_x2] * reduced0;
                            eval_point_infty += eq_poly.E_out_current()[prev_x2] * reducedInf;
                            inner_sums = (F::Unreduced::<9>::zero(), F::Unreduced::<9>::zero());
                            prev_x2 = x2;
                        }

                        let mut block = [F::zero(); 6];
                        for coeff in sparse_block {
                            block[coeff.index % 6] = coeff.value;
                        }

                        let az = (block[0], block[3]);
                        let bz = (block[1], block[4]);
                        let cz0 = block[2];

                        let az_eval_infty = az.1 - az.0;
                        let bz_eval_infty = bz.1 - bz.0;

                        inner_sums.0 +=
                            E_in_evals.mul_unreduced::<9>(az.0.mul_0_optimized(bz.0) - cz0);
                        inner_sums.1 += E_in_evals
                            .mul_unreduced::<9>(az_eval_infty.mul_0_optimized(bz_eval_infty));
                    }

                    let reduced0 = F::from_montgomery_reduce::<9>(inner_sums.0);
                    let reducedInf = F::from_montgomery_reduce::<9>(inner_sums.1);
                    eval_point_0 += eq_poly.E_out_current()[prev_x2] * reduced0;
                    eval_point_infty += eq_poly.E_out_current()[prev_x2] * reducedInf;

                    (eval_point_0, eval_point_infty)
                })
                .reduce(
                    || (F::zero(), F::zero()),
                    |sum, evals| (sum.0 + evals.0, sum.1 + evals.1),
                )
        }
    }

    /// Bind the streaming round after deriving challenge r_i.
    ///
    /// As we compute each `{a/b/c}(x_out, x_in, {0,∞}, r)`, we will
    /// store them in `bound_coeffs` in sparse format (the eval at 1 will be eval
    /// at 0 + eval at ∞). We then bind these bound coeffs with r_i for the next round.
    fn bind_streaming_round(&mut self, r_i: F::Challenge) {
        // Lagrange basis over the univariate-skip domain at r0
        let lagrange_evals_r = LagrangePolynomial::<F>::evals::<
            F::Challenge,
            UNIVARIATE_SKIP_DOMAIN_SIZE,
        >(&self.r0_uniskip);
        let eq_poly = &mut self.split_eq_poly;
        let num_x_out_vals = eq_poly.E_out_current_len();
        let num_x_in_vals = eq_poly.E_in_current_len();
        let iter_num_x_in_vars = if num_x_in_vals > 0 {
            num_x_in_vals.log_2()
        } else {
            0
        };

        // Build bound6_at_r sparsely
        let mut bound6_at_r: Vec<SparseCoefficient<F>> = Vec::new();
        let mut reserve = num_x_out_vals
            .saturating_mul(core::cmp::max(1, num_x_in_vals))
            .saturating_mul(4);
        reserve = reserve.max(1024);
        bound6_at_r.reserve(reserve);

        for x_out_val in 0..num_x_out_vals {
            for x_in_val in 0..num_x_in_vals {
                let current_step_idx = (x_out_val << iter_num_x_in_vars) | x_in_val;
                let row_inputs = R1CSCycleInputs::from_trace::<F>(
                    self.preprocess.as_ref(),
                    self.trace.as_slice(),
                    current_step_idx,
                );
                let az0 = compute_az_r_group0(&row_inputs, &lagrange_evals_r[..]);
                let bz0 = compute_bz_r_group0(&row_inputs, &lagrange_evals_r[..]);
                let az1 = compute_az_r_group1(&row_inputs, &lagrange_evals_r[..]);
                let bz1 = compute_bz_r_group1(&row_inputs, &lagrange_evals_r[..]);
                let cz1 = compute_cz_r_group1(&row_inputs, &lagrange_evals_r[..]);

                let block_id = current_step_idx;
                if !az0.is_zero() {
                    bound6_at_r.push((6 * block_id, az0).into());
                }
                if !bz0.is_zero() {
                    bound6_at_r.push((6 * block_id + 1, bz0).into());
                }
                if !az1.is_zero() {
                    bound6_at_r.push((6 * block_id + 3, az1).into());
                }
                if !bz1.is_zero() {
                    bound6_at_r.push((6 * block_id + 4, bz1).into());
                }
                if !cz1.is_zero() {
                    bound6_at_r.push((6 * block_id + 5, cz1).into());
                }
            }
        }

        // Size output buffer (sequential; cheap compared to streaming work)
        let mut total_len: usize = 0;
        for block6 in bound6_at_r.chunk_by(|a, b| a.index / 6 == b.index / 6) {
            total_len += OuterSumcheck::<F>::binding_output_length(block6);
        }
        if self.interleaved_poly.binding_scratch_space.capacity() < total_len {
            self.interleaved_poly
                .binding_scratch_space
                .reserve_exact(total_len - self.interleaved_poly.binding_scratch_space.capacity());
        }
        unsafe {
            self.interleaved_poly
                .binding_scratch_space
                .set_len(total_len);
        }

        // Partition scratch and bind per block with r_i
        let mut output_index = 0usize;
        let out_slice = self.interleaved_poly.binding_scratch_space.as_mut_slice();
        for block6 in bound6_at_r.chunk_by(|a, b| a.index / 6 == b.index / 6) {
            if block6.is_empty() {
                continue;
            }
            let blk = block6[0].index / 6;
            let mut az0 = F::zero();
            let mut bz0 = F::zero();
            let mut cz0 = F::zero();
            let mut az1 = F::zero();
            let mut bz1 = F::zero();
            let mut cz1 = F::zero();
            let mut has_az = false;
            let mut has_bz = false;
            let mut has_cz = false;
            for c in block6 {
                match c.index % 6 {
                    0 => {
                        az0 = c.value;
                        has_az = true;
                    }
                    1 => {
                        bz0 = c.value;
                        has_bz = true;
                    }
                    2 => {
                        cz0 = c.value;
                        has_cz = true;
                    }
                    3 => {
                        az1 = c.value;
                        has_az = true;
                    }
                    4 => {
                        bz1 = c.value;
                        has_bz = true;
                    }
                    5 => {
                        cz1 = c.value;
                        has_cz = true;
                    }
                    _ => {}
                }
            }
            let azb = az0 + r_i * (az1 - az0);
            if has_az {
                out_slice[output_index] = (3 * blk, azb).into();
                output_index += 1;
            }
            let bzb = bz0 + r_i * (bz1 - bz0);
            if has_bz {
                out_slice[output_index] = (3 * blk + 1, bzb).into();
                output_index += 1;
            }
            if has_cz {
                let czb = cz0 + r_i * (cz1 - cz0);
                out_slice[output_index] = (3 * blk + 2, czb).into();
                output_index += 1;
            }
        }
        debug_assert_eq!(output_index, out_slice.len());

        core::mem::swap(
            &mut self.interleaved_poly.bound_coeffs,
            &mut self.interleaved_poly.binding_scratch_space,
        );
    }

    pub fn final_sumcheck_evals(&self) -> [F; 3] {
        let mut final_az_eval = F::zero();
        let mut final_bz_eval = F::zero();
        let mut final_cz_eval = F::zero();
        for coeff in &self.interleaved_poly.bound_coeffs {
            match coeff.index {
                0 => final_az_eval = coeff.value,
                1 => final_bz_eval = coeff.value,
                2 => final_cz_eval = coeff.value,
                _ => {}
            }
        }
        [final_az_eval, final_bz_eval, final_cz_eval]
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstance<F, T> for OuterRemainingSumcheck<F> {
    fn degree(&self) -> usize {
        3
    }

    fn num_rounds(&self) -> usize {
        self.total_rounds - 1 /* exclude first already handled? */ + 1
    }

    fn input_claim(&self) -> F {
        self.input_claim
    }

    fn compute_prover_message(&mut self, round: usize, previous_claim: F) -> Vec<F> {
        let (t0, t_inf) = if round == 0 {
            self.streaming_quadratic_evals()
        } else {
            self.remaining_quadratic_evals()
        };
        let cubic =
            Self::build_cubic_from_quadratic(&self.split_eq_poly, t0, t_inf, previous_claim);
        vec![
            cubic.evaluate::<F>(&F::zero()),
            cubic.evaluate::<F>(&F::from_u64(2)),
            cubic.evaluate::<F>(&F::from_u64(3)),
        ]
    }

    fn bind(&mut self, r_j: F::Challenge, round: usize) {
        if round == 0 {
            self.bind_streaming_round(r_j);
        } else {
            // Remaining rounds binding (reuse existing logic)
            let block_size = self
                .interleaved_poly
                .bound_coeffs
                .len()
                .div_ceil(rayon::current_num_threads())
                .next_multiple_of(6);
            let chunk_ranges = OuterSumcheck::<F>::compute_block_ranges(
                &self.interleaved_poly.bound_coeffs,
                block_size,
            );

            let output_sizes: Vec<_> = chunk_ranges
                .par_iter()
                .map(|&(start, end)| {
                    OuterSumcheck::<F>::binding_output_length(
                        &self.interleaved_poly.bound_coeffs[start..end],
                    )
                })
                .collect();
            let total_output_len = output_sizes.iter().sum();
            if self.interleaved_poly.binding_scratch_space.is_empty() {
                self.interleaved_poly.binding_scratch_space = Vec::with_capacity(total_output_len);
            }
            unsafe {
                self.interleaved_poly
                    .binding_scratch_space
                    .set_len(total_output_len);
            }

            let mut output_slices: Vec<&mut [SparseCoefficient<F>]> =
                Vec::with_capacity(chunk_ranges.len());
            let mut remainder = self.interleaved_poly.binding_scratch_space.as_mut_slice();
            for slice_len in output_sizes {
                let (first, second) = remainder.split_at_mut(slice_len);
                output_slices.push(first);
                remainder = second;
            }

            chunk_ranges
                .par_iter()
                .zip_eq(output_slices.into_par_iter())
                .for_each(|(&(start, end), output_slice)| {
                    let coeffs = &self.interleaved_poly.bound_coeffs[start..end];
                    let mut output_index = 0;
                    for block in coeffs.chunk_by(|x, y| x.index / 6 == y.index / 6) {
                        let block_index = block[0].index / 6;
                        let mut az_coeff: (Option<F>, Option<F>) = (None, None);
                        let mut bz_coeff: (Option<F>, Option<F>) = (None, None);
                        let mut cz_coeff: (Option<F>, Option<F>) = (None, None);
                        for coeff in block {
                            match coeff.index % 6 {
                                0 => az_coeff.0 = Some(coeff.value),
                                1 => bz_coeff.0 = Some(coeff.value),
                                2 => cz_coeff.0 = Some(coeff.value),
                                3 => az_coeff.1 = Some(coeff.value),
                                4 => bz_coeff.1 = Some(coeff.value),
                                5 => cz_coeff.1 = Some(coeff.value),
                                _ => unreachable!(),
                            }
                        }
                        if az_coeff != (None, None) {
                            let (low, high) = (
                                az_coeff.0.unwrap_or(F::zero()),
                                az_coeff.1.unwrap_or(F::zero()),
                            );
                            output_slice[output_index] =
                                (3 * block_index, low + r_j * (high - low)).into();
                            output_index += 1;
                        }
                        if bz_coeff != (None, None) {
                            let (low, high) = (
                                bz_coeff.0.unwrap_or(F::zero()),
                                bz_coeff.1.unwrap_or(F::zero()),
                            );
                            output_slice[output_index] =
                                (3 * block_index + 1, low + r_j * (high - low)).into();
                            output_index += 1;
                        }
                        if cz_coeff != (None, None) {
                            let (low, high) = (
                                cz_coeff.0.unwrap_or(F::zero()),
                                cz_coeff.1.unwrap_or(F::zero()),
                            );
                            output_slice[output_index] =
                                (3 * block_index + 2, low + r_j * (high - low)).into();
                            output_index += 1;
                        }
                    }
                    debug_assert_eq!(output_index, output_slice.len())
                });

            std::mem::swap(
                &mut self.interleaved_poly.bound_coeffs,
                &mut self.interleaved_poly.binding_scratch_space,
            );
        }

        // Bind eq_poly for next round
        self.split_eq_poly.bind(r_j);
    }

    fn expected_output_claim(
        &self,
        accumulator: Option<std::rc::Rc<std::cell::RefCell<VerifierOpeningAccumulator<F>>>>,
        r_tail: &[F::Challenge],
    ) -> F {
        let tau = self
            .tau
            .as_ref()
            .expect("Verifier tau not set in OuterRemainingSumcheck");

        // Reconstruct full r = [r0] + r_tail, then reverse (outer is bound from the top)
        let mut r_full: Vec<F::Challenge> = Vec::with_capacity(1 + r_tail.len());
        r_full.push(self.r0_uniskip);
        r_full.extend_from_slice(r_tail);
        let r_reversed: Vec<F::Challenge> = r_full.into_iter().rev().collect();

        let acc_cell = accumulator.as_ref().expect("accumulator required");
        let acc_ref = acc_cell.borrow();
        let (_, claim_Az) = acc_ref
            .get_virtual_polynomial_opening(VirtualPolynomial::SpartanAz, SumcheckId::SpartanOuter);
        let (_, claim_Bz) = acc_ref
            .get_virtual_polynomial_opening(VirtualPolynomial::SpartanBz, SumcheckId::SpartanOuter);
        let (_, claim_Cz) = acc_ref
            .get_virtual_polynomial_opening(VirtualPolynomial::SpartanCz, SumcheckId::SpartanOuter);

        let tau_bound_rx = EqPolynomial::mle(tau, &r_reversed);
        tau_bound_rx * (claim_Az * claim_Bz - claim_Cz)
    }

    fn normalize_opening_point(
        &self,
        r_tail: &[F::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        // Construct full r and reverse to match outer convention
        let mut r_full: Vec<F::Challenge> = Vec::with_capacity(1 + r_tail.len());
        r_full.push(self.r0_uniskip);
        r_full.extend_from_slice(r_tail);
        let r_reversed: Vec<F::Challenge> = r_full.into_iter().rev().collect();
        OpeningPoint::new(r_reversed)
    }

    fn cache_openings_prover(
        &self,
        accumulator: std::rc::Rc<std::cell::RefCell<ProverOpeningAccumulator<F>>>,
        transcript: &mut T,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        // Append Az, Bz, Cz claims and corresponding opening point
        let claims = self.final_sumcheck_evals();
        <T as crate::transcripts::Transcript>::append_scalars(transcript, &claims);
        let mut acc = accumulator.borrow_mut();
        acc.append_virtual(
            transcript,
            VirtualPolynomial::SpartanAz,
            SumcheckId::SpartanOuter,
            opening_point.clone(),
            claims[0],
        );
        acc.append_virtual(
            transcript,
            VirtualPolynomial::SpartanBz,
            SumcheckId::SpartanOuter,
            opening_point.clone(),
            claims[1],
        );
        acc.append_virtual(
            transcript,
            VirtualPolynomial::SpartanCz,
            SumcheckId::SpartanOuter,
            opening_point.clone(),
            claims[2],
        );

        // Handle witness openings at r_cycle
        let num_cycles = self.trace.len().next_power_of_two();
        let num_cycles_bits = num_cycles.ilog2() as usize;
        let r_rev = opening_point.r.clone();
        let (r_cycle, _rx_var) = r_rev.split_at(num_cycles_bits);

        // Compute claimed witness evals and append commitments and virtuals
        let claimed_witness_evals = compute_claimed_witness_evals::<F>(&self.preprocess, &self.trace, r_cycle);
        let committed_polys: Vec<_> = COMMITTED_R1CS_INPUTS
            .iter()
            .map(|input| CommittedPolynomial::try_from(input).ok().unwrap())
            .collect();
        let committed_poly_claims: Vec<_> = COMMITTED_R1CS_INPUTS
            .iter()
            .map(|input| claimed_witness_evals[input.to_index()])
            .collect();
        acc.append_dense(
            transcript,
            committed_polys,
            SumcheckId::SpartanOuter,
            r_cycle.to_vec(),
            &committed_poly_claims,
        );
        for (input, eval) in ALL_R1CS_INPUTS.iter().zip(claimed_witness_evals.iter()) {
            if !COMMITTED_R1CS_INPUTS.contains(input) {
                acc.append_virtual(
                    transcript,
                    VirtualPolynomial::try_from(input).ok().unwrap(),
                    SumcheckId::SpartanOuter,
                    OpeningPoint::new(r_cycle.to_vec()),
                    *eval,
                );
            }
        }
    }

    fn cache_openings_verifier(
        &self,
        accumulator: std::rc::Rc<std::cell::RefCell<VerifierOpeningAccumulator<F>>>,
        transcript: &mut T,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let mut acc = accumulator.borrow_mut();
        // Populate Az, Bz, Cz openings at the full outer opening point
        acc.append_virtual(
            transcript,
            VirtualPolynomial::SpartanAz,
            SumcheckId::SpartanOuter,
            opening_point.clone(),
        );
        acc.append_virtual(
            transcript,
            VirtualPolynomial::SpartanBz,
            SumcheckId::SpartanOuter,
            opening_point.clone(),
        );
        acc.append_virtual(
            transcript,
            VirtualPolynomial::SpartanCz,
            SumcheckId::SpartanOuter,
            opening_point.clone(),
        );

        // Append witness openings at r_cycle (no claims at verifier)
        let num_cycles_bits = self.split_eq_poly.E_out_current_len().ilog2() as usize;
        let (r_cycle, _rx_var) = opening_point.r.split_at(num_cycles_bits);
        let committed_polys: Vec<_> = COMMITTED_R1CS_INPUTS
            .iter()
            .map(|input| CommittedPolynomial::try_from(input).ok().unwrap())
            .collect();
        acc.append_dense(
            transcript,
            committed_polys,
            SumcheckId::SpartanOuter,
            r_cycle.to_vec(),
        );
        ALL_R1CS_INPUTS.iter().for_each(|input| {
            if !COMMITTED_R1CS_INPUTS.contains(input) {
                acc.append_virtual(
                    transcript,
                    VirtualPolynomial::try_from(input).ok().unwrap(),
                    SumcheckId::SpartanOuter,
                    OpeningPoint::new(r_cycle.to_vec()),
                );
            }
        });
    }
}
