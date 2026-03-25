//! Program-image (initial RAM) claim reduction.
//!
//! In committed bytecode mode, Stage 4 consumes prover-supplied scalar claims for the
//! program-image contribution to `Val_init(r_address)` without materializing the initial RAM.
//! This sumcheck binds those scalars to a trusted commitment to the program-image words polynomial.

use allocative::Allocative;
use std::cell::RefCell;

use crate::field::JoltField;
use crate::poly::commitment::dory::DoryGlobals;
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation};
#[cfg(feature = "zk")]
use crate::poly::opening_proof::OpeningId;
use crate::poly::opening_proof::{
    OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
    VerifierOpeningAccumulator, BIG_ENDIAN, LITTLE_ENDIAN,
};
use crate::poly::unipoly::UniPoly;
#[cfg(feature = "zk")]
use crate::subprotocols::blindfold::{InputClaimConstraint, OutputClaimConstraint, ValueSource};
use crate::subprotocols::sumcheck_prover::SumcheckInstanceProver;
use crate::subprotocols::sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier};
use crate::transcripts::Transcript;
use crate::utils::math::Math;
use crate::zkvm::claim_reductions::{
    permute_precommitted_polys, precommitted_skip_round_scale, PrecomittedParams,
    PrecomittedProver, PrecommittedClaimReduction, PrecommittedPhase,
    PrecommittedSchedulingReference, TWO_PHASE_DEGREE_BOUND,
};
use crate::zkvm::ram::remap_address;
use crate::zkvm::witness::{CommittedPolynomial, VirtualPolynomial};
use tracer::JoltDevice;

#[derive(Clone, Allocative)]
pub struct ProgramImageClaimReductionParams<F: JoltField> {
    pub phase: PrecommittedPhase,
    pub precommitted: PrecommittedClaimReduction<F>,
    pub prog_col_vars: usize,
    pub prog_row_vars: usize,
    pub ram_num_vars: usize,
    pub start_index: usize,
    pub padded_len_words: usize,
    pub m: usize,
    pub r_addr_rw: Vec<F::Challenge>,
    pub shifted_eq_coeffs: Vec<F>,
}

impl<F: JoltField> ProgramImageClaimReductionParams<F> {
    pub fn num_address_phase_rounds(&self) -> usize {
        self.precommitted.num_address_phase_rounds()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        program_io: &JoltDevice,
        ram_min_bytecode_address: u64,
        padded_len_words: usize,
        ram_K: usize,
        scheduling_reference: PrecommittedSchedulingReference,
        accumulator: &dyn OpeningAccumulator<F>,
        _transcript: &mut impl Transcript,
    ) -> Self {
        let ram_num_vars = ram_K.log_2();
        let start_index =
            remap_address(ram_min_bytecode_address, &program_io.memory_layout).unwrap() as usize;
        let m = padded_len_words.log_2();
        debug_assert!(padded_len_words.is_power_of_two());
        debug_assert!(padded_len_words > 0);
        let (prog_col_vars, prog_row_vars) = DoryGlobals::balanced_sigma_nu(m);
        let precommitted =
            PrecommittedClaimReduction::new(prog_row_vars, prog_col_vars, scheduling_reference);

        let (r_rw, _) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RamVal,
            SumcheckId::RamReadWriteChecking,
        );
        let (r_addr_rw, _) = r_rw.split_at(ram_num_vars);
        let shifted_eq_coeffs =
            shifted_program_image_eq_slice::<F>(&r_addr_rw.r, start_index, padded_len_words);

        Self {
            phase: PrecommittedPhase::CycleVariables,
            precommitted,
            prog_col_vars,
            prog_row_vars,
            ram_num_vars,
            start_index,
            padded_len_words,
            m,
            r_addr_rw: r_addr_rw.r,
            shifted_eq_coeffs,
        }
    }
}

impl<F: JoltField> ProgramImageClaimReductionParams<F> {
    fn is_cycle_phase(&self) -> bool {
        self.phase == PrecommittedPhase::CycleVariables
    }

    pub fn transition_to_address_phase(&mut self) {
        self.phase = PrecommittedPhase::AddressVariables;
    }

    pub fn round_offset(&self, max_num_rounds: usize) -> usize {
        self.precommitted
            .round_offset(self.is_cycle_phase(), max_num_rounds)
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for ProgramImageClaimReductionParams<F> {
    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        match self.phase {
            PrecommittedPhase::CycleVariables => {
                // Scalar claims were staged in Stage 4 as virtual openings.
                accumulator
                    .get_virtual_polynomial_opening(
                        VirtualPolynomial::ProgramImageInitContributionRw,
                        SumcheckId::RamValCheck,
                    )
                    .1
            }
            PrecommittedPhase::AddressVariables => {
                accumulator
                    .get_committed_polynomial_opening(
                        CommittedPolynomial::ProgramImageInit,
                        SumcheckId::ProgramImageClaimReductionCyclePhase,
                    )
                    .1
            }
        }
    }

    fn degree(&self) -> usize {
        TWO_PHASE_DEGREE_BOUND
    }

    fn num_rounds(&self) -> usize {
        self.precommitted
            .num_rounds_for_phase(self.is_cycle_phase())
    }

    fn normalize_opening_point(&self, challenges: &[F::Challenge]) -> OpeningPoint<BIG_ENDIAN, F> {
        self.precommitted
            .normalize_opening_point(self.is_cycle_phase(), challenges)
    }

    #[cfg(feature = "zk")]
    fn input_claim_constraint(&self) -> InputClaimConstraint {
        match self.phase {
            PrecommittedPhase::CycleVariables => InputClaimConstraint::direct(OpeningId::virt(
                VirtualPolynomial::ProgramImageInitContributionRw,
                SumcheckId::RamValCheck,
            )),
            PrecommittedPhase::AddressVariables => {
                InputClaimConstraint::direct(OpeningId::committed(
                    CommittedPolynomial::ProgramImageInit,
                    SumcheckId::ProgramImageClaimReductionCyclePhase,
                ))
            }
        }
    }

    #[cfg(feature = "zk")]
    fn input_constraint_challenge_values(&self, _: &dyn OpeningAccumulator<F>) -> Vec<F> {
        Vec::new()
    }

    #[cfg(feature = "zk")]
    fn output_claim_constraint(&self) -> Option<OutputClaimConstraint> {
        match self.phase {
            PrecommittedPhase::CycleVariables => {
                Some(OutputClaimConstraint::direct(OpeningId::committed(
                    CommittedPolynomial::ProgramImageInit,
                    SumcheckId::ProgramImageClaimReductionCyclePhase,
                )))
            }
            PrecommittedPhase::AddressVariables => Some(OutputClaimConstraint::linear(vec![(
                ValueSource::Challenge(0),
                ValueSource::Opening(OpeningId::committed(
                    CommittedPolynomial::ProgramImageInit,
                    SumcheckId::ProgramImageClaimReduction,
                )),
            )])),
        }
    }

    #[cfg(feature = "zk")]
    fn output_constraint_challenge_values(&self, sumcheck_challenges: &[F::Challenge]) -> Vec<F> {
        match self.phase {
            PrecommittedPhase::CycleVariables => vec![],
            PrecommittedPhase::AddressVariables => {
                let opening_point = self.normalize_opening_point(sumcheck_challenges);
                let eq_combined = eval_shifted_eq_poly_at_opening_point::<F>(
                    &self.r_addr_rw,
                    self.start_index,
                    &opening_point.r,
                );
                debug_assert_eq!(
                    eq_combined,
                    evaluate_shifted_eq_poly::<F, _>(&self.shifted_eq_coeffs, &opening_point.r),
                    "program_image eq_slice optimized evaluation mismatch"
                );
                let scale: F = precommitted_skip_round_scale(&self.precommitted);
                vec![eq_combined * scale]
            }
        }
    }
}

impl<F: JoltField> PrecomittedParams<F> for ProgramImageClaimReductionParams<F> {
    fn is_cycle_phase(&self) -> bool {
        self.phase == PrecommittedPhase::CycleVariables
    }

    fn is_cycle_phase_round(&self, round: usize) -> bool {
        self.precommitted.is_cycle_phase_round(round)
    }

    fn is_address_phase_round(&self, round: usize) -> bool {
        self.precommitted.is_address_phase_round(round)
    }

    fn cycle_alignment_rounds(&self) -> usize {
        self.precommitted.cycle_alignment_rounds()
    }

    fn address_alignment_rounds(&self) -> usize {
        self.precommitted.address_alignment_rounds()
    }

    fn record_cycle_challenge(&mut self, challenge: F::Challenge) {
        self.precommitted.record_cycle_challenge(challenge);
    }
}

#[derive(Allocative)]
pub struct ProgramImageClaimReductionProver<F: JoltField> {
    core: PrecomittedProver<F, ProgramImageClaimReductionParams<F>>,
}

fn shifted_program_image_eq_slice<F>(
    r_addr: &[F::Challenge],
    start_index: usize,
    padded_len_words: usize,
) -> Vec<F>
where
    F: JoltField + std::ops::Mul<F::Challenge, Output = F> + std::ops::SubAssign<F>,
{
    let mut eq_slice = Vec::with_capacity(padded_len_words);
    let mut idx = start_index;
    let mut remaining = padded_len_words;

    while remaining > 0 {
        let (block_size, block_evals) =
            EqPolynomial::<F>::evals_for_max_aligned_block(r_addr, idx, remaining);
        eq_slice.extend(block_evals);
        idx += block_size;
        remaining -= block_size;
    }

    eq_slice
}

fn evaluate_shifted_eq_poly<F, C>(shifted_eq_coeffs: &[F], opening_point: &[C]) -> F
where
    C: Copy + Send + Sync + Into<F> + crate::field::ChallengeFieldOps<F>,
    F: JoltField + crate::field::FieldChallengeOps<C>,
{
    MultilinearPolynomial::from(shifted_eq_coeffs.to_vec()).evaluate(opening_point)
}

impl<F: JoltField> ProgramImageClaimReductionProver<F> {
    pub fn params(&self) -> &ProgramImageClaimReductionParams<F> {
        self.core.params()
    }

    pub fn transition_to_address_phase(&mut self) {
        self.core.params_mut().transition_to_address_phase();
    }

    #[tracing::instrument(skip_all, name = "ProgramImageClaimReductionProver::initialize")]
    pub fn initialize(
        params: ProgramImageClaimReductionParams<F>,
        program_image_words_padded: Vec<u64>,
    ) -> Self {
        debug_assert_eq!(program_image_words_padded.len(), params.padded_len_words);
        debug_assert_eq!(params.padded_len_words, 1usize << params.m);

        let eq_slice = permute_precommitted_polys(
            vec![params.shifted_eq_coeffs.clone()],
            &params.precommitted,
        )
        .into_iter()
        .next()
        .expect("expected one permuted shifted eq polynomial");

        // Permute ProgramWord and eq_slice so low-to-high binding follows the two-phase
        // schedule while preserving top-left projection semantics against the joint point.
        let program_word: MultilinearPolynomial<F> = {
            let mut permuted =
                permute_precommitted_polys(vec![program_image_words_padded], &params.precommitted)
                    .into_iter();
            permuted
                .next()
                .expect("expected one permuted program image polynomial")
        };

        Self {
            core: PrecomittedProver::new(params, program_word, eq_slice),
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T>
    for ProgramImageClaimReductionProver<F>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        self.core.params()
    }

    fn round_offset(&self, max_num_rounds: usize) -> usize {
        self.core.params().round_offset(max_num_rounds)
    }

    fn compute_message(&mut self, round: usize, previous_claim: F) -> UniPoly<F> {
        self.core.compute_message(round, previous_claim)
    }

    fn ingest_challenge(&mut self, r_j: F::Challenge, round: usize) {
        self.core.ingest_challenge(r_j, round);
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let params = self.core.params();
        let opening_point = params.normalize_opening_point(sumcheck_challenges);
        if params.phase == PrecommittedPhase::CycleVariables {
            accumulator.append_dense(
                CommittedPolynomial::ProgramImageInit,
                SumcheckId::ProgramImageClaimReductionCyclePhase,
                // This is a phase-boundary intermediate claim, not a real program-image opening.
                // Keep a sentinel point so it cannot alias with the final opening claim.
                vec![],
                self.core.cycle_intermediate_claim(),
            );
        }

        if let Some(claim) = self.core.final_claim_if_ready() {
            accumulator.append_dense(
                CommittedPolynomial::ProgramImageInit,
                SumcheckId::ProgramImageClaimReduction,
                opening_point.r,
                claim,
            );
        }
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut allocative::FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

pub struct ProgramImageClaimReductionVerifier<F: JoltField> {
    pub params: RefCell<ProgramImageClaimReductionParams<F>>,
}

impl<F: JoltField> ProgramImageClaimReductionVerifier<F> {
    pub fn new(params: ProgramImageClaimReductionParams<F>) -> Self {
        Self {
            params: RefCell::new(params),
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T>
    for ProgramImageClaimReductionVerifier<F>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        unsafe { &*self.params.as_ptr() }
    }

    fn round_offset(&self, max_num_rounds: usize) -> usize {
        let params = self.params.borrow();
        params.round_offset(max_num_rounds)
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let params = self.params.borrow();
        match params.phase {
            PrecommittedPhase::CycleVariables => {
                accumulator
                    .get_committed_polynomial_opening(
                        CommittedPolynomial::ProgramImageInit,
                        SumcheckId::ProgramImageClaimReductionCyclePhase,
                    )
                    .1
            }
            PrecommittedPhase::AddressVariables => {
                let opening_point = params.normalize_opening_point(sumcheck_challenges);
                debug_assert_eq!(opening_point.len(), params.m);
                let pw_eval = accumulator
                    .get_committed_polynomial_opening(
                        CommittedPolynomial::ProgramImageInit,
                        SumcheckId::ProgramImageClaimReduction,
                    )
                    .1;
                let eq_combined = eval_shifted_eq_poly_at_opening_point::<F>(
                    &params.r_addr_rw,
                    params.start_index,
                    &opening_point.r,
                );
                debug_assert_eq!(
                    eq_combined,
                    evaluate_shifted_eq_poly::<F, _>(&params.shifted_eq_coeffs, &opening_point.r),
                    "program_image eq_slice optimized evaluation mismatch"
                );
                let scale: F = precommitted_skip_round_scale(&params.precommitted);
                pw_eval * eq_combined * scale
            }
        }
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let mut params = self.params.borrow_mut();
        let opening_point = params.normalize_opening_point(sumcheck_challenges);
        if params.phase == PrecommittedPhase::CycleVariables {
            accumulator.append_dense(
                CommittedPolynomial::ProgramImageInit,
                SumcheckId::ProgramImageClaimReductionCyclePhase,
                // Match prover behavior: the cycle-phase intermediate claim is not a real opening.
                vec![],
            );
            let opening_point_le: OpeningPoint<LITTLE_ENDIAN, F> = opening_point.match_endianness();
            params
                .precommitted
                .set_cycle_var_challenges(opening_point_le.r);
        }

        if params.phase == PrecommittedPhase::AddressVariables
            || params.num_address_phase_rounds() == 0
        {
            accumulator.append_dense(
                CommittedPolynomial::ProgramImageInit,
                SumcheckId::ProgramImageClaimReduction,
                opening_point.r,
            );
        }
    }
}

fn eval_shifted_eq_poly_at_opening_point<F>(
    r_addr_be: &[F::Challenge],
    start_index: usize,
    opening_point_be: &[F::Challenge],
) -> F
where
    F: JoltField,
{
    let ell = r_addr_be.len();
    let m = opening_point_be.len();
    debug_assert!(m <= ell);

    let challenge_for_old_lsb = |old_lsb: usize| -> F {
        debug_assert!(old_lsb < m);
        opening_point_be[m - 1 - old_lsb].into()
    };

    // Match the current verifier path exactly: `opening_point_be` is already arranged in the
    // variable order expected by `evaluate_shifted_eq_poly`.
    let mut dp0 = F::one();
    let mut dp1 = F::zero();

    for old_lsb in 0..ell {
        let start_bit = ((start_index >> old_lsb) & 1) as u8;
        let r_addr_bit: F = r_addr_be[ell - 1 - old_lsb].into();
        let k0 = F::one() - r_addr_bit;
        let k1 = r_addr_bit;
        let y_var = old_lsb < m;
        let r_y = if y_var {
            challenge_for_old_lsb(old_lsb)
        } else {
            F::zero()
        };

        let mut next_dp0 = F::zero();
        let mut next_dp1 = F::zero();

        let update_state = |weight: F, carry: u8, next_dp0: &mut F, next_dp1: &mut F| {
            if weight.is_zero() {
                return;
            }

            if y_var {
                let sum0 = start_bit + carry;
                let k_bit0 = sum0 & 1;
                let carry0 = (sum0 >> 1) & 1;
                let addr_factor0 = if k_bit0 == 1 { k1 } else { k0 };
                let y_factor0 = F::one() - r_y;
                if carry0 == 0 {
                    *next_dp0 += weight * addr_factor0 * y_factor0;
                } else {
                    *next_dp1 += weight * addr_factor0 * y_factor0;
                }

                let sum1 = start_bit + carry + 1;
                let k_bit1 = sum1 & 1;
                let carry1 = (sum1 >> 1) & 1;
                let addr_factor1 = if k_bit1 == 1 { k1 } else { k0 };
                if carry1 == 0 {
                    *next_dp0 += weight * addr_factor1 * r_y;
                } else {
                    *next_dp1 += weight * addr_factor1 * r_y;
                }
            } else {
                let sum0 = start_bit + carry;
                let k_bit0 = sum0 & 1;
                let carry0 = (sum0 >> 1) & 1;
                let addr_factor0 = if k_bit0 == 1 { k1 } else { k0 };
                if carry0 == 0 {
                    *next_dp0 += weight * addr_factor0;
                } else {
                    *next_dp1 += weight * addr_factor0;
                }
            }
        };

        update_state(dp0, 0, &mut next_dp0, &mut next_dp1);
        update_state(dp1, 1, &mut next_dp0, &mut next_dp1);
        dp0 = next_dp0;
        dp1 = next_dp1;
    }

    dp0 + dp1
}
