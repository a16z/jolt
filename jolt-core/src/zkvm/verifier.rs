use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
use std::sync::Arc;

use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::commitment::dory::{DoryContext, DoryGlobals};
use crate::subprotocols::sumcheck::BatchedSumcheck;
use crate::zkvm::bytecode::chunks::total_lanes;
use crate::zkvm::bytecode::{BytecodePreprocessing, TrustedBytecodeCommitments, VerifierBytecode};
use crate::zkvm::claim_reductions::advice::ReductionPhase;
use crate::zkvm::claim_reductions::RegistersClaimReductionSumcheckVerifier;
use crate::zkvm::config::BytecodeMode;
use crate::zkvm::config::OneHotParams;
#[cfg(feature = "prover")]
use crate::zkvm::prover::JoltProverPreprocessing;
use crate::zkvm::ram::val_final::ValFinalSumcheckVerifier;
use crate::zkvm::ram::RAMPreprocessing;
use crate::zkvm::witness::all_committed_polynomials;
use crate::zkvm::Serializable;
use crate::zkvm::{
    bytecode::read_raf_checking::{
        BytecodeReadRafAddressSumcheckVerifier, BytecodeReadRafCycleSumcheckVerifier,
        BytecodeReadRafSumcheckParams,
    },
    claim_reductions::{
        AdviceClaimReductionVerifier, AdviceKind, BytecodeClaimReductionParams,
        BytecodeClaimReductionVerifier, BytecodeReductionPhase,
        HammingWeightClaimReductionVerifier, IncClaimReductionSumcheckVerifier,
        InstructionLookupsClaimReductionSumcheckVerifier, RamRaClaimReductionSumcheckVerifier,
    },
    fiat_shamir_preamble,
    instruction_lookups::{
        ra_virtual::RaSumcheckVerifier as LookupsRaSumcheckVerifier,
        read_raf_checking::InstructionReadRafSumcheckVerifier,
    },
    proof_serialization::JoltProof,
    r1cs::key::UniformSpartanKey,
    ram::{
        hamming_booleanity::HammingBooleanitySumcheckVerifier,
        output_check::OutputSumcheckVerifier, ra_virtual::RamRaVirtualSumcheckVerifier,
        raf_evaluation::RafEvaluationSumcheckVerifier as RamRafEvaluationSumcheckVerifier,
        read_write_checking::RamReadWriteCheckingVerifier,
        val_evaluation::ValEvaluationSumcheckVerifier as RamValEvaluationSumcheckVerifier,
        verifier_accumulate_advice,
    },
    registers::{
        read_write_checking::RegistersReadWriteCheckingVerifier,
        val_evaluation::ValEvaluationSumcheckVerifier as RegistersValEvaluationSumcheckVerifier,
    },
    spartan::{
        instruction_input::InstructionInputSumcheckVerifier, outer::OuterRemainingSumcheckVerifier,
        product::ProductVirtualRemainderVerifier, shift::ShiftSumcheckVerifier,
        verify_stage1_uni_skip, verify_stage2_uni_skip,
    },
    ProverDebugInfo,
};
use crate::{
    field::JoltField,
    poly::opening_proof::{
        compute_advice_lagrange_factor, DoryOpeningState, OpeningAccumulator, OpeningPoint,
        SumcheckId, VerifierOpeningAccumulator,
    },
    pprof_scope,
    subprotocols::{
        booleanity::{
            BooleanityAddressSumcheckVerifier, BooleanityCycleSumcheckVerifier,
            BooleanitySumcheckParams,
        },
        sumcheck_verifier::SumcheckInstanceVerifier,
    },
    transcripts::Transcript,
    utils::{errors::ProofVerifyError, math::Math},
    zkvm::witness::CommittedPolynomial,
};
use anyhow::Context;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use common::jolt_device::MemoryLayout;
use itertools::Itertools;
use tracer::JoltDevice;

pub struct JoltVerifier<
    'a,
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    ProofTranscript: Transcript,
> {
    pub trusted_advice_commitment: Option<PCS::Commitment>,
    pub program_io: JoltDevice,
    pub proof: JoltProof<F, PCS, ProofTranscript>,
    pub preprocessing: &'a JoltVerifierPreprocessing<F, PCS>,
    pub transcript: ProofTranscript,
    pub opening_accumulator: VerifierOpeningAccumulator<F>,
    /// The advice claim reduction sumcheck effectively spans two stages (6 and 7).
    /// Cache the verifier state here between stages.
    advice_reduction_verifier_trusted: Option<AdviceClaimReductionVerifier<F>>,
    /// The advice claim reduction sumcheck effectively spans two stages (6 and 7).
    /// Cache the verifier state here between stages.
    advice_reduction_verifier_untrusted: Option<AdviceClaimReductionVerifier<F>>,
    /// The bytecode claim reduction sumcheck effectively spans two stages (6b and 7).
    /// Cache the verifier state here between stages.
    bytecode_reduction_verifier: Option<BytecodeClaimReductionVerifier<F>>,
    pub spartan_key: UniformSpartanKey<F>,
    pub one_hot_params: OneHotParams,
}

impl<'a, F: JoltField, PCS: CommitmentScheme<Field = F>, ProofTranscript: Transcript>
    JoltVerifier<'a, F, PCS, ProofTranscript>
{
    pub fn new(
        preprocessing: &'a JoltVerifierPreprocessing<F, PCS>,
        proof: JoltProof<F, PCS, ProofTranscript>,
        mut program_io: JoltDevice,
        trusted_advice_commitment: Option<PCS::Commitment>,
        _debug_info: Option<ProverDebugInfo<F, ProofTranscript, PCS>>,
    ) -> Result<Self, ProofVerifyError> {
        // Memory layout checks
        if program_io.memory_layout != preprocessing.shared.memory_layout {
            return Err(ProofVerifyError::MemoryLayoutMismatch);
        }
        if program_io.inputs.len() > preprocessing.shared.memory_layout.max_input_size as usize {
            return Err(ProofVerifyError::InputTooLarge);
        }
        if program_io.outputs.len() > preprocessing.shared.memory_layout.max_output_size as usize {
            return Err(ProofVerifyError::OutputTooLarge);
        }

        // truncate trailing zeros on device outputs
        program_io.outputs.truncate(
            program_io
                .outputs
                .iter()
                .rposition(|&b| b != 0)
                .map_or(0, |pos| pos + 1),
        );

        let mut opening_accumulator = VerifierOpeningAccumulator::new(proof.trace_length.log_2());
        // Populate claims in the verifier accumulator
        for (key, (_, claim)) in &proof.opening_claims.0 {
            opening_accumulator
                .openings
                .insert(*key, (OpeningPoint::default(), *claim));
        }

        #[cfg(test)]
        let mut transcript = ProofTranscript::new(b"Jolt");
        #[cfg(not(test))]
        let transcript = ProofTranscript::new(b"Jolt");

        #[cfg(test)]
        {
            if let Some(debug_info) = _debug_info {
                transcript.compare_to(debug_info.transcript);
                opening_accumulator.compare_to(debug_info.opening_accumulator);
            }
        }

        let spartan_key = UniformSpartanKey::new(proof.trace_length.next_power_of_two());

        // Validate configs from the proof
        proof
            .one_hot_config
            .validate()
            .map_err(ProofVerifyError::InvalidOneHotConfig)?;

        proof
            .rw_config
            .validate(proof.trace_length.log_2(), proof.ram_K.log_2())
            .map_err(ProofVerifyError::InvalidReadWriteConfig)?;

        // Construct full params from the validated config
        let one_hot_params =
            OneHotParams::from_config(&proof.one_hot_config, proof.bytecode_K, proof.ram_K);

        if proof.bytecode_mode == BytecodeMode::Committed {
            let committed = preprocessing.bytecode.as_committed()?;
            if committed.log_k_chunk != proof.one_hot_config.log_k_chunk {
                return Err(ProofVerifyError::InvalidBytecodeConfig(format!(
                    "bytecode log_k_chunk mismatch: commitments={}, proof={}",
                    committed.log_k_chunk, proof.one_hot_config.log_k_chunk
                )));
            }
            if committed.bytecode_len != preprocessing.shared.bytecode_size {
                return Err(ProofVerifyError::InvalidBytecodeConfig(format!(
                    "bytecode length mismatch: commitments={}, shared={}",
                    committed.bytecode_len, preprocessing.shared.bytecode_size
                )));
            }
            let k_chunk = 1usize << (committed.log_k_chunk as usize);
            let expected_chunks = total_lanes().div_ceil(k_chunk);
            if committed.commitments.len() != expected_chunks {
                return Err(ProofVerifyError::InvalidBytecodeConfig(format!(
                    "expected {expected_chunks} bytecode commitments, got {}",
                    committed.commitments.len()
                )));
            }
        }

        Ok(Self {
            trusted_advice_commitment,
            program_io,
            proof,
            preprocessing,
            transcript,
            opening_accumulator,
            advice_reduction_verifier_trusted: None,
            advice_reduction_verifier_untrusted: None,
            bytecode_reduction_verifier: None,
            spartan_key,
            one_hot_params,
        })
    }

    #[tracing::instrument(skip_all)]
    pub fn verify(mut self) -> Result<(), anyhow::Error> {
        let _pprof_verify = pprof_scope!("verify");

        fiat_shamir_preamble(
            &self.program_io,
            self.proof.ram_K,
            self.proof.trace_length,
            &mut self.transcript,
        );

        // Append commitments to transcript
        for commitment in &self.proof.commitments {
            self.transcript.append_serializable(commitment);
        }
        // Append untrusted advice commitment to transcript
        if let Some(ref untrusted_advice_commitment) = self.proof.untrusted_advice_commitment {
            self.transcript
                .append_serializable(untrusted_advice_commitment);
        }
        // Append trusted advice commitment to transcript
        if let Some(ref trusted_advice_commitment) = self.trusted_advice_commitment {
            self.transcript
                .append_serializable(trusted_advice_commitment);
        }
        if self.proof.bytecode_mode == BytecodeMode::Committed {
            let trusted = self.preprocessing.bytecode.as_committed()?;
            for commitment in &trusted.commitments {
                self.transcript.append_serializable(commitment);
            }
        }

        self.verify_stage1()?;
        self.verify_stage2()?;
        self.verify_stage3()?;
        self.verify_stage4()?;
        self.verify_stage5()?;
        let (bytecode_read_raf_params, booleanity_params) = self.verify_stage6a()?;
        self.verify_stage6b(bytecode_read_raf_params, booleanity_params)?;
        self.verify_stage7()?;
        self.verify_stage8()?;

        Ok(())
    }

    fn verify_stage1(&mut self) -> Result<(), anyhow::Error> {
        let uni_skip_params = verify_stage1_uni_skip(
            &self.proof.stage1_uni_skip_first_round_proof,
            &self.spartan_key,
            &mut self.opening_accumulator,
            &mut self.transcript,
        )
        .context("Stage 1 univariate skip first round")?;

        let spartan_outer_remaining = OuterRemainingSumcheckVerifier::new(
            self.spartan_key,
            self.proof.trace_length,
            uni_skip_params,
            &self.opening_accumulator,
        );

        let _r_stage1 = BatchedSumcheck::verify(
            &self.proof.stage1_sumcheck_proof,
            vec![&spartan_outer_remaining],
            &mut self.opening_accumulator,
            &mut self.transcript,
        )
        .context("Stage 1")?;

        Ok(())
    }

    fn verify_stage2(&mut self) -> Result<(), anyhow::Error> {
        let uni_skip_params = verify_stage2_uni_skip(
            &self.proof.stage2_uni_skip_first_round_proof,
            &mut self.opening_accumulator,
            &mut self.transcript,
        )
        .context("Stage 2 univariate skip first round")?;

        let spartan_product_virtual_remainder = ProductVirtualRemainderVerifier::new(
            self.proof.trace_length,
            uni_skip_params,
            &self.opening_accumulator,
        );
        let ram_raf_evaluation = RamRafEvaluationSumcheckVerifier::new(
            &self.program_io.memory_layout,
            &self.one_hot_params,
            &self.opening_accumulator,
        );
        let ram_read_write_checking = RamReadWriteCheckingVerifier::new(
            &self.opening_accumulator,
            &mut self.transcript,
            &self.one_hot_params,
            self.proof.trace_length,
            &self.proof.rw_config,
        );
        let ram_output_check =
            OutputSumcheckVerifier::new(self.proof.ram_K, &self.program_io, &mut self.transcript);
        let instruction_claim_reduction = InstructionLookupsClaimReductionSumcheckVerifier::new(
            self.proof.trace_length,
            &self.opening_accumulator,
            &mut self.transcript,
        );

        let _r_stage2 = BatchedSumcheck::verify(
            &self.proof.stage2_sumcheck_proof,
            vec![
                &spartan_product_virtual_remainder,
                &ram_raf_evaluation,
                &ram_read_write_checking,
                &ram_output_check,
                &instruction_claim_reduction,
            ],
            &mut self.opening_accumulator,
            &mut self.transcript,
        )
        .context("Stage 2")?;

        Ok(())
    }

    fn verify_stage3(&mut self) -> Result<(), anyhow::Error> {
        let spartan_shift = ShiftSumcheckVerifier::new(
            self.proof.trace_length.log_2(),
            &self.opening_accumulator,
            &mut self.transcript,
        );
        let spartan_instruction_input =
            InstructionInputSumcheckVerifier::new(&self.opening_accumulator, &mut self.transcript);
        let spartan_registers_claim_reduction = RegistersClaimReductionSumcheckVerifier::new(
            self.proof.trace_length,
            &self.opening_accumulator,
            &mut self.transcript,
        );

        let _r_stage3 = BatchedSumcheck::verify(
            &self.proof.stage3_sumcheck_proof,
            vec![
                &spartan_shift as &dyn SumcheckInstanceVerifier<F, ProofTranscript>,
                &spartan_instruction_input,
                &spartan_registers_claim_reduction,
            ],
            &mut self.opening_accumulator,
            &mut self.transcript,
        )
        .context("Stage 3")?;

        Ok(())
    }

    fn verify_stage4(&mut self) -> Result<(), anyhow::Error> {
        verifier_accumulate_advice::<F>(
            self.proof.ram_K,
            &self.program_io,
            self.proof.untrusted_advice_commitment.is_some(),
            self.trusted_advice_commitment.is_some(),
            &mut self.opening_accumulator,
            &mut self.transcript,
            self.proof
                .rw_config
                .needs_single_advice_opening(self.proof.trace_length.log_2()),
        );
        let registers_read_write_checking = RegistersReadWriteCheckingVerifier::new(
            self.proof.trace_length,
            &self.opening_accumulator,
            &mut self.transcript,
            &self.proof.rw_config,
        );
        let ram_val_evaluation = RamValEvaluationSumcheckVerifier::new(
            &self.preprocessing.shared.ram,
            &self.program_io,
            self.proof.trace_length,
            self.proof.ram_K,
            &self.opening_accumulator,
        );
        let ram_val_final = ValFinalSumcheckVerifier::new(
            &self.preprocessing.shared.ram,
            &self.program_io,
            self.proof.trace_length,
            self.proof.ram_K,
            &self.opening_accumulator,
            &self.proof.rw_config,
        );

        let _r_stage4 = BatchedSumcheck::verify(
            &self.proof.stage4_sumcheck_proof,
            vec![
                &registers_read_write_checking as &dyn SumcheckInstanceVerifier<F, ProofTranscript>,
                &ram_val_evaluation,
                &ram_val_final,
            ],
            &mut self.opening_accumulator,
            &mut self.transcript,
        )
        .context("Stage 4")?;

        Ok(())
    }

    fn verify_stage5(&mut self) -> Result<(), anyhow::Error> {
        let n_cycle_vars = self.proof.trace_length.log_2();
        let registers_val_evaluation =
            RegistersValEvaluationSumcheckVerifier::new(&self.opening_accumulator);
        let ram_ra_reduction = RamRaClaimReductionSumcheckVerifier::new(
            self.proof.trace_length,
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
        );
        let lookups_read_raf = InstructionReadRafSumcheckVerifier::new(
            n_cycle_vars,
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
        );

        let _r_stage5 = BatchedSumcheck::verify(
            &self.proof.stage5_sumcheck_proof,
            vec![
                &registers_val_evaluation as &dyn SumcheckInstanceVerifier<F, ProofTranscript>,
                &ram_ra_reduction,
                &lookups_read_raf,
            ],
            &mut self.opening_accumulator,
            &mut self.transcript,
        )
        .context("Stage 5")?;

        Ok(())
    }

    fn verify_stage6a(
        &mut self,
    ) -> Result<
        (
            BytecodeReadRafSumcheckParams<F>,
            BooleanitySumcheckParams<F>,
        ),
        anyhow::Error,
    > {
        let n_cycle_vars = self.proof.trace_length.log_2();
        let bytecode_preprocessing = match self.proof.bytecode_mode {
            BytecodeMode::Committed => {
                // Ensure we have committed bytecode commitments for committed mode.
                let _ = self.preprocessing.bytecode.as_committed()?;
                None
            }
            BytecodeMode::Full => Some(self.preprocessing.bytecode.as_full()?.as_ref()),
        };
        let bytecode_read_raf = BytecodeReadRafAddressSumcheckVerifier::new(
            bytecode_preprocessing,
            n_cycle_vars,
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
            self.proof.bytecode_mode,
        )?;
        let booleanity_params = BooleanitySumcheckParams::new(
            n_cycle_vars,
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
        );
        let booleanity = BooleanityAddressSumcheckVerifier::new(booleanity_params);

        let instances: Vec<&dyn SumcheckInstanceVerifier<F, ProofTranscript>> =
            vec![&bytecode_read_raf, &booleanity];

        let _r_stage6a = BatchedSumcheck::verify(
            &self.proof.stage6a_sumcheck_proof,
            instances,
            &mut self.opening_accumulator,
            &mut self.transcript,
        )
        .context("Stage 6a")?;
        Ok((bytecode_read_raf.into_params(), booleanity.into_params()))
    }

    fn verify_stage6b(
        &mut self,
        bytecode_read_raf_params: BytecodeReadRafSumcheckParams<F>,
        booleanity_params: BooleanitySumcheckParams<F>,
    ) -> Result<(), anyhow::Error> {
        // Initialize Stage 6b cycle verifiers from scratch (Option B).
        let booleanity = BooleanityCycleSumcheckVerifier::new(booleanity_params);
        let ram_hamming_booleanity =
            HammingBooleanitySumcheckVerifier::new(&self.opening_accumulator);
        let ram_ra_virtual = RamRaVirtualSumcheckVerifier::new(
            self.proof.trace_length,
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
        );
        let lookups_ra_virtual = LookupsRaSumcheckVerifier::new(
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
        );
        let inc_reduction = IncClaimReductionSumcheckVerifier::new(
            self.proof.trace_length,
            &self.opening_accumulator,
            &mut self.transcript,
        );

        // Bytecode claim reduction (Phase 1 in Stage 6b): consumes Val_s(r_bc) from Stage 6a and
        // caches an intermediate claim for Stage 7.
        //
        // IMPORTANT: This must be sampled *after* other Stage 6b params (e.g. lookup/inc gammas),
        // to match the prover's transcript order.
        if self.proof.bytecode_mode == BytecodeMode::Committed {
            let bytecode_reduction_params = BytecodeClaimReductionParams::new(
                &bytecode_read_raf_params,
                &self.opening_accumulator,
                &mut self.transcript,
            );
            self.bytecode_reduction_verifier = Some(BytecodeClaimReductionVerifier::new(
                bytecode_reduction_params,
            ));
        } else {
            // Legacy mode: do not run the bytecode claim reduction.
            self.bytecode_reduction_verifier = None;
        }

        // Advice claim reduction (Phase 1 in Stage 6b): trusted and untrusted are separate instances.
        if self.trusted_advice_commitment.is_some() {
            self.advice_reduction_verifier_trusted = Some(AdviceClaimReductionVerifier::new(
                AdviceKind::Trusted,
                &self.program_io.memory_layout,
                self.proof.trace_length,
                &self.opening_accumulator,
                &mut self.transcript,
                self.proof
                    .rw_config
                    .needs_single_advice_opening(self.proof.trace_length.log_2()),
            ));
        }
        if self.proof.untrusted_advice_commitment.is_some() {
            self.advice_reduction_verifier_untrusted = Some(AdviceClaimReductionVerifier::new(
                AdviceKind::Untrusted,
                &self.program_io.memory_layout,
                self.proof.trace_length,
                &self.opening_accumulator,
                &mut self.transcript,
                self.proof
                    .rw_config
                    .needs_single_advice_opening(self.proof.trace_length.log_2()),
            ));
        }

        let bytecode_read_raf = BytecodeReadRafCycleSumcheckVerifier::new(bytecode_read_raf_params);

        let mut instances: Vec<&dyn SumcheckInstanceVerifier<F, ProofTranscript>> = vec![
            &bytecode_read_raf,
            &ram_hamming_booleanity,
            &booleanity,
            &ram_ra_virtual,
            &lookups_ra_virtual,
            &inc_reduction,
        ];
        if let Some(ref bytecode) = self.bytecode_reduction_verifier {
            instances.push(bytecode);
        }
        if let Some(ref advice) = self.advice_reduction_verifier_trusted {
            instances.push(advice);
        }
        if let Some(ref advice) = self.advice_reduction_verifier_untrusted {
            instances.push(advice);
        }

        let _r_stage6b = BatchedSumcheck::verify(
            &self.proof.stage6b_sumcheck_proof,
            instances,
            &mut self.opening_accumulator,
            &mut self.transcript,
        )
        .context("Stage 6b")?;

        Ok(())
    }

    /// Stage 7: HammingWeight claim reduction verification.
    fn verify_stage7(&mut self) -> Result<(), anyhow::Error> {
        // Create verifier for HammingWeightClaimReduction
        // (r_cycle and r_addr_bool are extracted from Booleanity opening internally)
        let hw_verifier = HammingWeightClaimReductionVerifier::new(
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
        );

        let mut instances: Vec<&dyn SumcheckInstanceVerifier<F, ProofTranscript>> =
            vec![&hw_verifier];

        if let Some(bytecode_reduction_verifier) = self.bytecode_reduction_verifier.as_mut() {
            bytecode_reduction_verifier.params.borrow_mut().phase =
                BytecodeReductionPhase::LaneVariables;
            instances.push(bytecode_reduction_verifier);
        }
        if let Some(advice_reduction_verifier_trusted) =
            self.advice_reduction_verifier_trusted.as_mut()
        {
            let mut params = advice_reduction_verifier_trusted.params.borrow_mut();
            if params.num_address_phase_rounds() > 0 {
                // Transition phase
                params.phase = ReductionPhase::AddressVariables;
                instances.push(advice_reduction_verifier_trusted);
            }
        }
        if let Some(advice_reduction_verifier_untrusted) =
            self.advice_reduction_verifier_untrusted.as_mut()
        {
            let mut params = advice_reduction_verifier_untrusted.params.borrow_mut();
            if params.num_address_phase_rounds() > 0 {
                // Transition phase
                params.phase = ReductionPhase::AddressVariables;
                instances.push(advice_reduction_verifier_untrusted);
            }
        }

        let _r_address_stage7 = BatchedSumcheck::verify(
            &self.proof.stage7_sumcheck_proof,
            instances,
            &mut self.opening_accumulator,
            &mut self.transcript,
        )
        .context("Stage 7")?;

        Ok(())
    }

    /// Stage 8: Dory batch opening verification.
    fn verify_stage8(&mut self) -> Result<(), anyhow::Error> {
        // Initialize DoryGlobals with the layout from the proof
        // This ensures the verifier uses the same layout as the prover
        let _guard = DoryGlobals::initialize_context(
            1 << self.one_hot_params.log_k_chunk,
            self.proof.trace_length.next_power_of_two(),
            DoryContext::Main,
            Some(self.proof.dory_layout),
        );

        // Get the unified opening point from HammingWeightClaimReduction
        // This contains (r_address_stage7 || r_cycle_stage6) in big-endian
        let (opening_point, _) = self.opening_accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::InstructionRa(0),
            SumcheckId::HammingWeightClaimReduction,
        );
        let log_k_chunk = self.one_hot_params.log_k_chunk;
        let r_address_stage7 = &opening_point.r[..log_k_chunk];

        // 1. Collect all (polynomial, claim) pairs
        let mut polynomial_claims = Vec::new();

        // Dense polynomials: RamInc and RdInc (from IncClaimReduction in Stage 6)
        let (_, ram_inc_claim) = self.opening_accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::RamInc,
            SumcheckId::IncClaimReduction,
        );
        let (_, rd_inc_claim) = self.opening_accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::RdInc,
            SumcheckId::IncClaimReduction,
        );

        // Apply Lagrange factor for dense polys
        // Note: r_address is in big-endian, Lagrange factor uses ∏(1 - r_i)
        let lagrange_factor: F = r_address_stage7.iter().map(|r| F::one() - *r).product();

        polynomial_claims.push((CommittedPolynomial::RamInc, ram_inc_claim * lagrange_factor));
        polynomial_claims.push((CommittedPolynomial::RdInc, rd_inc_claim * lagrange_factor));

        // Sparse polynomials: all RA polys (from HammingWeightClaimReduction)
        for i in 0..self.one_hot_params.instruction_d {
            let (_, claim) = self.opening_accumulator.get_committed_polynomial_opening(
                CommittedPolynomial::InstructionRa(i),
                SumcheckId::HammingWeightClaimReduction,
            );
            polynomial_claims.push((CommittedPolynomial::InstructionRa(i), claim));
        }
        for i in 0..self.one_hot_params.bytecode_d {
            let (_, claim) = self.opening_accumulator.get_committed_polynomial_opening(
                CommittedPolynomial::BytecodeRa(i),
                SumcheckId::HammingWeightClaimReduction,
            );
            polynomial_claims.push((CommittedPolynomial::BytecodeRa(i), claim));
        }
        for i in 0..self.one_hot_params.ram_d {
            let (_, claim) = self.opening_accumulator.get_committed_polynomial_opening(
                CommittedPolynomial::RamRa(i),
                SumcheckId::HammingWeightClaimReduction,
            );
            polynomial_claims.push((CommittedPolynomial::RamRa(i), claim));
        }

        // Advice polynomials: TrustedAdvice and UntrustedAdvice (from AdviceClaimReduction in Stage 6)
        // These are committed with smaller dimensions, so we apply Lagrange factors to embed
        // them in the top-left block of the main Dory matrix.
        if let Some((advice_point, advice_claim)) = self
            .opening_accumulator
            .get_advice_opening(AdviceKind::Trusted, SumcheckId::AdviceClaimReduction)
        {
            let lagrange_factor =
                compute_advice_lagrange_factor::<F>(&opening_point.r, &advice_point.r);
            polynomial_claims.push((
                CommittedPolynomial::TrustedAdvice,
                advice_claim * lagrange_factor,
            ));
        }

        if let Some((advice_point, advice_claim)) = self
            .opening_accumulator
            .get_advice_opening(AdviceKind::Untrusted, SumcheckId::AdviceClaimReduction)
        {
            let lagrange_factor =
                compute_advice_lagrange_factor::<F>(&opening_point.r, &advice_point.r);
            polynomial_claims.push((
                CommittedPolynomial::UntrustedAdvice,
                advice_claim * lagrange_factor,
            ));
        }

        // Bytecode chunk polynomials: committed in Bytecode context and embedded into the
        // main opening point by fixing the extra cycle variables to 0.
        if self.proof.bytecode_mode == BytecodeMode::Committed {
            let (bytecode_point, _) = self.opening_accumulator.get_committed_polynomial_opening(
                CommittedPolynomial::BytecodeChunk(0),
                SumcheckId::BytecodeClaimReduction,
            );
            let log_t = opening_point.r.len() - log_k_chunk;
            let log_k = bytecode_point.r.len() - log_k_chunk;
            if log_k > log_t {
                return Err(ProofVerifyError::InvalidBytecodeConfig(format!(
                    "bytecode folding requires log_T >= log_K (got log_T={log_t}, log_K={log_k})"
                ))
                .into());
            }
            #[cfg(test)]
            {
                if log_k == log_t {
                    assert_eq!(
                        bytecode_point.r, opening_point.r,
                        "BytecodeChunk opening point must equal unified opening point when log_K == log_T"
                    );
                } else {
                    let (r_lane_main, r_cycle_main) = opening_point.split_at(log_k_chunk);
                    let (r_lane_bc, r_cycle_bc) = bytecode_point.split_at(log_k_chunk);
                    debug_assert_eq!(r_lane_main.r, r_lane_bc.r);
                    debug_assert_eq!(&r_cycle_main.r[(log_t - log_k)..], r_cycle_bc.r.as_slice());
                }
            }
            let lagrange_factor =
                compute_advice_lagrange_factor::<F>(&opening_point.r, &bytecode_point.r);

            let num_chunks = total_lanes().div_ceil(self.one_hot_params.k_chunk);
            for i in 0..num_chunks {
                let (_, claim) = self.opening_accumulator.get_committed_polynomial_opening(
                    CommittedPolynomial::BytecodeChunk(i),
                    SumcheckId::BytecodeClaimReduction,
                );
                polynomial_claims.push((
                    CommittedPolynomial::BytecodeChunk(i),
                    claim * lagrange_factor,
                ));
            }
        }

        // 2. Sample gamma and compute powers for RLC
        let claims: Vec<F> = polynomial_claims.iter().map(|(_, c)| *c).collect();
        self.transcript.append_scalars(&claims);
        let gamma_powers: Vec<F> = self.transcript.challenge_scalar_powers(claims.len());

        // Build state for computing joint commitment/claim
        let state = DoryOpeningState {
            opening_point: opening_point.r.clone(),
            gamma_powers: gamma_powers.clone(),
            polynomial_claims,
        };

        // Build commitments map
        let mut commitments_map = HashMap::new();
        for (polynomial, commitment) in all_committed_polynomials(&self.one_hot_params)
            .into_iter()
            .zip_eq(&self.proof.commitments)
        {
            commitments_map.insert(polynomial, commitment.clone());
        }

        // Add advice commitments if they're part of the batch
        if let Some(ref commitment) = self.trusted_advice_commitment {
            if state
                .polynomial_claims
                .iter()
                .any(|(p, _)| *p == CommittedPolynomial::TrustedAdvice)
            {
                commitments_map.insert(CommittedPolynomial::TrustedAdvice, commitment.clone());
            }
        }
        if let Some(ref commitment) = self.proof.untrusted_advice_commitment {
            if state
                .polynomial_claims
                .iter()
                .any(|(p, _)| *p == CommittedPolynomial::UntrustedAdvice)
            {
                commitments_map.insert(CommittedPolynomial::UntrustedAdvice, commitment.clone());
            }
        }

        if self.proof.bytecode_mode == BytecodeMode::Committed {
            let committed = self.preprocessing.bytecode.as_committed()?;
            for (idx, commitment) in committed.commitments.iter().enumerate() {
                commitments_map
                    .entry(CommittedPolynomial::BytecodeChunk(idx))
                    .or_insert_with(|| commitment.clone());
            }
        }

        // Compute joint commitment: Σ γ_i · C_i
        let joint_commitment = self.compute_joint_commitment(&mut commitments_map, &state);

        // Compute joint claim: Σ γ_i · claim_i
        let joint_claim: F = gamma_powers
            .iter()
            .zip(claims.iter())
            .map(|(gamma, claim)| *gamma * claim)
            .sum();

        // Verify opening
        PCS::verify(
            &self.proof.joint_opening_proof,
            &self.preprocessing.generators,
            &mut self.transcript,
            &opening_point.r,
            &joint_claim,
            &joint_commitment,
        )
        .context("Stage 8")
    }

    /// Compute joint commitment for the batch opening.
    fn compute_joint_commitment(
        &self,
        commitment_map: &mut HashMap<CommittedPolynomial, PCS::Commitment>,
        state: &DoryOpeningState<F>,
    ) -> PCS::Commitment {
        // Accumulate gamma coefficients per polynomial
        let mut rlc_map = HashMap::new();
        for (gamma, (poly, _claim)) in state
            .gamma_powers
            .iter()
            .zip(state.polynomial_claims.iter())
        {
            *rlc_map.entry(*poly).or_insert(F::zero()) += *gamma;
        }

        let (coeffs, commitments): (Vec<F>, Vec<PCS::Commitment>) = rlc_map
            .into_iter()
            .map(|(k, v)| (v, commitment_map.remove(&k).unwrap()))
            .unzip();

        PCS::combine_commitments(&commitments, &coeffs)
    }
}

/// Shared preprocessing between prover and verifier.
///
/// **Note**: This struct does NOT contain the full bytecode data.
/// - Bytecode size K is stored here as the single source of truth.
/// - Full bytecode data is in `JoltProverPreprocessing.bytecode`.
/// - Verifier bytecode (Full or Committed) is in `JoltVerifierPreprocessing.bytecode`.
#[derive(Debug, Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct JoltSharedPreprocessing {
    pub bytecode_size: usize,
    pub ram: RAMPreprocessing,
    pub memory_layout: MemoryLayout,
    pub max_padded_trace_length: usize,
}

impl JoltSharedPreprocessing {
    /// Create shared preprocessing from bytecode.
    ///
    /// Bytecode size K is derived from `bytecode.bytecode.len()` (already padded).
    /// The caller is responsible for wrapping bytecode in `Arc` and passing to prover/verifier.
    #[tracing::instrument(skip_all, name = "JoltSharedPreprocessing::new")]
    pub fn new(
        bytecode: &BytecodePreprocessing,
        memory_layout: MemoryLayout,
        memory_init: Vec<(u64, u8)>,
        max_padded_trace_length: usize,
    ) -> JoltSharedPreprocessing {
        let ram = RAMPreprocessing::preprocess(memory_init);
        Self {
            bytecode_size: bytecode.bytecode.len(),
            ram,
            memory_layout,
            max_padded_trace_length,
        }
    }
}

#[derive(Debug, Clone)]
pub struct JoltVerifierPreprocessing<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    pub generators: PCS::VerifierSetup,
    pub shared: JoltSharedPreprocessing,
    /// Bytecode information for verification.
    ///
    /// In Full mode: contains full bytecode preprocessing (O(K) data).
    /// In Committed mode: contains only commitments (succinct).
    pub bytecode: VerifierBytecode<PCS>,
}

impl<F, PCS> CanonicalSerialize for JoltVerifierPreprocessing<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    fn serialize_with_mode<W: std::io::Write>(
        &self,
        mut writer: W,
        compress: ark_serialize::Compress,
    ) -> Result<(), ark_serialize::SerializationError> {
        self.generators.serialize_with_mode(&mut writer, compress)?;
        self.shared.serialize_with_mode(&mut writer, compress)?;
        self.bytecode.serialize_with_mode(&mut writer, compress)?;
        Ok(())
    }

    fn serialized_size(&self, compress: ark_serialize::Compress) -> usize {
        self.generators.serialized_size(compress)
            + self.shared.serialized_size(compress)
            + self.bytecode.serialized_size(compress)
    }
}

impl<F, PCS> ark_serialize::Valid for JoltVerifierPreprocessing<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    fn check(&self) -> Result<(), ark_serialize::SerializationError> {
        self.generators.check()?;
        self.shared.check()?;
        self.bytecode.check()
    }
}

impl<F, PCS> CanonicalDeserialize for JoltVerifierPreprocessing<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    fn deserialize_with_mode<R: std::io::Read>(
        mut reader: R,
        compress: ark_serialize::Compress,
        validate: ark_serialize::Validate,
    ) -> Result<Self, ark_serialize::SerializationError> {
        let generators =
            PCS::VerifierSetup::deserialize_with_mode(&mut reader, compress, validate)?;
        let shared =
            JoltSharedPreprocessing::deserialize_with_mode(&mut reader, compress, validate)?;
        let bytecode = VerifierBytecode::deserialize_with_mode(&mut reader, compress, validate)?;
        Ok(Self {
            generators,
            shared,
            bytecode,
        })
    }
}

impl<F, PCS> Serializable for JoltVerifierPreprocessing<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
}

impl<F, PCS> JoltVerifierPreprocessing<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    pub fn save_to_target_dir(&self, target_dir: &str) -> std::io::Result<()> {
        let filename = Path::new(target_dir).join("jolt_verifier_preprocessing.dat");
        let mut file = File::create(filename.as_path())?;
        let mut data = Vec::new();
        self.serialize_compressed(&mut data).unwrap();
        file.write_all(&data)?;
        Ok(())
    }

    pub fn read_from_target_dir(target_dir: &str) -> std::io::Result<Self> {
        let filename = Path::new(target_dir).join("jolt_verifier_preprocessing.dat");
        let mut file = File::open(filename.as_path())?;
        let mut data = Vec::new();
        file.read_to_end(&mut data)?;
        Ok(Self::deserialize_compressed(&*data).unwrap())
    }
}

impl<F: JoltField, PCS: CommitmentScheme<Field = F>> JoltVerifierPreprocessing<F, PCS> {
    /// Create verifier preprocessing in Full mode (verifier has full bytecode).
    #[tracing::instrument(skip_all, name = "JoltVerifierPreprocessing::new_full")]
    pub fn new_full(
        shared: JoltSharedPreprocessing,
        generators: PCS::VerifierSetup,
        bytecode: Arc<BytecodePreprocessing>,
    ) -> JoltVerifierPreprocessing<F, PCS> {
        Self {
            generators,
            shared,
            bytecode: VerifierBytecode::Full(bytecode),
        }
    }

    /// Create verifier preprocessing in Committed mode with trusted commitments.
    ///
    /// This is the "fast path" for online verification. The `TrustedBytecodeCommitments`
    /// type guarantees (at the type level) that these commitments were derived from
    /// actual bytecode via `TrustedBytecodeCommitments::derive()`.
    ///
    /// # Trust Model
    /// The caller must ensure the commitments were honestly derived (e.g., loaded from
    /// a trusted file or received from trusted preprocessing).
    #[tracing::instrument(skip_all, name = "JoltVerifierPreprocessing::new_committed")]
    pub fn new_committed(
        shared: JoltSharedPreprocessing,
        generators: PCS::VerifierSetup,
        bytecode_commitments: TrustedBytecodeCommitments<PCS>,
    ) -> JoltVerifierPreprocessing<F, PCS> {
        Self {
            generators,
            shared,
            bytecode: VerifierBytecode::Committed(bytecode_commitments),
        }
    }
}

#[cfg(feature = "prover")]
impl<F: JoltField, PCS: CommitmentScheme<Field = F>> From<&JoltProverPreprocessing<F, PCS>>
    for JoltVerifierPreprocessing<F, PCS>
{
    fn from(prover_preprocessing: &JoltProverPreprocessing<F, PCS>) -> Self {
        let generators = PCS::setup_verifier(&prover_preprocessing.generators);
        // Choose VerifierBytecode variant based on whether prover has bytecode commitments
        let bytecode = match &prover_preprocessing.bytecode_commitments {
            Some(commitments) => VerifierBytecode::Committed(commitments.clone()),
            None => VerifierBytecode::Full(Arc::clone(&prover_preprocessing.bytecode)),
        };
        Self {
            generators,
            shared: prover_preprocessing.shared.clone(),
            bytecode,
        }
    }
}
