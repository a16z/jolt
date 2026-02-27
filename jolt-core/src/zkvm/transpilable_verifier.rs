//! Transpilable Verifier: Generic version of JoltVerifier for symbolic execution.
//!
//! # Temporary Module
//!
//! **This module is intended to be temporary.** The long-term goal is to make
//! `verifier.rs` itself generic over the accumulator type, eliminating the need
//! for this separate file. This would ensure transpilation automatically stays
//! in sync with verifier changes.
//!
//! The current separation exists to:
//! 1. Minimize changes to core Jolt code
//! 2. Allow rapid iteration on transpilation without coordination overhead
//!
//!
//! # Overview
//!
//! This module provides a verifier that is generic over the OpeningAccumulator,
//! allowing it to be used with both:
//! - `VerifierOpeningAccumulator<F>` for real verification
//! - `AstOpeningAccumulator` for symbolic transpilation to circuit code
//!
//! The verification logic is identical to `verifier.rs`, ensuring that the
//! transpiled circuit matches the real verifier exactly.
//!
//! ## Stages Implemented
//!
//! This verifier implements stages 1-7 (all sumcheck stages):
//! - Stages 1-6: Standard sumcheck verifications
//! - Stage 7: HammingWeight claim reduction sumcheck
//!
//! Stage 8 (PCS verification) is NOT transpiled by this module. It requires
//! native elliptic curve operations that are handled separately by the target
//! proving system (e.g., native Gnark Hyrax for BN254/Grumpkin).
//!
//! ## Advice Verifiers
//!
//! Note: `AdviceClaimReduction` verifiers are NOT included in stage 7. They require
//! state management across stages 6-7 (phase transitions).

use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::subprotocols::sumcheck::BatchedSumcheck;
use crate::zkvm::claim_reductions::{
    HammingWeightClaimReductionVerifier, RegistersClaimReductionSumcheckVerifier,
};
use crate::zkvm::config::OneHotParams;
use crate::zkvm::ram::val_final::ValFinalSumcheckVerifier;
use crate::zkvm::{
    bytecode::read_raf_checking::BytecodeReadRafSumcheckVerifier,
    claim_reductions::{
        IncClaimReductionSumcheckVerifier, InstructionLookupsClaimReductionSumcheckVerifier,
        RamRaClaimReductionSumcheckVerifier,
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
    verifier::JoltVerifierPreprocessing,
    ProverDebugInfo,
};
use crate::{
    field::JoltField,
    poly::opening_proof::{OpeningAccumulator, OpeningPoint, VerifierOpeningAccumulator},
    pprof_scope,
    subprotocols::{
        booleanity::{BooleanitySumcheckParams, BooleanitySumcheckVerifier},
        sumcheck_verifier::SumcheckInstanceVerifier,
    },
    transcripts::Transcript,
    utils::{errors::ProofVerifyError, math::Math},
};
use anyhow::Context;
use tracer::JoltDevice;

/// Generic verifier that can be used for both real verification and symbolic transpilation.
///
/// The type parameter `A` is the OpeningAccumulator:
/// - For real verification: `A = VerifierOpeningAccumulator<F>`
/// - For transpilation: `A = AstOpeningAccumulator` (symbolic accumulator)
pub struct TranspilableVerifier<
    'a,
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    ProofTranscript: Transcript,
    A: OpeningAccumulator<F> = VerifierOpeningAccumulator<F>,
> {
    pub trusted_advice_commitment: Option<PCS::Commitment>,
    pub program_io: JoltDevice,
    pub proof: JoltProof<F, PCS, ProofTranscript>,
    pub preprocessing: &'a JoltVerifierPreprocessing<F, PCS>,
    pub transcript: ProofTranscript,
    pub opening_accumulator: A,
    pub spartan_key: UniformSpartanKey<F>,
    pub one_hot_params: OneHotParams,
}

impl<
        'a,
        F: JoltField,
        PCS: CommitmentScheme<Field = F>,
        ProofTranscript: Transcript,
        A: OpeningAccumulator<F>,
    > TranspilableVerifier<'a, F, PCS, ProofTranscript, A>
{
    /// Create a TranspilableVerifier for real verification.
    ///
    /// This constructor creates a new `VerifierOpeningAccumulator` and populates
    /// it with claims from the proof. Only available when `A = VerifierOpeningAccumulator<F>`.
    pub fn new(
        preprocessing: &'a JoltVerifierPreprocessing<F, PCS>,
        proof: JoltProof<F, PCS, ProofTranscript>,
        mut program_io: JoltDevice,
        trusted_advice_commitment: Option<PCS::Commitment>,
        _debug_info: Option<ProverDebugInfo<F, ProofTranscript, PCS>>,
    ) -> Result<
        TranspilableVerifier<'a, F, PCS, ProofTranscript, VerifierOpeningAccumulator<F>>,
        ProofVerifyError,
    > {
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

        Ok(TranspilableVerifier {
            trusted_advice_commitment,
            program_io,
            proof,
            preprocessing,
            transcript,
            opening_accumulator,
            spartan_key,
            one_hot_params,
        })
    }

    /// Create a TranspilableVerifier with a pre-configured opening accumulator.
    ///
    /// This constructor is used for symbolic transpilation where the accumulator
    /// is already populated with MleAst claims (or similar symbolic values).
    pub fn new_with_accumulator(
        preprocessing: &'a JoltVerifierPreprocessing<F, PCS>,
        proof: JoltProof<F, PCS, ProofTranscript>,
        program_io: JoltDevice,
        trusted_advice_commitment: Option<PCS::Commitment>,
        transcript: ProofTranscript,
        opening_accumulator: A,
    ) -> Self {
        let spartan_key = UniformSpartanKey::new(proof.trace_length.next_power_of_two());
        let one_hot_params =
            OneHotParams::from_config(&proof.one_hot_config, proof.bytecode_K, proof.ram_K);

        Self {
            trusted_advice_commitment,
            program_io,
            proof,
            preprocessing,
            transcript,
            opening_accumulator,
            spartan_key,
            one_hot_params,
        }
    }

    /// Verify the Jolt proof (stages 1-7).
    ///
    /// Note: Stage 8 (PCS verification) is not included because it uses
    /// VerifierOpeningAccumulator-specific methods. For Gnark transpilation,
    /// this is replaced by native Gnark pairing checks.
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
            self.transcript
                .append_serializable(b"commitment", commitment);
        }
        // Append untrusted advice commitment to transcript
        if let Some(ref untrusted_advice_commitment) = self.proof.untrusted_advice_commitment {
            self.transcript
                .append_serializable(b"untrusted_advice", untrusted_advice_commitment);
        }
        // Append trusted advice commitment to transcript
        if let Some(ref trusted_advice_commitment) = self.trusted_advice_commitment {
            self.transcript
                .append_serializable(b"trusted_advice", trusted_advice_commitment);
        }

        self.verify_stage1()?;
        self.verify_stage2()?;
        self.verify_stage3()?;
        self.verify_stage4()?;
        self.verify_stage5()?;
        self.verify_stage6()?;
        self.verify_stage7()?;
        // Stage 8 (PCS) is not being transpiled in this version.

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
            vec![&spartan_outer_remaining as &dyn SumcheckInstanceVerifier<F, ProofTranscript, A>],
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

        let ram_read_write_checking = RamReadWriteCheckingVerifier::new(
            &self.opening_accumulator,
            &mut self.transcript,
            &self.one_hot_params,
            self.proof.trace_length,
            &self.proof.rw_config,
        );

        let spartan_product_virtual_remainder = ProductVirtualRemainderVerifier::new(
            self.proof.trace_length,
            uni_skip_params,
            &self.opening_accumulator,
        );

        let instruction_claim_reduction = InstructionLookupsClaimReductionSumcheckVerifier::new(
            self.proof.trace_length,
            &self.opening_accumulator,
            &mut self.transcript,
        );

        let ram_raf_evaluation = RamRafEvaluationSumcheckVerifier::new(
            &self.program_io.memory_layout,
            &self.one_hot_params,
            &self.opening_accumulator,
        );

        let ram_output_check =
            OutputSumcheckVerifier::new(self.proof.ram_K, &self.program_io, &mut self.transcript);

        let _r_stage2 = BatchedSumcheck::verify(
            &self.proof.stage2_sumcheck_proof,
            vec![
                &ram_read_write_checking as &dyn SumcheckInstanceVerifier<F, ProofTranscript, A>,
                &spartan_product_virtual_remainder,
                &instruction_claim_reduction,
                &ram_raf_evaluation,
                &ram_output_check,
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
                &spartan_shift as &dyn SumcheckInstanceVerifier<F, ProofTranscript, A>,
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
        verifier_accumulate_advice::<F, A>(
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
                &registers_read_write_checking
                    as &dyn SumcheckInstanceVerifier<F, ProofTranscript, A>,
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

        let lookups_read_raf = InstructionReadRafSumcheckVerifier::new(
            n_cycle_vars,
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
        );
        let ram_ra_reduction = RamRaClaimReductionSumcheckVerifier::new(
            self.proof.trace_length,
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
        );
        let registers_val_evaluation =
            RegistersValEvaluationSumcheckVerifier::new(&self.opening_accumulator);

        let _r_stage5 = BatchedSumcheck::verify(
            &self.proof.stage5_sumcheck_proof,
            vec![
                &lookups_read_raf as &dyn SumcheckInstanceVerifier<F, ProofTranscript, A>,
                &ram_ra_reduction,
                &registers_val_evaluation,
            ],
            &mut self.opening_accumulator,
            &mut self.transcript,
        )
        .context("Stage 5")?;

        Ok(())
    }

    fn verify_stage6(&mut self) -> Result<(), anyhow::Error> {
        let n_cycle_vars = self.proof.trace_length.log_2();
        let bytecode_read_raf = BytecodeReadRafSumcheckVerifier::gen(
            &self.preprocessing.shared.bytecode,
            n_cycle_vars,
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
        );

        let ram_hamming_booleanity =
            HammingBooleanitySumcheckVerifier::new(&self.opening_accumulator);
        let booleanity_params = BooleanitySumcheckParams::new(
            n_cycle_vars,
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
        );

        let booleanity = BooleanitySumcheckVerifier::new(booleanity_params);
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

        // Note: AdviceClaimReduction verifiers are not included here because they
        // require VerifierOpeningAccumulator-specific state management across stages.
        // For transpilation, advice handling would need a different approach.

        let instances: Vec<&dyn SumcheckInstanceVerifier<F, ProofTranscript, A>> = vec![
            &bytecode_read_raf,
            &booleanity,
            &ram_hamming_booleanity,
            &ram_ra_virtual,
            &lookups_ra_virtual,
            &inc_reduction,
        ];

        let _r_stage6 = BatchedSumcheck::verify(
            &self.proof.stage6_sumcheck_proof,
            instances,
            &mut self.opening_accumulator,
            &mut self.transcript,
        )
        .context("Stage 6")?;

        Ok(())
    }

    /// Stage 7: HammingWeight claim reduction verification.
    fn verify_stage7(&mut self) -> Result<(), anyhow::Error> {
        // Create verifier for HammingWeight claim reduction.
        // This sumcheck fuses HammingWeight + Address Reduction into a single degree-2 sumcheck.
        let hw_verifier = HammingWeightClaimReductionVerifier::new(
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
        );

        // Note: AdviceClaimReduction verifiers are not included here.
        // They require state management across stages 6-7 (phase transitions).
        // For proofs with advice, this would need to be extended.
        // See .claude/tasks/006-add-advice-verifiers-stage7.md for planned implementation.
        let instances: Vec<&dyn SumcheckInstanceVerifier<F, ProofTranscript, A>> =
            vec![&hw_verifier];

        let _r_stage7 = BatchedSumcheck::verify(
            &self.proof.stage7_sumcheck_proof,
            instances,
            &mut self.opening_accumulator,
            &mut self.transcript,
        )
        .context("Stage 7")?;

        Ok(())
    }
}
