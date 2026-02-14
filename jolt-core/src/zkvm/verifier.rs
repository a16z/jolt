use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
use std::sync::Arc;

use crate::curve::JoltCurve;
use crate::poly::commitment::commitment_scheme::{CommitmentScheme, ZkEvalCommitment};
use crate::poly::commitment::dory::{bind_opening_inputs_zk, DoryContext, DoryGlobals};
use crate::poly::commitment::pedersen::PedersenGenerators;
use crate::poly::lagrange_poly::LagrangeHelper;
use crate::subprotocols::blindfold::{
    pedersen_generator_count_for_r1cs, BakedPublicInputs, BlindFoldVerifier,
    BlindFoldVerifierInput, FinalOutputConfig, InputClaimConstraint, OutputClaimConstraint,
    StageConfig, ValueSource, VerifierR1CSBuilder,
};
use crate::subprotocols::sumcheck::{BatchedSumcheck, SumcheckInstanceProof};
use crate::subprotocols::sumcheck_verifier::SumcheckInstanceParams;
use crate::subprotocols::univariate_skip::UniSkipFirstRoundProofVariant;
use crate::zkvm::bytecode::BytecodePreprocessing;
use crate::zkvm::claim_reductions::advice::ReductionPhase;
use crate::zkvm::claim_reductions::RegistersClaimReductionSumcheckVerifier;
use crate::zkvm::config::OneHotParams;
#[cfg(feature = "prover")]
use crate::zkvm::prover::JoltProverPreprocessing;
use crate::zkvm::ram::val_final::ValFinalSumcheckVerifier;
use crate::zkvm::ram::RAMPreprocessing;
use crate::zkvm::witness::all_committed_polynomials;
use crate::zkvm::Serializable;
use crate::zkvm::{
    bytecode::read_raf_checking::BytecodeReadRafSumcheckVerifier,
    claim_reductions::{
        AdviceClaimReductionVerifier, AdviceKind, HammingWeightClaimReductionVerifier,
        IncClaimReductionSumcheckVerifier, InstructionLookupsClaimReductionSumcheckVerifier,
        RamRaClaimReductionSumcheckVerifier,
    },
    fiat_shamir_preamble,
    instruction_lookups::{
        ra_virtual::RaSumcheckVerifier as LookupsRaSumcheckVerifier,
        read_raf_checking::InstructionReadRafSumcheckVerifier,
    },
    proof_serialization::JoltProof,
    r1cs::{
        constraints::{
            OUTER_FIRST_ROUND_POLY_NUM_COEFFS, OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE,
            PRODUCT_VIRTUAL_FIRST_ROUND_POLY_NUM_COEFFS,
            PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE,
        },
        key::UniformSpartanKey,
    },
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
    stage8_opening_ids, ProverDebugInfo,
};
use crate::{
    field::JoltField,
    poly::opening_proof::{
        compute_advice_lagrange_factor, DoryOpeningState, OpeningAccumulator, OpeningId,
        SumcheckId, VerifierOpeningAccumulator,
    },
    pprof_scope,
    subprotocols::{
        booleanity::{BooleanitySumcheckParams, BooleanitySumcheckVerifier},
        sumcheck_verifier::SumcheckInstanceVerifier,
    },
    transcripts::Transcript,
    utils::{errors::ProofVerifyError, math::Math},
    zkvm::witness::CommittedPolynomial,
};

use anyhow::Context;

/// Result of verifying a sumcheck stage.
struct StageVerifyResult<F: JoltField> {
    /// Sumcheck challenges from this stage
    challenges: Vec<F::Challenge>,
    /// Batched output constraint (if any instances have constraints)
    batched_output_constraint: Option<OutputClaimConstraint>,
    /// Challenge values for the batched output constraint (instance-specific challenges)
    output_constraint_challenge_values: Vec<F>,
    /// Batched input constraint (all instances have constraints now)
    batched_input_constraint: InputClaimConstraint,
    /// Challenge values for the batched input constraint
    input_constraint_challenge_values: Vec<F>,
    /// Uni-skip input constraint (only for stages 0-1)
    uniskip_input_constraint: Option<InputClaimConstraint>,
    /// Uni-skip input constraint challenge values
    uniskip_input_constraint_challenge_values: Vec<F>,
}

impl<F: JoltField> StageVerifyResult<F> {
    fn new(
        challenges: Vec<F::Challenge>,
        batched_output_constraint: Option<OutputClaimConstraint>,
        output_constraint_challenge_values: Vec<F>,
        batched_input_constraint: InputClaimConstraint,
        input_constraint_challenge_values: Vec<F>,
    ) -> Self {
        Self {
            challenges,
            batched_output_constraint,
            output_constraint_challenge_values,
            batched_input_constraint,
            input_constraint_challenge_values,
            uniskip_input_constraint: None,
            uniskip_input_constraint_challenge_values: Vec::new(),
        }
    }

    fn with_uniskip(
        challenges: Vec<F::Challenge>,
        batched_output_constraint: Option<OutputClaimConstraint>,
        output_constraint_challenge_values: Vec<F>,
        batched_input_constraint: InputClaimConstraint,
        input_constraint_challenge_values: Vec<F>,
        uniskip_input_constraint: InputClaimConstraint,
        uniskip_input_constraint_challenge_values: Vec<F>,
    ) -> Self {
        Self {
            challenges,
            batched_output_constraint,
            output_constraint_challenge_values,
            batched_input_constraint,
            input_constraint_challenge_values,
            uniskip_input_constraint: Some(uniskip_input_constraint),
            uniskip_input_constraint_challenge_values,
        }
    }
}

/// Collect and batch output constraints from sumcheck verifier instances.
fn batch_output_constraints<F: JoltField, T: Transcript>(
    instances: &[&dyn SumcheckInstanceVerifier<F, T>],
) -> Option<OutputClaimConstraint> {
    let constraints: Vec<Option<OutputClaimConstraint>> = instances
        .iter()
        .map(|instance| instance.get_params().output_claim_constraint())
        .collect();
    OutputClaimConstraint::batch(&constraints, instances.len())
}

/// Collect and batch input constraints from sumcheck verifier instances.
fn batch_input_constraints<F: JoltField, T: Transcript>(
    instances: &[&dyn SumcheckInstanceVerifier<F, T>],
) -> InputClaimConstraint {
    let constraints: Vec<InputClaimConstraint> = instances
        .iter()
        .map(|instance| instance.get_params().input_claim_constraint())
        .collect();
    InputClaimConstraint::batch_required(&constraints, instances.len())
}

/// Scale batching coefficients by 2^(max_rounds - instance_rounds) to account for
/// different-round sumchecks in a batch. This aligns with how BatchedSumcheck::prove_zk
/// scales individual claims before batching.
fn scale_batching_coefficients<F: JoltField, T: Transcript>(
    batching_coefficients: &[F],
    instances: &[&dyn SumcheckInstanceVerifier<F, T>],
) -> Vec<F> {
    let max_num_rounds = instances.iter().map(|i| i.num_rounds()).max().unwrap_or(0);
    batching_coefficients
        .iter()
        .zip(instances.iter())
        .map(|(coeff, instance)| {
            let scale = max_num_rounds - instance.num_rounds();
            coeff.mul_pow_2(scale)
        })
        .collect()
}
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use common::jolt_device::MemoryLayout;
use itertools::Itertools;
use tracer::instruction::Instruction;
use tracer::JoltDevice;

pub struct JoltVerifier<
    'a,
    F: JoltField,
    C: JoltCurve,
    PCS: CommitmentScheme<Field = F>,
    ProofTranscript: Transcript,
> {
    pub trusted_advice_commitment: Option<PCS::Commitment>,
    pub program_io: JoltDevice,
    pub proof: JoltProof<F, C, PCS, ProofTranscript>,
    pub preprocessing: &'a JoltVerifierPreprocessing<F, PCS>,
    pub transcript: ProofTranscript,
    pub opening_accumulator: VerifierOpeningAccumulator<F>,
    /// The advice claim reduction sumcheck effectively spans two stages (6 and 7).
    /// Cache the verifier state here between stages.
    advice_reduction_verifier_trusted: Option<AdviceClaimReductionVerifier<F>>,
    /// The advice claim reduction sumcheck effectively spans two stages (6 and 7).
    /// Cache the verifier state here between stages.
    advice_reduction_verifier_untrusted: Option<AdviceClaimReductionVerifier<F>>,
    pub spartan_key: UniformSpartanKey<F>,
    pub one_hot_params: OneHotParams,
    pub pedersen_generators: PedersenGenerators<C>,
}

#[derive(Clone, Debug)]
struct Stage8VerifyData<F: JoltField> {
    opening_ids: Vec<OpeningId>,
    constraint_coeffs: Vec<F>,
}

impl<
        'a,
        F: JoltField,
        C: JoltCurve,
        PCS: CommitmentScheme<Field = F> + ZkEvalCommitment<C>,
        ProofTranscript: Transcript,
    > JoltVerifier<'a, F, C, PCS, ProofTranscript>
{
    pub fn new(
        preprocessing: &'a JoltVerifierPreprocessing<F, PCS>,
        proof: JoltProof<F, C, PCS, ProofTranscript>,
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

        #[cfg(test)]
        let mut opening_accumulator = VerifierOpeningAccumulator::new(proof.trace_length.log_2());
        #[cfg(not(test))]
        let opening_accumulator = VerifierOpeningAccumulator::new(proof.trace_length.log_2());

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

        // Use deterministic Pedersen generators for BlindFold verification
        // This ensures prover and verifier use the same generators
        let pedersen_generators = PedersenGenerators::<C>::deterministic(4096);

        Ok(Self {
            trusted_advice_commitment,
            program_io,
            proof,
            preprocessing,
            transcript,
            opening_accumulator,
            advice_reduction_verifier_trusted: None,
            advice_reduction_verifier_untrusted: None,
            spartan_key,
            one_hot_params,
            pedersen_generators,
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

        let (stage1_result, uniskip_challenge1) = self.verify_stage1()?;
        let (stage2_result, uniskip_challenge2) = self.verify_stage2()?;
        let stage3_result = self.verify_stage3()?;
        let stage4_result = self.verify_stage4()?;
        let stage5_result = self.verify_stage5()?;
        let stage6_result = self.verify_stage6()?;
        let stage7_result = self.verify_stage7()?;
        let sumcheck_challenges = [
            stage1_result.challenges.clone(),
            stage2_result.challenges.clone(),
            stage3_result.challenges.clone(),
            stage4_result.challenges.clone(),
            stage5_result.challenges.clone(),
            stage6_result.challenges.clone(),
            stage7_result.challenges.clone(),
        ];
        let uniskip_challenges = [uniskip_challenge1, uniskip_challenge2];

        // Collect batched constraints for each stage (indexed 0-6)
        let stage_output_constraints = [
            stage1_result.batched_output_constraint, // Stage 0 (outer remaining)
            stage2_result.batched_output_constraint, // Stage 1 (product virtual + ram)
            stage3_result.batched_output_constraint, // Stage 2 (shift + instruction input + registers claim reduction)
            stage4_result.batched_output_constraint, // Stage 3 (registers rw + ram val eval + ram val final)
            stage5_result.batched_output_constraint, // Stage 4 (registers val eval + ram ra reduction + lookups read raf)
            stage6_result.batched_output_constraint, // Stage 5 (bytecode + booleanity + etc)
            stage7_result.batched_output_constraint, // Stage 6 (hamming weight + advice phase 2)
        ];

        // For stages 0-1: use uni-skip input constraints (not regular round constraints)
        // For stages 2-6: use batched input constraints from regular rounds
        let stage_input_constraints = [
            stage1_result.uniskip_input_constraint.clone().unwrap(), // Stage 0: uni-skip input
            stage2_result.uniskip_input_constraint.clone().unwrap(), // Stage 1: uni-skip input
            stage3_result.batched_input_constraint.clone(),          // Stage 2
            stage4_result.batched_input_constraint.clone(),          // Stage 3
            stage5_result.batched_input_constraint.clone(),          // Stage 4
            stage6_result.batched_input_constraint.clone(),          // Stage 5
            stage7_result.batched_input_constraint.clone(),          // Stage 6
        ];

        let stage_input_constraint_values = [
            stage1_result
                .uniskip_input_constraint_challenge_values
                .clone(), // Stage 0: uni-skip
            stage2_result
                .uniskip_input_constraint_challenge_values
                .clone(), // Stage 1: uni-skip
            stage3_result.input_constraint_challenge_values.clone(), // Stage 2
            stage4_result.input_constraint_challenge_values.clone(), // Stage 3
            stage5_result.input_constraint_challenge_values.clone(), // Stage 4
            stage6_result.input_constraint_challenge_values.clone(), // Stage 5
            stage7_result.input_constraint_challenge_values.clone(), // Stage 6
        ];

        // Collect output constraint challenge values per stage for explicit BlindFold verification
        let output_constraint_challenge_values: [Vec<F>; 7] = [
            stage1_result.output_constraint_challenge_values.clone(), // Stage 0
            stage2_result.output_constraint_challenge_values.clone(), // Stage 1
            stage3_result.output_constraint_challenge_values.clone(), // Stage 2
            stage4_result.output_constraint_challenge_values.clone(), // Stage 3
            stage5_result.output_constraint_challenge_values.clone(), // Stage 4
            stage6_result.output_constraint_challenge_values.clone(), // Stage 5
            stage7_result.output_constraint_challenge_values.clone(), // Stage 6
        ];

        let stage8_data = self.verify_stage8()?;
        self.verify_blindfold(
            &sumcheck_challenges,
            uniskip_challenges,
            &stage_output_constraints,
            &output_constraint_challenge_values,
            &stage_input_constraints,
            &stage_input_constraint_values,
            &stage1_result.batched_input_constraint,
            &stage2_result.batched_input_constraint,
            &stage1_result.input_constraint_challenge_values,
            &stage2_result.input_constraint_challenge_values,
            &stage8_data,
        )?;

        Ok(())
    }

    /// Returns (StageVerifyResult, uni_skip_challenge)
    fn verify_stage1(&mut self) -> Result<(StageVerifyResult<F>, F::Challenge), anyhow::Error> {
        let (uni_skip_params, uni_skip_challenge) = verify_stage1_uni_skip(
            &self.proof.stage1_uni_skip_first_round_proof,
            &self.spartan_key,
            &mut self.opening_accumulator,
            &mut self.transcript,
        )
        .context("Stage 1 univariate skip first round")?;

        let spartan_outer_remaining = OuterRemainingSumcheckVerifier::new(
            self.spartan_key,
            self.proof.trace_length,
            &uni_skip_params,
            &self.opening_accumulator,
        );

        let instances: Vec<&dyn SumcheckInstanceVerifier<F, ProofTranscript>> =
            vec![&spartan_outer_remaining];

        let batching_coefficients: Vec<F> = {
            let mut transcript_clone = self.transcript.clone();
            if !matches!(
                self.proof.stage1_sumcheck_proof,
                SumcheckInstanceProof::Zk(_)
            ) {
                for instance in &instances {
                    let input_claim = instance.input_claim(&self.opening_accumulator);
                    transcript_clone.append_scalar(b"sumcheck_claim", &input_claim);
                }
            }
            transcript_clone.challenge_vector(instances.len())
        };

        let r_stage1 = BatchedSumcheck::verify(
            &self.proof.stage1_sumcheck_proof,
            instances.clone(),
            &mut self.opening_accumulator,
            &mut self.transcript,
        )
        .context("Stage 1")?;

        // Stage 1 (outer remaining) has no output constraint, but does have input constraint
        let batched_output_constraint = batch_output_constraints(&instances);
        let batched_input_constraint = batch_input_constraints(&instances);

        let max_num_rounds = instances.iter().map(|i| i.num_rounds()).max().unwrap();

        // Output constraint challenge values - only include batching coefficients if there's an output constraint
        let output_constraint_challenge_values = if batched_output_constraint.is_some() {
            let mut values = batching_coefficients.clone();
            for instance in &instances {
                let num_rounds = instance.num_rounds();
                let offset = instance.round_offset(max_num_rounds);
                let r_slice = &r_stage1[offset..offset + num_rounds];
                values.extend(
                    instance
                        .get_params()
                        .output_constraint_challenge_values(r_slice),
                );
            }
            values
        } else {
            Vec::new()
        };

        // Input constraint challenge values include scaled batching coefficients
        // (scaled by 2^(max_rounds - instance_rounds) to match prover's claim scaling)
        let mut input_constraint_challenge_values: Vec<F> =
            scale_batching_coefficients(&batching_coefficients, &instances);
        for instance in &instances {
            input_constraint_challenge_values.extend(
                instance
                    .get_params()
                    .input_constraint_challenge_values(&self.opening_accumulator),
            );
        }

        // Get uni-skip's input constraint (for BlindFold - this is what constrains the uni-skip's initial claim)
        let uniskip_input_constraint = uni_skip_params.input_claim_constraint();
        let uniskip_input_constraint_challenge_values =
            uni_skip_params.input_constraint_challenge_values(&self.opening_accumulator);

        let stage_result = StageVerifyResult::with_uniskip(
            r_stage1,
            batched_output_constraint,
            output_constraint_challenge_values,
            batched_input_constraint,
            input_constraint_challenge_values,
            uniskip_input_constraint,
            uniskip_input_constraint_challenge_values,
        );

        Ok((stage_result, uni_skip_challenge))
    }

    /// Returns (StageVerifyResult, uni_skip_challenge)
    fn verify_stage2(&mut self) -> Result<(StageVerifyResult<F>, F::Challenge), anyhow::Error> {
        let (uni_skip_params, uni_skip_challenge) = verify_stage2_uni_skip(
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
            uni_skip_params.clone(),
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

        let instances: Vec<&dyn SumcheckInstanceVerifier<F, ProofTranscript>> = vec![
            &ram_read_write_checking,
            &spartan_product_virtual_remainder,
            &instruction_claim_reduction,
            &ram_raf_evaluation,
            &ram_output_check,
        ];

        let batching_coefficients: Vec<F> = {
            let mut transcript_clone = self.transcript.clone();
            if !matches!(
                self.proof.stage2_sumcheck_proof,
                SumcheckInstanceProof::Zk(_)
            ) {
                for instance in &instances {
                    let input_claim = instance.input_claim(&self.opening_accumulator);
                    transcript_clone.append_scalar(b"sumcheck_claim", &input_claim);
                }
            }
            transcript_clone.challenge_vector(instances.len())
        };

        let r_stage2 = BatchedSumcheck::verify(
            &self.proof.stage2_sumcheck_proof,
            instances.clone(),
            &mut self.opening_accumulator,
            &mut self.transcript,
        )
        .context("Stage 2")?;

        // Collect and batch output constraints from verifier instances
        let batched_output_constraint = batch_output_constraints(&instances);

        // Collect and batch input constraints from verifier instances
        let batched_input_constraint = batch_input_constraints(&instances);

        // Build expected constraint challenge values for explicit verification
        // Pass instance-local challenges (same slice as expected_output_claim receives)
        let max_num_rounds = instances.iter().map(|i| i.num_rounds()).max().unwrap();
        let mut output_constraint_challenge_values: Vec<F> = batching_coefficients.clone();
        // Input constraint uses scaled batching coefficients (2^(max_rounds - instance_rounds))
        let mut input_constraint_challenge_values: Vec<F> =
            scale_batching_coefficients(&batching_coefficients, &instances);
        for instance in &instances {
            let num_rounds = instance.num_rounds();
            let offset = instance.round_offset(max_num_rounds);
            let r_slice = &r_stage2[offset..offset + num_rounds];
            output_constraint_challenge_values.extend(
                instance
                    .get_params()
                    .output_constraint_challenge_values(r_slice),
            );
            input_constraint_challenge_values.extend(
                instance
                    .get_params()
                    .input_constraint_challenge_values(&self.opening_accumulator),
            );
        }

        // Get uni-skip's input constraint (for BlindFold - this is what constrains the uni-skip's initial claim)
        let uniskip_input_constraint = uni_skip_params.input_claim_constraint();
        let uniskip_input_constraint_challenge_values =
            uni_skip_params.input_constraint_challenge_values(&self.opening_accumulator);

        let stage_result = StageVerifyResult::with_uniskip(
            r_stage2,
            batched_output_constraint,
            output_constraint_challenge_values,
            batched_input_constraint,
            input_constraint_challenge_values,
            uniskip_input_constraint,
            uniskip_input_constraint_challenge_values,
        );

        Ok((stage_result, uni_skip_challenge))
    }

    fn verify_stage3(&mut self) -> Result<StageVerifyResult<F>, anyhow::Error> {
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

        let instances: Vec<&dyn SumcheckInstanceVerifier<F, ProofTranscript>> = vec![
            &spartan_shift,
            &spartan_instruction_input,
            &spartan_registers_claim_reduction,
        ];

        let batching_coefficients: Vec<F> = {
            let mut transcript_clone = self.transcript.clone();
            if !matches!(
                self.proof.stage3_sumcheck_proof,
                SumcheckInstanceProof::Zk(_)
            ) {
                for instance in &instances {
                    let input_claim = instance.input_claim(&self.opening_accumulator);
                    transcript_clone.append_scalar(b"sumcheck_claim", &input_claim);
                }
            }
            transcript_clone.challenge_vector(instances.len())
        };

        let r_stage3 = BatchedSumcheck::verify(
            &self.proof.stage3_sumcheck_proof,
            instances.clone(),
            &mut self.opening_accumulator,
            &mut self.transcript,
        )
        .context("Stage 3")?;

        // Collect and batch output constraints from verifier instances
        let batched_output_constraint = batch_output_constraints(&instances);

        // Collect and batch input constraints from verifier instances
        let batched_input_constraint = batch_input_constraints(&instances);

        // Build expected constraint challenge values for explicit verification in verify_blindfold.
        // Pass instance-local challenges (same slice as expected_output_claim receives)
        let max_num_rounds = instances.iter().map(|i| i.num_rounds()).max().unwrap();
        let mut output_constraint_challenge_values: Vec<F> = batching_coefficients.clone();
        // Input constraint uses scaled batching coefficients (2^(max_rounds - instance_rounds))
        let mut input_constraint_challenge_values: Vec<F> =
            scale_batching_coefficients(&batching_coefficients, &instances);
        for instance in &instances {
            let num_rounds = instance.num_rounds();
            let offset = instance.round_offset(max_num_rounds);
            let r_slice = &r_stage3[offset..offset + num_rounds];
            output_constraint_challenge_values.extend(
                instance
                    .get_params()
                    .output_constraint_challenge_values(r_slice),
            );
            input_constraint_challenge_values.extend(
                instance
                    .get_params()
                    .input_constraint_challenge_values(&self.opening_accumulator),
            );
        }

        Ok(StageVerifyResult::new(
            r_stage3,
            batched_output_constraint,
            output_constraint_challenge_values,
            batched_input_constraint,
            input_constraint_challenge_values,
        ))
    }

    fn verify_stage4(&mut self) -> Result<StageVerifyResult<F>, anyhow::Error> {
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

        let instances: Vec<&dyn SumcheckInstanceVerifier<F, ProofTranscript>> = vec![
            &registers_read_write_checking,
            &ram_val_evaluation,
            &ram_val_final,
        ];

        let batching_coefficients: Vec<F> = {
            let mut transcript_clone = self.transcript.clone();
            if !matches!(
                self.proof.stage4_sumcheck_proof,
                SumcheckInstanceProof::Zk(_)
            ) {
                for instance in &instances {
                    let input_claim = instance.input_claim(&self.opening_accumulator);
                    transcript_clone.append_scalar(b"sumcheck_claim", &input_claim);
                }
            }
            transcript_clone.challenge_vector(instances.len())
        };

        let r_stage4 = BatchedSumcheck::verify(
            &self.proof.stage4_sumcheck_proof,
            instances.clone(),
            &mut self.opening_accumulator,
            &mut self.transcript,
        )
        .context("Stage 4")?;

        // Collect and batch output constraints from verifier instances
        let batched_output_constraint = batch_output_constraints(&instances);

        // Collect and batch input constraints from verifier instances
        let batched_input_constraint = batch_input_constraints(&instances);

        // Build expected constraint challenge values for explicit verification in verify_blindfold.
        let max_num_rounds = instances.iter().map(|i| i.num_rounds()).max().unwrap();
        let mut output_constraint_challenge_values: Vec<F> = batching_coefficients.clone();
        // Input constraint uses scaled batching coefficients (2^(max_rounds - instance_rounds))
        let mut input_constraint_challenge_values: Vec<F> =
            scale_batching_coefficients(&batching_coefficients, &instances);
        for instance in &instances {
            let num_rounds = instance.num_rounds();
            let offset = instance.round_offset(max_num_rounds);
            let r_slice = &r_stage4[offset..offset + num_rounds];
            output_constraint_challenge_values.extend(
                instance
                    .get_params()
                    .output_constraint_challenge_values(r_slice),
            );
            input_constraint_challenge_values.extend(
                instance
                    .get_params()
                    .input_constraint_challenge_values(&self.opening_accumulator),
            );
        }

        Ok(StageVerifyResult::new(
            r_stage4,
            batched_output_constraint,
            output_constraint_challenge_values,
            batched_input_constraint,
            input_constraint_challenge_values,
        ))
    }

    fn verify_stage5(&mut self) -> Result<StageVerifyResult<F>, anyhow::Error> {
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

        let instances: Vec<&dyn SumcheckInstanceVerifier<F, ProofTranscript>> = vec![
            &lookups_read_raf,
            &ram_ra_reduction,
            &registers_val_evaluation,
        ];

        let batching_coefficients: Vec<F> = {
            let mut transcript_clone = self.transcript.clone();
            if !matches!(
                self.proof.stage5_sumcheck_proof,
                SumcheckInstanceProof::Zk(_)
            ) {
                for instance in &instances {
                    let input_claim = instance.input_claim(&self.opening_accumulator);
                    transcript_clone.append_scalar(b"sumcheck_claim", &input_claim);
                }
            }
            transcript_clone.challenge_vector(instances.len())
        };

        let r_stage5 = BatchedSumcheck::verify(
            &self.proof.stage5_sumcheck_proof,
            instances.clone(),
            &mut self.opening_accumulator,
            &mut self.transcript,
        )
        .context("Stage 5")?;

        // Collect and batch output constraints from verifier instances
        let batched_output_constraint = batch_output_constraints(&instances);

        // Collect and batch input constraints from verifier instances
        let batched_input_constraint = batch_input_constraints(&instances);

        // Build expected constraint challenge values for explicit verification
        let max_num_rounds = instances.iter().map(|i| i.num_rounds()).max().unwrap();
        let mut output_constraint_challenge_values: Vec<F> = batching_coefficients.clone();
        // Input constraint uses scaled batching coefficients (2^(max_rounds - instance_rounds))
        let mut input_constraint_challenge_values: Vec<F> =
            scale_batching_coefficients(&batching_coefficients, &instances);
        for instance in &instances {
            let num_rounds = instance.num_rounds();
            let offset = instance.round_offset(max_num_rounds);
            let r_slice = &r_stage5[offset..offset + num_rounds];
            output_constraint_challenge_values.extend(
                instance
                    .get_params()
                    .output_constraint_challenge_values(r_slice),
            );
            input_constraint_challenge_values.extend(
                instance
                    .get_params()
                    .input_constraint_challenge_values(&self.opening_accumulator),
            );
        }

        Ok(StageVerifyResult::new(
            r_stage5,
            batched_output_constraint,
            output_constraint_challenge_values,
            batched_input_constraint,
            input_constraint_challenge_values,
        ))
    }

    fn verify_stage6(&mut self) -> Result<StageVerifyResult<F>, anyhow::Error> {
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

        // Advice claim reduction (Phase 1 in Stage 6): trusted and untrusted are separate instances.
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

        let mut instances: Vec<&dyn SumcheckInstanceVerifier<F, ProofTranscript>> = vec![
            &bytecode_read_raf,
            &booleanity,
            &ram_hamming_booleanity,
            &ram_ra_virtual,
            &lookups_ra_virtual,
            &inc_reduction,
        ];
        if let Some(ref advice) = self.advice_reduction_verifier_trusted {
            instances.push(advice);
        }
        if let Some(ref advice) = self.advice_reduction_verifier_untrusted {
            instances.push(advice);
        }

        let batching_coefficients: Vec<F> = {
            let mut transcript_clone = self.transcript.clone();
            if !matches!(
                self.proof.stage6_sumcheck_proof,
                SumcheckInstanceProof::Zk(_)
            ) {
                for instance in &instances {
                    let input_claim = instance.input_claim(&self.opening_accumulator);
                    transcript_clone.append_scalar(b"sumcheck_claim", &input_claim);
                }
            }
            transcript_clone.challenge_vector(instances.len())
        };

        let r_stage6 = BatchedSumcheck::verify(
            &self.proof.stage6_sumcheck_proof,
            instances.clone(),
            &mut self.opening_accumulator,
            &mut self.transcript,
        )
        .context("Stage 6")?;

        // Collect and batch output constraints from verifier instances
        let batched_output_constraint = batch_output_constraints(&instances);

        // Collect and batch input constraints from verifier instances
        let batched_input_constraint = batch_input_constraints(&instances);

        // Build expected constraint challenge values for explicit verification
        let max_num_rounds = instances.iter().map(|i| i.num_rounds()).max().unwrap();
        let mut output_constraint_challenge_values: Vec<F> = batching_coefficients.clone();
        // Input constraint uses scaled batching coefficients (2^(max_rounds - instance_rounds))
        let mut input_constraint_challenge_values: Vec<F> =
            scale_batching_coefficients(&batching_coefficients, &instances);
        for instance in &instances {
            let num_rounds = instance.num_rounds();
            let offset = instance.round_offset(max_num_rounds);
            let r_slice = &r_stage6[offset..offset + num_rounds];
            output_constraint_challenge_values.extend(
                instance
                    .get_params()
                    .output_constraint_challenge_values(r_slice),
            );
            input_constraint_challenge_values.extend(
                instance
                    .get_params()
                    .input_constraint_challenge_values(&self.opening_accumulator),
            );
        }

        Ok(StageVerifyResult::new(
            r_stage6,
            batched_output_constraint,
            output_constraint_challenge_values,
            batched_input_constraint,
            input_constraint_challenge_values,
        ))
    }

    /// Verify BlindFold proof binding to sumcheck challenges.
    ///
    /// Stages 1-2 uni-skip first rounds use Pedersen commitments (ZkUniSkipFirstRoundProof).
    /// The polynomial coefficients are hidden in the transcript - verifier only sees commitments.
    /// BlindFold verifies the uni-skip polynomial constraints (sum check + evaluation) using
    /// power sums for the symmetric domain.
    #[allow(clippy::too_many_arguments)]
    fn verify_blindfold(
        &mut self,
        sumcheck_challenges: &[Vec<F::Challenge>; 7],
        uniskip_challenges: [F::Challenge; 2],
        stage_output_constraints: &[Option<OutputClaimConstraint>; 7],
        output_constraint_challenge_values: &[Vec<F>; 7],
        stage_input_constraints: &[InputClaimConstraint; 7],
        input_constraint_challenge_values: &[Vec<F>; 7],
        // For stages 0-1: batched input constraint for regular rounds (different from uni-skip)
        stage1_batched_input: &InputClaimConstraint,
        stage2_batched_input: &InputClaimConstraint,
        stage1_batched_input_values: &[F],
        stage2_batched_input_values: &[F],
        stage8_data: &Stage8VerifyData<F>,
    ) -> Result<(), anyhow::Error> {
        // Build stage configurations including uni-skip rounds.
        // Uni-skip rounds are the first round of stages 1 and 2 (indices 0 and 1).
        let stage_proofs = [
            &self.proof.stage1_sumcheck_proof,
            &self.proof.stage2_sumcheck_proof,
            &self.proof.stage3_sumcheck_proof,
            &self.proof.stage4_sumcheck_proof,
            &self.proof.stage5_sumcheck_proof,
            &self.proof.stage6_sumcheck_proof,
            &self.proof.stage7_sumcheck_proof,
        ];

        // Precompute power sums for uni-skip domains
        let outer_power_sums = LagrangeHelper::power_sums::<
            OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE,
            OUTER_FIRST_ROUND_POLY_NUM_COEFFS,
        >();
        let product_power_sums = LagrangeHelper::power_sums::<
            PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE,
            PRODUCT_VIRTUAL_FIRST_ROUND_POLY_NUM_COEFFS,
        >();

        let mut stage_configs = Vec::new();
        // Track which stage_config index corresponds to uni-skip and regular first rounds
        let mut uniskip_indices: Vec<usize> = Vec::new(); // Only 2 elements for stages 0-1
        let mut regular_first_round_indices: Vec<usize> = Vec::new(); // 7 elements for all stages
        let mut last_round_indices: Vec<usize> = Vec::new();

        for (stage_idx, proof) in stage_proofs.iter().enumerate() {
            // For stages 0 and 1 (Jolt stages 1 and 2), add uni-skip config first
            if stage_idx < 2 {
                let uniskip_proof = if stage_idx == 0 {
                    &self.proof.stage1_uni_skip_first_round_proof
                } else {
                    &self.proof.stage2_uni_skip_first_round_proof
                };
                let poly_degree = uniskip_proof.poly_degree();

                let power_sums: Vec<i128> = if stage_idx == 0 {
                    outer_power_sums.to_vec()
                } else {
                    product_power_sums.to_vec()
                };

                // Record uni-skip index for its input constraint
                uniskip_indices.push(stage_configs.len());

                let config = if stage_idx == 0 {
                    StageConfig::new_uniskip(poly_degree, power_sums)
                } else {
                    StageConfig::new_uniskip_chain(poly_degree, power_sums)
                };
                stage_configs.push(config);
            }

            // Record first regular round index for its input constraint
            regular_first_round_indices.push(stage_configs.len());

            // Add regular sumcheck rounds
            let num_rounds = proof.num_rounds();
            for round_idx in 0..num_rounds {
                let poly_degree = match proof {
                    crate::subprotocols::sumcheck::SumcheckInstanceProof::Standard(std_proof) => {
                        std_proof.compressed_polys[round_idx]
                            .coeffs_except_linear_term
                            .len()
                    }
                    crate::subprotocols::sumcheck::SumcheckInstanceProof::Zk(zk_proof) => {
                        zk_proof.poly_degrees[round_idx]
                    }
                };
                // First regular round ALWAYS starts a new chain
                // (batched claims differ from uni-skip output due to batching coefficients)
                let starts_new_chain = round_idx == 0;
                let config = if starts_new_chain {
                    StageConfig::new_chain(1, poly_degree)
                } else {
                    StageConfig::new(1, poly_degree)
                };
                stage_configs.push(config);
            }

            // Record the last round index for output constraint
            last_round_indices.push(stage_configs.len() - 1);
        }

        // Add final_output configurations using the batched constraints from verifier instances
        for (stage_idx, constraint) in stage_output_constraints.iter().enumerate() {
            if let Some(batched) = constraint {
                let last_round_idx = last_round_indices[stage_idx];
                stage_configs[last_round_idx].final_output =
                    Some(FinalOutputConfig::with_constraint(batched.clone()));
            }
        }

        // Add initial_input configurations for uni-skip stages (stages 0-1)
        // These use the uni-skip's own input constraints
        let uniskip_constraints = [
            stage_input_constraints[0].clone(), // Stage 0 uni-skip
            stage_input_constraints[1].clone(), // Stage 1 uni-skip
        ];
        for (i, constraint) in uniskip_constraints.iter().enumerate() {
            let idx = uniskip_indices[i];
            stage_configs[idx].initial_input =
                Some(FinalOutputConfig::with_constraint(constraint.clone()));
        }

        // Add initial_input configurations for regular first rounds (all 7 stages)
        // These use the batched input constraints from the stage results
        let regular_constraints = [
            stage1_batched_input.clone(),       // Stage 0 regular
            stage2_batched_input.clone(),       // Stage 1 regular
            stage_input_constraints[2].clone(), // Stage 2
            stage_input_constraints[3].clone(), // Stage 3
            stage_input_constraints[4].clone(), // Stage 4
            stage_input_constraints[5].clone(), // Stage 5
            stage_input_constraints[6].clone(), // Stage 6
        ];
        for (i, constraint) in regular_constraints.iter().enumerate() {
            let idx = regular_first_round_indices[i];
            stage_configs[idx].initial_input =
                Some(FinalOutputConfig::with_constraint(constraint.clone()));
        }

        let extra_constraint_terms: Vec<(ValueSource, ValueSource)> = stage8_data
            .opening_ids
            .iter()
            .enumerate()
            .map(|(i, id)| (ValueSource::challenge(i), ValueSource::opening(*id)))
            .collect();
        let extra_constraint = OutputClaimConstraint::linear(extra_constraint_terms);
        let extra_constraints = vec![extra_constraint];

        // Build baked public inputs from expected values
        let mut baked_challenges: Vec<F> = Vec::new();
        for (stage_idx, stage_challenges) in sumcheck_challenges.iter().enumerate() {
            if stage_idx < 2 {
                baked_challenges.push(uniskip_challenges[stage_idx].into());
            }
            for challenge in stage_challenges.iter() {
                baked_challenges.push((*challenge).into());
            }
        }

        let all_input_challenge_values: [&[F]; 9] = [
            &input_constraint_challenge_values[0],
            stage1_batched_input_values,
            &input_constraint_challenge_values[1],
            stage2_batched_input_values,
            &input_constraint_challenge_values[2],
            &input_constraint_challenge_values[3],
            &input_constraint_challenge_values[4],
            &input_constraint_challenge_values[5],
            &input_constraint_challenge_values[6],
        ];
        let mut baked_input_challenges: Vec<F> = Vec::new();
        for expected_values in all_input_challenge_values.iter() {
            baked_input_challenges.extend_from_slice(expected_values);
        }

        let mut baked_output_challenges: Vec<F> = Vec::new();
        for expected_values in output_constraint_challenge_values.iter() {
            baked_output_challenges.extend_from_slice(expected_values);
        }

        let baked = BakedPublicInputs {
            challenges: baked_challenges,
            initial_claims: self.proof.blindfold_initial_claims.to_vec(),
            batching_coefficients: Vec::new(),
            output_constraint_challenges: baked_output_challenges,
            input_constraint_challenges: baked_input_challenges,
            extra_constraint_challenges: stage8_data.constraint_coeffs.clone(),
        };

        let builder =
            VerifierR1CSBuilder::new_with_extra(&stage_configs, &extra_constraints, &baked);
        let r1cs = builder.build();

        // 6. Build round_commitments from main sumcheck proofs
        let mut round_commitments: Vec<C::G1> = Vec::new();
        for (stage_idx, proof) in stage_proofs.iter().enumerate() {
            // For stages 0-1, include uni-skip commitment first
            if stage_idx < 2 {
                let uniskip_proof = if stage_idx == 0 {
                    &self.proof.stage1_uni_skip_first_round_proof
                } else {
                    &self.proof.stage2_uni_skip_first_round_proof
                };
                if let UniSkipFirstRoundProofVariant::Zk(zk_uniskip) = uniskip_proof {
                    round_commitments.push(zk_uniskip.commitment);
                }
            }
            // Add regular sumcheck round commitments
            if let SumcheckInstanceProof::Zk(zk_proof) = proof {
                round_commitments.extend(zk_proof.round_commitments.iter().cloned());
            }
        }

        // 7. Build eval_commitments from PCS proof
        let eval_commitment = PCS::eval_commitment(&self.proof.joint_opening_proof)
            .ok_or_else(|| anyhow::anyhow!("Missing evaluation commitment in PCS proof"))?;
        let eval_commitments = vec![eval_commitment];

        let verifier_input = BlindFoldVerifierInput {
            round_commitments,
            eval_commitments,
        };

        // Create BlindFold verifier and verify the proof
        let pedersen_generator_count = pedersen_generator_count_for_r1cs(&r1cs);
        let pedersen_generators = PedersenGenerators::<C>::deterministic(pedersen_generator_count);
        let eval_commitment_gens =
            PCS::eval_commitment_gens_verifier(&self.preprocessing.generators);
        let verifier =
            BlindFoldVerifier::<_, _>::new(&pedersen_generators, &r1cs, eval_commitment_gens);
        let mut blindfold_transcript = ProofTranscript::new(b"BlindFold");

        verifier
            .verify(
                &self.proof.blindfold_proof,
                &verifier_input,
                &mut blindfold_transcript,
            )
            .map_err(|e| anyhow::anyhow!("BlindFold verification failed: {e:?}"))?;

        tracing::debug!(
            "BlindFold verification passed: {} R1CS constraints",
            r1cs.num_constraints
        );

        Ok(())
    }

    /// Stage 7: HammingWeight claim reduction verification.
    fn verify_stage7(&mut self) -> Result<StageVerifyResult<F>, anyhow::Error> {
        // Create verifier for HammingWeightClaimReduction
        // (r_cycle and r_addr_bool are extracted from Booleanity opening internally)
        let hw_verifier = HammingWeightClaimReductionVerifier::new(
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
        );

        let mut instances: Vec<&dyn SumcheckInstanceVerifier<F, ProofTranscript>> =
            vec![&hw_verifier];
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

        let batching_coefficients: Vec<F> = {
            let mut transcript_clone = self.transcript.clone();
            if !matches!(
                self.proof.stage7_sumcheck_proof,
                SumcheckInstanceProof::Zk(_)
            ) {
                for instance in &instances {
                    let input_claim = instance.input_claim(&self.opening_accumulator);
                    transcript_clone.append_scalar(b"sumcheck_claim", &input_claim);
                }
            }
            transcript_clone.challenge_vector(instances.len())
        };

        let r_stage7 = BatchedSumcheck::verify(
            &self.proof.stage7_sumcheck_proof,
            instances.clone(),
            &mut self.opening_accumulator,
            &mut self.transcript,
        )
        .context("Stage 7")?;

        // Collect and batch output constraints from verifier instances
        let batched_output_constraint = batch_output_constraints(&instances);

        // Collect and batch input constraints from verifier instances
        let batched_input_constraint = batch_input_constraints(&instances);

        // Build expected constraint challenge values for explicit verification
        let max_num_rounds = instances.iter().map(|i| i.num_rounds()).max().unwrap();
        let mut output_constraint_challenge_values: Vec<F> = batching_coefficients.clone();
        // Input constraint uses scaled batching coefficients (2^(max_rounds - instance_rounds))
        let mut input_constraint_challenge_values: Vec<F> =
            scale_batching_coefficients(&batching_coefficients, &instances);
        for instance in &instances {
            let num_rounds = instance.num_rounds();
            let offset = instance.round_offset(max_num_rounds);
            let r_slice = &r_stage7[offset..offset + num_rounds];
            output_constraint_challenge_values.extend(
                instance
                    .get_params()
                    .output_constraint_challenge_values(r_slice),
            );
            input_constraint_challenge_values.extend(
                instance
                    .get_params()
                    .input_constraint_challenge_values(&self.opening_accumulator),
            );
        }

        Ok(StageVerifyResult::new(
            r_stage7,
            batched_output_constraint,
            output_constraint_challenge_values,
            batched_input_constraint,
            input_constraint_challenge_values,
        ))
    }

    /// Stage 8: Dory batch opening verification.
    fn verify_stage8(&mut self) -> Result<Stage8VerifyData<F>, anyhow::Error> {
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
        let _r_address_stage7 = &opening_point.r[..log_k_chunk];

        // 1. Collect all (polynomial, claim) pairs
        let mut polynomial_claims = Vec::new();
        let mut scaling_factors = Vec::new();

        // Dense polynomials: RamInc and RdInc (from IncClaimReduction in Stage 6)
        let (_, ram_inc_claim) = self.opening_accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::RamInc,
            SumcheckId::IncClaimReduction,
        );
        let (_, rd_inc_claim) = self.opening_accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::RdInc,
            SumcheckId::IncClaimReduction,
        );

        // Dense polynomials are independent of address variables, so no Lagrange scaling.
        polynomial_claims.push((CommittedPolynomial::RamInc, ram_inc_claim));
        scaling_factors.push(F::one());
        polynomial_claims.push((CommittedPolynomial::RdInc, rd_inc_claim));
        scaling_factors.push(F::one());

        // Sparse polynomials: all RA polys (from HammingWeightClaimReduction)
        for i in 0..self.one_hot_params.instruction_d {
            let (_, claim) = self.opening_accumulator.get_committed_polynomial_opening(
                CommittedPolynomial::InstructionRa(i),
                SumcheckId::HammingWeightClaimReduction,
            );
            polynomial_claims.push((CommittedPolynomial::InstructionRa(i), claim));
            scaling_factors.push(F::one());
        }
        for i in 0..self.one_hot_params.bytecode_d {
            let (_, claim) = self.opening_accumulator.get_committed_polynomial_opening(
                CommittedPolynomial::BytecodeRa(i),
                SumcheckId::HammingWeightClaimReduction,
            );
            polynomial_claims.push((CommittedPolynomial::BytecodeRa(i), claim));
            scaling_factors.push(F::one());
        }
        for i in 0..self.one_hot_params.ram_d {
            let (_, claim) = self.opening_accumulator.get_committed_polynomial_opening(
                CommittedPolynomial::RamRa(i),
                SumcheckId::HammingWeightClaimReduction,
            );
            polynomial_claims.push((CommittedPolynomial::RamRa(i), claim));
            scaling_factors.push(F::one());
        }

        // Advice polynomials: TrustedAdvice and UntrustedAdvice (from AdviceClaimReduction in Stage 6)
        // These are committed with smaller dimensions, so we apply Lagrange factors to embed
        // them in the top-left block of the main Dory matrix.
        let mut include_trusted_advice = false;
        let mut include_untrusted_advice = false;

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
            scaling_factors.push(lagrange_factor);
            include_trusted_advice = true;
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
            scaling_factors.push(lagrange_factor);
            include_untrusted_advice = true;
        }

        // 2. Sample gamma and compute powers for RLC
        // Claims NOT absorbed — binding comes from polynomial commitments already in transcript.
        let gamma_powers: Vec<F> = self
            .transcript
            .challenge_scalar_powers(polynomial_claims.len());
        let constraint_coeffs: Vec<F> = gamma_powers
            .iter()
            .zip(&scaling_factors)
            .map(|(gamma, scale)| *gamma * *scale)
            .collect();

        let opening_ids = stage8_opening_ids(
            &self.one_hot_params,
            include_trusted_advice,
            include_untrusted_advice,
        );

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

        let joint_commitment = self.compute_joint_commitment(&mut commitments_map, &state)?;

        PCS::verify(
            &self.proof.joint_opening_proof,
            &self.preprocessing.generators,
            &mut self.transcript,
            &opening_point.r,
            &F::zero(),
            &joint_commitment,
        )
        .context("Stage 8")?;

        let y_com: C::G1 = PCS::eval_commitment(&self.proof.joint_opening_proof)
            .expect("ZK proof must have y_com");
        bind_opening_inputs_zk::<F, C, _>(&mut self.transcript, &opening_point.r, &y_com);

        Ok(Stage8VerifyData {
            opening_ids,
            constraint_coeffs,
        })
    }

    /// Compute joint commitment for the batch opening.
    fn compute_joint_commitment(
        &self,
        commitment_map: &mut HashMap<CommittedPolynomial, PCS::Commitment>,
        state: &DoryOpeningState<F>,
    ) -> Result<PCS::Commitment, ProofVerifyError> {
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
            .map(|(k, v)| {
                commitment_map
                    .remove(&k)
                    .map(|c| (v, c))
                    .ok_or(ProofVerifyError::InternalError)
            })
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .unzip();

        Ok(PCS::combine_commitments(&commitments, &coeffs))
    }
}

#[derive(Debug, Clone)]
pub struct JoltSharedPreprocessing {
    pub bytecode: Arc<BytecodePreprocessing>,
    pub ram: RAMPreprocessing,
    pub memory_layout: MemoryLayout,
    pub max_padded_trace_length: usize,
}

impl CanonicalSerialize for JoltSharedPreprocessing {
    fn serialize_with_mode<W: std::io::Write>(
        &self,
        mut writer: W,
        compress: ark_serialize::Compress,
    ) -> Result<(), ark_serialize::SerializationError> {
        // Serialize the inner BytecodePreprocessing (not the Arc wrapper)
        self.bytecode
            .as_ref()
            .serialize_with_mode(&mut writer, compress)?;
        self.ram.serialize_with_mode(&mut writer, compress)?;
        self.memory_layout
            .serialize_with_mode(&mut writer, compress)?;
        self.max_padded_trace_length
            .serialize_with_mode(&mut writer, compress)?;
        Ok(())
    }

    fn serialized_size(&self, compress: ark_serialize::Compress) -> usize {
        self.bytecode.serialized_size(compress)
            + self.ram.serialized_size(compress)
            + self.memory_layout.serialized_size(compress)
            + self.max_padded_trace_length.serialized_size(compress)
    }
}

impl CanonicalDeserialize for JoltSharedPreprocessing {
    fn deserialize_with_mode<R: std::io::Read>(
        mut reader: R,
        compress: ark_serialize::Compress,
        validate: ark_serialize::Validate,
    ) -> Result<Self, ark_serialize::SerializationError> {
        let bytecode =
            BytecodePreprocessing::deserialize_with_mode(&mut reader, compress, validate)?;
        let ram = RAMPreprocessing::deserialize_with_mode(&mut reader, compress, validate)?;
        let memory_layout = MemoryLayout::deserialize_with_mode(&mut reader, compress, validate)?;
        let max_padded_trace_length =
            usize::deserialize_with_mode(&mut reader, compress, validate)?;
        Ok(Self {
            bytecode: Arc::new(bytecode),
            ram,
            memory_layout,
            max_padded_trace_length,
        })
    }
}

impl ark_serialize::Valid for JoltSharedPreprocessing {
    fn check(&self) -> Result<(), ark_serialize::SerializationError> {
        self.bytecode.check()?;
        self.ram.check()?;
        self.memory_layout.check()
    }
}

impl JoltSharedPreprocessing {
    #[tracing::instrument(skip_all, name = "JoltSharedPreprocessing::new")]
    pub fn new(
        bytecode: Vec<Instruction>,
        memory_layout: MemoryLayout,
        memory_init: Vec<(u64, u8)>,
        max_padded_trace_length: usize,
    ) -> JoltSharedPreprocessing {
        let bytecode = Arc::new(BytecodePreprocessing::preprocess(bytecode));
        let ram = RAMPreprocessing::preprocess(memory_init);
        Self {
            bytecode,
            ram,
            memory_layout,
            max_padded_trace_length,
        }
    }
}

#[derive(Debug, Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct JoltVerifierPreprocessing<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    pub generators: PCS::VerifierSetup,
    pub shared: JoltSharedPreprocessing,
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
    #[tracing::instrument(skip_all, name = "JoltVerifierPreprocessing::new")]
    pub fn new(
        shared: JoltSharedPreprocessing,
        generators: PCS::VerifierSetup,
    ) -> JoltVerifierPreprocessing<F, PCS> {
        Self {
            generators,
            shared: shared.clone(),
        }
    }
}

#[cfg(feature = "prover")]
impl<F: JoltField, PCS: CommitmentScheme<Field = F>> From<&JoltProverPreprocessing<F, PCS>>
    for JoltVerifierPreprocessing<F, PCS>
{
    fn from(prover_preprocessing: &JoltProverPreprocessing<F, PCS>) -> Self {
        let generators = PCS::setup_verifier(&prover_preprocessing.generators);
        Self {
            generators,
            shared: prover_preprocessing.shared.clone(),
        }
    }
}
