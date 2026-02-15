use crate::curve::JoltCurve;
use crate::field::JoltField;
use crate::poly::commitment::commitment_scheme::{CommitmentScheme, ZkEvalCommitment};
use crate::poly::opening_proof::VerifierOpeningAccumulator;
use crate::subprotocols::booleanity::{BooleanitySumcheckParams, BooleanitySumcheckVerifier};
use crate::subprotocols::sumcheck::{BatchedSumcheck, SumcheckInstanceProof};
use crate::subprotocols::sumcheck_verifier::SumcheckInstanceVerifier;
use crate::transcripts::Transcript;
use crate::utils::errors::ProofVerifyError;
use crate::utils::math::Math;
use crate::zkvm::bytecode::read_raf_checking::BytecodeReadRafSumcheckVerifier;
use crate::zkvm::claim_reductions::advice::ReductionPhase;
use crate::zkvm::claim_reductions::{
    AdviceClaimReductionVerifier, AdviceKind, HammingWeightClaimReductionVerifier,
    IncClaimReductionSumcheckVerifier, InstructionLookupsClaimReductionSumcheckVerifier,
    RamRaClaimReductionSumcheckVerifier, RegistersClaimReductionSumcheckVerifier,
};
use crate::zkvm::instruction_lookups::ra_virtual::RaSumcheckVerifier as LookupsRaSumcheckVerifier;
use crate::zkvm::instruction_lookups::read_raf_checking::InstructionReadRafSumcheckVerifier;
use crate::zkvm::ram::hamming_booleanity::HammingBooleanitySumcheckVerifier;
use crate::zkvm::ram::output_check::OutputSumcheckVerifier;
use crate::zkvm::ram::ra_virtual::RamRaVirtualSumcheckVerifier;
use crate::zkvm::ram::raf_evaluation::RafEvaluationSumcheckVerifier as RamRafEvaluationSumcheckVerifier;
use crate::zkvm::ram::read_write_checking::RamReadWriteCheckingVerifier;
use crate::zkvm::ram::val_evaluation::ValEvaluationSumcheckVerifier as RamValEvaluationSumcheckVerifier;
use crate::zkvm::ram::val_final::ValFinalSumcheckVerifier;
use crate::zkvm::ram::verifier_accumulate_advice;
use crate::zkvm::registers::read_write_checking::RegistersReadWriteCheckingVerifier;
use crate::zkvm::registers::val_evaluation::ValEvaluationSumcheckVerifier as RegistersValEvaluationSumcheckVerifier;
use crate::zkvm::spartan::instruction_input::InstructionInputSumcheckVerifier;
use crate::zkvm::spartan::outer::OuterRemainingSumcheckVerifier;
use crate::zkvm::spartan::product::ProductVirtualRemainderVerifier;
use crate::zkvm::spartan::shift::ShiftSumcheckVerifier;
use crate::zkvm::spartan::{verify_stage1_uni_skip, verify_stage2_uni_skip};

#[cfg(feature = "zk")]
use super::preprocessing::{
    batch_input_constraints, batch_output_constraints, scale_batching_coefficients,
};
#[cfg(feature = "zk")]
use crate::subprotocols::sumcheck_verifier::SumcheckInstanceParams;

use super::preprocessing::StageVerifyResult;
use super::JoltVerifier;

#[cfg_attr(not(feature = "zk"), allow(unused_variables))]
fn verify_batched_stage<F: JoltField, C: JoltCurve, ProofTranscript: Transcript>(
    transcript: &mut ProofTranscript,
    opening_accumulator: &mut VerifierOpeningAccumulator<F>,
    instances: Vec<&dyn SumcheckInstanceVerifier<F, ProofTranscript>>,
    proof: &SumcheckInstanceProof<F, C, ProofTranscript>,
) -> Result<StageVerifyResult<F>, ProofVerifyError> {
    let batching_coefficients: Vec<F> = {
        let mut transcript_clone = transcript.clone();
        if !matches!(proof, SumcheckInstanceProof::Zk(_)) {
            for instance in &instances {
                let input_claim = instance.input_claim(opening_accumulator);
                transcript_clone.append_scalar(b"sumcheck_claim", &input_claim);
            }
        }
        transcript_clone.challenge_vector(instances.len())
    };

    let challenges =
        BatchedSumcheck::verify(proof, instances.clone(), opening_accumulator, transcript)?;

    #[cfg(feature = "zk")]
    {
        let batched_output_constraint = batch_output_constraints(&instances);
        let batched_input_constraint = batch_input_constraints(&instances);
        let max_num_rounds = instances.iter().map(|i| i.num_rounds()).max().unwrap();
        let mut output_constraint_challenge_values: Vec<F> = batching_coefficients.clone();
        let mut input_constraint_challenge_values: Vec<F> =
            scale_batching_coefficients(&batching_coefficients, &instances);
        for instance in &instances {
            let num_rounds = instance.num_rounds();
            let offset = instance.round_offset(max_num_rounds);
            let r_slice = &challenges[offset..offset + num_rounds];
            output_constraint_challenge_values.extend(
                instance
                    .get_params()
                    .output_constraint_challenge_values(r_slice),
            );
            input_constraint_challenge_values.extend(
                instance
                    .get_params()
                    .input_constraint_challenge_values(opening_accumulator),
            );
        }
        Ok(StageVerifyResult::new(
            challenges,
            batched_output_constraint,
            output_constraint_challenge_values,
            batched_input_constraint,
            input_constraint_challenge_values,
        ))
    }
    #[cfg(not(feature = "zk"))]
    Ok(StageVerifyResult { challenges })
}

impl<
        'a,
        F: JoltField,
        C: JoltCurve,
        PCS: CommitmentScheme<Field = F> + ZkEvalCommitment<C>,
        ProofTranscript: Transcript,
    > JoltVerifier<'a, F, C, PCS, ProofTranscript>
{
    #[cfg_attr(not(feature = "zk"), allow(unused_variables))]
    pub(super) fn verify_stage1(
        &mut self,
    ) -> Result<(StageVerifyResult<F>, F::Challenge), ProofVerifyError> {
        let (uni_skip_params, uni_skip_challenge) = verify_stage1_uni_skip(
            &self.proof.stage1_uni_skip_first_round_proof,
            &self.spartan_key,
            &mut self.opening_accumulator,
            &mut self.transcript,
        )?;

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
        )?;

        #[cfg(feature = "zk")]
        {
            let batched_output_constraint = batch_output_constraints(&instances);
            let batched_input_constraint = batch_input_constraints(&instances);

            let max_num_rounds = instances.iter().map(|i| i.num_rounds()).max().unwrap();

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

            let mut input_constraint_challenge_values: Vec<F> =
                scale_batching_coefficients(&batching_coefficients, &instances);
            for instance in &instances {
                input_constraint_challenge_values.extend(
                    instance
                        .get_params()
                        .input_constraint_challenge_values(&self.opening_accumulator),
                );
            }

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
        #[cfg(not(feature = "zk"))]
        Ok((
            StageVerifyResult {
                challenges: r_stage1,
            },
            uni_skip_challenge,
        ))
    }

    #[cfg_attr(not(feature = "zk"), allow(unused_variables))]
    pub(super) fn verify_stage2(
        &mut self,
    ) -> Result<(StageVerifyResult<F>, F::Challenge), ProofVerifyError> {
        let (uni_skip_params, uni_skip_challenge) = verify_stage2_uni_skip(
            &self.proof.stage2_uni_skip_first_round_proof,
            &mut self.opening_accumulator,
            &mut self.transcript,
        )?;

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
        )?;

        #[cfg(feature = "zk")]
        {
            let batched_output_constraint = batch_output_constraints(&instances);
            let batched_input_constraint = batch_input_constraints(&instances);

            let max_num_rounds = instances.iter().map(|i| i.num_rounds()).max().unwrap();
            let mut output_constraint_challenge_values: Vec<F> = batching_coefficients.clone();
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
        #[cfg(not(feature = "zk"))]
        Ok((
            StageVerifyResult {
                challenges: r_stage2,
            },
            uni_skip_challenge,
        ))
    }

    pub(super) fn verify_stage3(&mut self) -> Result<StageVerifyResult<F>, ProofVerifyError> {
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

        verify_batched_stage(
            &mut self.transcript,
            &mut self.opening_accumulator,
            instances,
            &self.proof.stage3_sumcheck_proof,
        )
    }

    pub(super) fn verify_stage4(&mut self) -> Result<StageVerifyResult<F>, ProofVerifyError> {
        verifier_accumulate_advice::<F>(
            self.proof.ram_K,
            &self.program_io,
            self.proof.untrusted_advice_commitment.is_some(),
            self.trusted_advice_commitment.is_some(),
            &mut self.opening_accumulator,
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

        verify_batched_stage(
            &mut self.transcript,
            &mut self.opening_accumulator,
            instances,
            &self.proof.stage4_sumcheck_proof,
        )
    }

    pub(super) fn verify_stage5(&mut self) -> Result<StageVerifyResult<F>, ProofVerifyError> {
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

        verify_batched_stage(
            &mut self.transcript,
            &mut self.opening_accumulator,
            instances,
            &self.proof.stage5_sumcheck_proof,
        )
    }

    pub(super) fn verify_stage6(&mut self) -> Result<StageVerifyResult<F>, ProofVerifyError> {
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

        verify_batched_stage(
            &mut self.transcript,
            &mut self.opening_accumulator,
            instances,
            &self.proof.stage6_sumcheck_proof,
        )
    }

    pub(super) fn verify_stage7(&mut self) -> Result<StageVerifyResult<F>, ProofVerifyError> {
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
                params.phase = ReductionPhase::AddressVariables;
                instances.push(advice_reduction_verifier_trusted);
            }
        }
        if let Some(advice_reduction_verifier_untrusted) =
            self.advice_reduction_verifier_untrusted.as_mut()
        {
            let mut params = advice_reduction_verifier_untrusted.params.borrow_mut();
            if params.num_address_phase_rounds() > 0 {
                params.phase = ReductionPhase::AddressVariables;
                instances.push(advice_reduction_verifier_untrusted);
            }
        }

        verify_batched_stage(
            &mut self.transcript,
            &mut self.opening_accumulator,
            instances,
            &self.proof.stage7_sumcheck_proof,
        )
    }
}
