use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::subprotocols::sumcheck::BatchedSumcheck;
use crate::zkvm::config::OneHotParams;
use crate::zkvm::ram::val_final::ValFinalSumcheckVerifier;
use crate::zkvm::spartan::claim_reductions::RegistersClaimReductionSumcheckVerifier;
use crate::zkvm::witness::all_committed_polynomials;
use crate::zkvm::{
    bytecode::{
        self, read_raf_checking::ReadRafSumcheckVerifier as BytecodeReadRafSumcheckVerifier,
        BytecodePreprocessing,
    },
    fiat_shamir_preamble,
    instruction_lookups::{
        self, ra_virtual::RaSumcheckVerifier as LookupsRaSumcheckVerifier,
        read_raf_checking::ReadRafSumcheckVerifier as LookupsReadRafSumcheckVerifier,
    },
    proof_serialization::JoltProof,
    r1cs::key::UniformSpartanKey,
    ram::{
        self, hamming_booleanity::HammingBooleanitySumcheckVerifier,
        output_check::OutputSumcheckVerifier, ra_reduction::RamRaReductionSumcheckVerifier,
        ra_virtual::RamRaVirtualSumcheckVerifier,
        raf_evaluation::RafEvaluationSumcheckVerifier as RamRafEvaluationSumcheckVerifier,
        read_write_checking::RamReadWriteCheckingVerifier,
        val_evaluation::ValEvaluationSumcheckVerifier as RamValEvaluationSumcheckVerifier,
        verifier_accumulate_advice, RAMPreprocessing,
    },
    registers::{
        read_write_checking::RegistersReadWriteCheckingVerifier,
        val_evaluation::ValEvaluationSumcheckVerifier as RegistersValEvaluationSumcheckVerifier,
    },
    spartan::{
        claim_reductions::InstructionLookupsClaimReductionSumcheckVerifier,
        instruction_input::InstructionInputSumcheckVerifier, outer::OuterRemainingSumcheckVerifier,
        product::ProductVirtualRemainderVerifier, shift::ShiftSumcheckVerifier,
        verify_stage1_uni_skip, verify_stage2_uni_skip,
    },
    ProverDebugInfo, Serializable,
};
use crate::{
    field::JoltField,
    poly::opening_proof::{OpeningPoint, SumcheckId, VerifierOpeningAccumulator},
    pprof_scope,
    subprotocols::sumcheck_verifier::SumcheckInstanceVerifier,
    transcripts::Transcript,
    utils::{errors::ProofVerifyError, math::Math},
};
use anyhow::Context;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use common::jolt_device::MemoryLayout;
use itertools::zip_eq;
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
        if program_io.memory_layout != preprocessing.memory_layout {
            return Err(ProofVerifyError::MemoryLayoutMismatch);
        }
        if program_io.inputs.len() > preprocessing.memory_layout.max_input_size as usize {
            return Err(ProofVerifyError::InputTooLarge);
        }
        if program_io.outputs.len() > preprocessing.memory_layout.max_output_size as usize {
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
        let one_hot_params = OneHotParams::new_with_log_k_chunk(
            proof.log_k_chunk,
            proof.lookups_ra_virtual_log_k_chunk,
            proof.bytecode_K,
            proof.ram_K,
        );

        Ok(Self {
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

    #[tracing::instrument(skip_all)]
    pub fn verify(mut self) -> Result<(), anyhow::Error> {
        let _pprof_verify = pprof_scope!("verify");
        // Parameters are computed from trace length as needed

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

        self.verify_stage1()?;
        self.verify_stage2()?;
        self.verify_stage3()?;
        self.verify_stage4()?;
        self.verify_stage5()?;
        self.verify_stage6()?;
        self.verify_trusted_advice_opening_proofs()?;
        self.verify_untrusted_advice_opening_proofs()?;
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
        let registers_read_write_checking = RegistersReadWriteCheckingVerifier::new(
            self.proof.trace_length.log_2(),
            &self.opening_accumulator,
            &mut self.transcript,
        );
        verifier_accumulate_advice::<F>(
            self.proof.ram_K,
            &self.program_io,
            self.proof.untrusted_advice_commitment.is_some(),
            self.trusted_advice_commitment.is_some(),
            &mut self.opening_accumulator,
            &mut self.transcript,
            ram::read_write_checking::needs_single_advice_opening(self.proof.trace_length),
        );
        let ram_ra_booleanity = ram::new_ra_booleanity_verifier(
            self.proof.trace_length,
            &self.one_hot_params,
            &mut self.transcript,
        );
        let initial_ram_state = ram::gen_ram_initial_memory_state::<F>(
            self.proof.ram_K,
            &self.preprocessing.ram,
            &self.program_io,
        );
        let ram_val_evaluation = RamValEvaluationSumcheckVerifier::new(
            &initial_ram_state,
            &self.program_io,
            self.proof.trace_length,
            self.proof.ram_K,
            &self.opening_accumulator,
        );
        let ram_val_final = ValFinalSumcheckVerifier::new(
            &initial_ram_state,
            &self.program_io,
            self.proof.trace_length,
            self.proof.ram_K,
            &self.opening_accumulator,
        );

        let _r_stage4 = BatchedSumcheck::verify(
            &self.proof.stage4_sumcheck_proof,
            vec![
                &registers_read_write_checking as &dyn SumcheckInstanceVerifier<F, ProofTranscript>,
                &ram_ra_booleanity,
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
        let ram_hamming_booleanity =
            HammingBooleanitySumcheckVerifier::new(&self.opening_accumulator);
        let ram_ra_reduction = RamRaReductionSumcheckVerifier::new(
            self.proof.trace_length,
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
        );
        let lookups_read_raf = LookupsReadRafSumcheckVerifier::new(
            n_cycle_vars,
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
        );

        let _r_stage5 = BatchedSumcheck::verify(
            &self.proof.stage5_sumcheck_proof,
            vec![
                &registers_val_evaluation as &dyn SumcheckInstanceVerifier<F, ProofTranscript>,
                &ram_hamming_booleanity,
                &ram_ra_reduction,
                &lookups_read_raf,
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
            &self.preprocessing.bytecode,
            n_cycle_vars,
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
        );
        let (bytecode_hamming_weight, bytecode_booleanity) = bytecode::new_ra_one_hot_verifiers(
            self.proof.trace_length,
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
        );
        let ram_hamming_weight = ram::new_ra_hamming_weight_verifier(
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
        );
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
        let (lookups_ra_booleanity, lookups_rs_hamming_weight) =
            instruction_lookups::new_ra_one_hot_verifiers(
                self.proof.trace_length,
                &self.one_hot_params,
                &self.opening_accumulator,
                &mut self.transcript,
            );

        let _r_stage6 = BatchedSumcheck::verify(
            &self.proof.stage6_sumcheck_proof,
            vec![
                &bytecode_read_raf as &dyn SumcheckInstanceVerifier<F, ProofTranscript>,
                &bytecode_hamming_weight,
                &bytecode_booleanity,
                &ram_hamming_weight,
                &ram_ra_virtual,
                &lookups_ra_virtual,
                &lookups_ra_booleanity,
                &lookups_rs_hamming_weight,
            ],
            &mut self.opening_accumulator,
            &mut self.transcript,
        )
        .context("Stage 6")?;

        Ok(())
    }

    /// Stage 7: Batch opening reduction sumcheck verification.
    fn verify_stage7(&mut self) -> Result<(), anyhow::Error> {
        // Prepare - populate sumcheck claims
        self.opening_accumulator
            .prepare_for_sumcheck(&self.proof.stage7_sumcheck_claims);

        // Verify sumcheck
        let r_sumcheck = self
            .opening_accumulator
            .verify_batch_opening_sumcheck(&self.proof.stage7_sumcheck_proof, &mut self.transcript)
            .context("Stage 7")?;

        // Finalize and store state in accumulator for Stage 8
        let state = self.opening_accumulator.finalize_batch_opening_sumcheck(
            r_sumcheck,
            &self.proof.stage7_sumcheck_claims,
            &mut self.transcript,
        );

        self.opening_accumulator.opening_reduction_state = Some(state);

        Ok(())
    }

    /// Stage 8: Dory batch opening verification.
    fn verify_stage8(&mut self) -> Result<(), anyhow::Error> {
        let state = self
            .opening_accumulator
            .opening_reduction_state
            .as_ref()
            .expect("Stage 7 must be called before Stage 8");

        // Build commitments map
        let mut commitments_map = HashMap::from_iter(zip_eq(
            all_committed_polynomials(&self.one_hot_params),
            self.proof.commitments.iter().cloned(),
        ));
        // Compute joint commitment
        let joint_commitment = self
            .opening_accumulator
            .compute_joint_commitment::<PCS>(&mut commitments_map, state);

        // Test assertion
        #[cfg(test)]
        if let Some(ref prover_joint_commitment) = self.proof.joint_commitment_for_test {
            assert_eq!(
                joint_commitment, *prover_joint_commitment,
                "joint commitment mismatch"
            );
        }

        // Verify joint opening
        self.opening_accumulator
            .verify_joint_opening::<ProofTranscript, PCS>(
                &self.preprocessing.generators,
                &self.proof.joint_opening_proof,
                &joint_commitment,
                state,
                &mut self.transcript,
            )
            .context("Stage 8")
    }

    fn verify_trusted_advice_opening_proofs(&mut self) -> Result<(), anyhow::Error> {
        if let Some(ref commitment) = self.trusted_advice_commitment {
            // Verify at RamValEvaluation point
            let Some(ref proof) = self.proof.trusted_advice_val_evaluation_proof else {
                return Err(anyhow::anyhow!(
                    "Trusted advice val evaluation proof not found"
                ));
            };
            let Some((point, eval)) = self
                .opening_accumulator
                .get_trusted_advice_opening(SumcheckId::RamValEvaluation)
            else {
                return Err(anyhow::anyhow!("Trusted advice opening not found"));
            };
            PCS::verify(
                proof,
                &self.preprocessing.generators,
                &mut self.transcript,
                &point.r,
                &eval,
                commitment,
            )
            .map_err(|e| {
                anyhow::anyhow!("Trusted advice opening proof verification failed: {e:?}")
            })?;

            // Verify at RamValFinalEvaluation point - only if different from ValEvaluation
            if !ram::read_write_checking::needs_single_advice_opening(self.proof.trace_length) {
                let Some(ref proof_val_final) = self.proof.trusted_advice_val_final_proof else {
                    return Err(anyhow::anyhow!("Trusted advice val final proof not found"));
                };
                let Some((point_val_final, eval_val_final)) = self
                    .opening_accumulator
                    .get_trusted_advice_opening(SumcheckId::RamValFinalEvaluation)
                else {
                    return Err(anyhow::anyhow!(
                        "Trusted advice val final opening not found"
                    ));
                };
                PCS::verify(
                    proof_val_final,
                    &self.preprocessing.generators,
                    &mut self.transcript,
                    &point_val_final.r,
                    &eval_val_final,
                    commitment,
                )
                .map_err(|e| {
                    anyhow::anyhow!(
                        "Trusted advice val final opening proof verification failed: {e:?}"
                    )
                })?;
            }
        }

        Ok(())
    }

    fn verify_untrusted_advice_opening_proofs(&mut self) -> Result<(), anyhow::Error> {
        use crate::poly::opening_proof::SumcheckId;
        if let Some(ref commitment) = self.proof.untrusted_advice_commitment {
            // Verify at RamValEvaluation point
            let Some(ref proof) = self.proof.untrusted_advice_val_evaluation_proof else {
                return Err(anyhow::anyhow!(
                    "Untrusted advice val evaluation proof not found"
                ));
            };
            let Some((point, eval)) = self
                .opening_accumulator
                .get_untrusted_advice_opening(SumcheckId::RamValEvaluation)
            else {
                return Err(anyhow::anyhow!("Untrusted advice opening not found"));
            };
            PCS::verify(
                proof,
                &self.preprocessing.generators,
                &mut self.transcript,
                &point.r,
                &eval,
                commitment,
            )
            .map_err(|e| {
                anyhow::anyhow!("Untrusted advice opening proof verification failed: {e:?}")
            })?;

            // Verify at RamValFinalEvaluation point - only if different from ValEvaluation
            if !ram::read_write_checking::needs_single_advice_opening(self.proof.trace_length) {
                let Some(ref proof_val_final) = self.proof.untrusted_advice_val_final_proof else {
                    return Err(anyhow::anyhow!(
                        "Untrusted advice val final proof not found"
                    ));
                };
                let Some((point_val_final, eval_val_final)) = self
                    .opening_accumulator
                    .get_untrusted_advice_opening(SumcheckId::RamValFinalEvaluation)
                else {
                    return Err(anyhow::anyhow!(
                        "Untrusted advice val final opening not found"
                    ));
                };
                PCS::verify(
                    proof_val_final,
                    &self.preprocessing.generators,
                    &mut self.transcript,
                    &point_val_final.r,
                    &eval_val_final,
                    commitment,
                )
                .map_err(|e| {
                    anyhow::anyhow!(
                        "Untrusted advice val final opening proof verification failed: {e:?}"
                    )
                })?;
            }
        }

        Ok(())
    }
}

#[derive(Debug, Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct JoltVerifierPreprocessing<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    pub generators: PCS::VerifierSetup,
    pub bytecode: BytecodePreprocessing,
    pub ram: RAMPreprocessing,
    pub memory_layout: MemoryLayout,
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
