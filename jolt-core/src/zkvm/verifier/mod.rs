#[cfg(feature = "zk")]
mod blindfold;
mod opening;
pub(crate) mod preprocessing;
mod stages;

pub use preprocessing::{JoltSharedPreprocessing, JoltVerifierPreprocessing};

use crate::curve::JoltCurve;
use crate::field::JoltField;
use crate::poly::commitment::commitment_scheme::{CommitmentScheme, ZkEvalCommitment};
use crate::poly::commitment::pedersen::PedersenGenerators;
use crate::poly::opening_proof::VerifierOpeningAccumulator;
use crate::pprof_scope;
use crate::transcripts::Transcript;
use crate::utils::errors::ProofVerifyError;
use crate::utils::math::Math;
use crate::zkvm::claim_reductions::AdviceClaimReductionVerifier;
use crate::zkvm::config::OneHotParams;
use crate::zkvm::proof_serialization::JoltProof;
use crate::zkvm::r1cs::key::UniformSpartanKey;
use crate::zkvm::{fiat_shamir_preamble, ProverDebugInfo};

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
    advice_reduction_verifier_trusted: Option<AdviceClaimReductionVerifier<F>>,
    advice_reduction_verifier_untrusted: Option<AdviceClaimReductionVerifier<F>>,
    pub spartan_key: UniformSpartanKey<F>,
    pub one_hot_params: OneHotParams,
    pub pedersen_generators: PedersenGenerators<C>,
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
        if program_io.memory_layout != preprocessing.shared.memory_layout {
            return Err(ProofVerifyError::MemoryLayoutMismatch);
        }
        if program_io.inputs.len() > preprocessing.shared.memory_layout.max_input_size as usize {
            return Err(ProofVerifyError::InputTooLarge);
        }
        if program_io.outputs.len() > preprocessing.shared.memory_layout.max_output_size as usize {
            return Err(ProofVerifyError::OutputTooLarge);
        }

        program_io.outputs.truncate(
            program_io
                .outputs
                .iter()
                .rposition(|&b| b != 0)
                .map_or(0, |pos| pos + 1),
        );

        let zk_mode = proof.stage1_sumcheck_proof.is_zk();
        #[cfg(test)]
        #[allow(unused_mut)]
        let mut opening_accumulator =
            VerifierOpeningAccumulator::new(proof.trace_length.log_2(), zk_mode);
        #[cfg(not(test))]
        #[allow(unused_mut)]
        let mut opening_accumulator =
            VerifierOpeningAccumulator::new(proof.trace_length.log_2(), zk_mode);

        #[cfg(not(feature = "zk"))]
        {
            use crate::poly::opening_proof::{OpeningPoint, BIG_ENDIAN};
            for (id, (_, claim)) in &proof.opening_claims.0 {
                let dummy_point = OpeningPoint::<BIG_ENDIAN, F>::new(vec![]);
                opening_accumulator
                    .openings
                    .insert(*id, (dummy_point, *claim));
            }
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

        proof
            .one_hot_config
            .validate()
            .map_err(ProofVerifyError::InvalidOneHotConfig)?;

        proof
            .rw_config
            .validate(proof.trace_length.log_2(), proof.ram_K.log_2())
            .map_err(ProofVerifyError::InvalidReadWriteConfig)?;

        let one_hot_params =
            OneHotParams::from_config(&proof.one_hot_config, proof.bytecode_K, proof.ram_K);

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
    #[cfg_attr(not(feature = "zk"), allow(unused_variables))]
    pub fn verify(mut self) -> Result<(), ProofVerifyError> {
        let _pprof_verify = pprof_scope!("verify");
        let zk_mode = self.opening_accumulator.zk_mode;

        fiat_shamir_preamble(
            &self.program_io,
            self.proof.ram_K,
            self.proof.trace_length,
            &mut self.transcript,
        );

        for commitment in &self.proof.commitments {
            self.transcript
                .append_serializable(b"commitment", commitment);
        }
        if let Some(ref untrusted_advice_commitment) = self.proof.untrusted_advice_commitment {
            self.transcript
                .append_serializable(b"untrusted_advice", untrusted_advice_commitment);
        }
        if let Some(ref trusted_advice_commitment) = self.trusted_advice_commitment {
            self.transcript
                .append_serializable(b"trusted_advice", trusted_advice_commitment);
        }

        let (stage1_result, uniskip_challenge1) = self
            .verify_stage1()
            .inspect_err(|e| tracing::error!("Stage 1: {e}"))?;
        let (stage2_result, uniskip_challenge2) = self
            .verify_stage2()
            .inspect_err(|e| tracing::error!("Stage 2: {e}"))?;
        let stage3_result = self
            .verify_stage3()
            .inspect_err(|e| tracing::error!("Stage 3: {e}"))?;
        let stage4_result = self
            .verify_stage4()
            .inspect_err(|e| tracing::error!("Stage 4: {e}"))?;
        let stage5_result = self
            .verify_stage5()
            .inspect_err(|e| tracing::error!("Stage 5: {e}"))?;
        let stage6_result = self
            .verify_stage6()
            .inspect_err(|e| tracing::error!("Stage 6: {e}"))?;
        let stage7_result = self
            .verify_stage7()
            .inspect_err(|e| tracing::error!("Stage 7: {e}"))?;
        let stage8_data = self
            .verify_stage8()
            .inspect_err(|e| tracing::error!("Stage 8: {e}"))?;

        if zk_mode {
            #[cfg(feature = "zk")]
            {
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

                let stage_output_constraints = [
                    stage1_result.batched_output_constraint,
                    stage2_result.batched_output_constraint,
                    stage3_result.batched_output_constraint,
                    stage4_result.batched_output_constraint,
                    stage5_result.batched_output_constraint,
                    stage6_result.batched_output_constraint,
                    stage7_result.batched_output_constraint,
                ];

                let stage_input_constraints = [
                    stage1_result.uniskip_input_constraint.clone().unwrap(),
                    stage2_result.uniskip_input_constraint.clone().unwrap(),
                    stage3_result.batched_input_constraint.clone(),
                    stage4_result.batched_input_constraint.clone(),
                    stage5_result.batched_input_constraint.clone(),
                    stage6_result.batched_input_constraint.clone(),
                    stage7_result.batched_input_constraint.clone(),
                ];

                let stage_input_constraint_values = [
                    stage1_result
                        .uniskip_input_constraint_challenge_values
                        .clone(),
                    stage2_result
                        .uniskip_input_constraint_challenge_values
                        .clone(),
                    stage3_result.input_constraint_challenge_values.clone(),
                    stage4_result.input_constraint_challenge_values.clone(),
                    stage5_result.input_constraint_challenge_values.clone(),
                    stage6_result.input_constraint_challenge_values.clone(),
                    stage7_result.input_constraint_challenge_values.clone(),
                ];

                let output_constraint_challenge_values: [Vec<F>; 7] = [
                    stage1_result.output_constraint_challenge_values.clone(),
                    stage2_result.output_constraint_challenge_values.clone(),
                    stage3_result.output_constraint_challenge_values.clone(),
                    stage4_result.output_constraint_challenge_values.clone(),
                    stage5_result.output_constraint_challenge_values.clone(),
                    stage6_result.output_constraint_challenge_values.clone(),
                    stage7_result.output_constraint_challenge_values.clone(),
                ];

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
            }
            #[cfg(not(feature = "zk"))]
            return Err(ProofVerifyError::ZkFeatureRequired);
        }

        Ok(())
    }
}
