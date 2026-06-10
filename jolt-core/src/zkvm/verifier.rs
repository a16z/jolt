use crate::curve::JoltCurve;
use crate::poly::commitment::commitment_scheme::{CommitmentScheme, ZkEvalCommitment};
#[cfg(feature = "zk")]
use crate::poly::commitment::dory::bind_opening_inputs_zk;
use crate::poly::commitment::dory::{bind_opening_inputs, DoryContext, DoryGlobals};
use crate::poly::commitment::pedersen::PedersenGenerators;
#[cfg(feature = "zk")]
use crate::poly::lagrange_poly::LagrangeHelper;
#[cfg(feature = "zk")]
use crate::poly::opening_proof::AbstractVerifierOpeningAccumulator;
#[cfg(feature = "zk")]
use crate::subprotocols::blindfold::{
    pedersen_generator_count_for_r1cs, BakedPublicInputs, BlindFoldVerifier,
    BlindFoldVerifierInput, ClaimBindingConfig, InputClaimConstraint, OutputClaimConstraint,
    StageConfig, ValueSource, VerifierR1CSBuilder,
};
use crate::subprotocols::sumcheck::BatchedSumcheck;
#[cfg(feature = "zk")]
use crate::subprotocols::sumcheck::ZkSumcheckReadback;
#[cfg(feature = "zk")]
use crate::subprotocols::sumcheck_verifier::SumcheckInstanceParams;
#[cfg(feature = "zk")]
use crate::subprotocols::univariate_skip::ZkUniSkipReadback;
use crate::zkvm::bytecode::chunks::DEFAULT_COMMITTED_BYTECODE_CHUNK_COUNT;
use crate::zkvm::bytecode::chunks::{
    committed_lanes, is_valid_committed_bytecode_chunking_for_len,
};
use crate::zkvm::claim_reductions::RegistersClaimReductionSumcheckVerifier;
use crate::zkvm::config::{OneHotParams, ProgramMode};
use crate::zkvm::program::{CommittedProgramProverData, ProgramMetadata, ProgramPreprocessing};
#[cfg(feature = "prover")]
use crate::zkvm::prover::JoltProverPreprocessing;
#[cfg(feature = "zk")]
use crate::zkvm::r1cs::constraints::{
    OUTER_FIRST_ROUND_POLY_NUM_COEFFS, OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE,
    PRODUCT_VIRTUAL_FIRST_ROUND_POLY_NUM_COEFFS, PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE,
};
use crate::zkvm::witness::all_committed_polynomials;
use crate::zkvm::Serializable;
use crate::zkvm::{
    bytecode::read_raf_checking::{
        BytecodeReadRafAddressSumcheckVerifier, BytecodeReadRafCycleSumcheckVerifier,
        BytecodeReadRafSumcheckParams,
    },
    claim_reductions::{
        AdviceClaimReductionVerifier, AdviceKind, BytecodeClaimReductionParams,
        BytecodeClaimReductionVerifier, HammingWeightClaimReductionVerifier,
        IncClaimReductionSumcheckVerifier, InstructionLookupsClaimReductionSumcheckVerifier,
        PrecommittedClaimReduction, PrecommittedParams, ProgramImageClaimReductionParams,
        ProgramImageClaimReductionVerifier, RamRaClaimReductionSumcheckVerifier,
    },
    compute_final_opening_point, fiat_shamir_instance,
    instruction_lookups::{
        ra_virtual::RaSumcheckVerifier as LookupsRaSumcheckVerifier,
        read_raf_checking::InstructionReadRafSumcheckVerifier,
    },
    proof_serialization::JoltProof,
    r1cs::key::UniformSpartanKey,
    ram::{
        compute_max_ram_K, compute_min_ram_K,
        hamming_booleanity::HammingBooleanitySumcheckVerifier,
        output_check::OutputSumcheckVerifier, ra_virtual::RamRaVirtualSumcheckVerifier,
        raf_evaluation::RafEvaluationSumcheckVerifier as RamRafEvaluationSumcheckVerifier,
        read_write_checking::RamReadWriteCheckingVerifier, val_check::RamValCheckSumcheckVerifier,
        verifier_accumulate_advice, verifier_accumulate_program_image,
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
        compute_lagrange_factor, DoryOpeningState, OpeningAccumulator, OpeningId, SumcheckId,
        VerifierOpeningAccumulator,
    },
    pprof_scope,
    subprotocols::{
        booleanity::{
            BooleanityAddressSumcheckVerifier, BooleanityCycleSumcheckVerifier,
            BooleanitySumcheckParams,
        },
        sumcheck_verifier::SumcheckInstanceVerifier,
    },
    transcript_msgs::{FsAbsorb, VerifierFs},
    utils::{errors::ProofVerifyError, math::Math},
    zkvm::witness::CommittedPolynomial,
};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

#[cfg(feature = "zk")]
struct StageVerifyResult<F: JoltField> {
    challenges: Vec<F::Challenge>,
    batched_output_constraint: Option<OutputClaimConstraint>,
    output_constraint_challenge_values: Vec<F>,
    batched_input_constraint: InputClaimConstraint,
    input_constraint_challenge_values: Vec<F>,
    uniskip_input_constraint: Option<InputClaimConstraint>,
    uniskip_input_constraint_challenge_values: Vec<F>,
    uniskip_output_constraint: Option<OutputClaimConstraint>,
    uniskip_output_constraint_challenge_values: Vec<F>,
    oc_block_ids: Vec<Vec<OpeningId>>,
}

#[cfg(not(feature = "zk"))]
struct StageVerifyResult<F: JoltField> {
    #[allow(dead_code)]
    challenges: Vec<F::Challenge>,
}

type Stage6aVerifyResult<F> = (
    BytecodeReadRafSumcheckParams<F>,
    BooleanitySumcheckParams<F>,
    StageVerifyResult<F>,
);

#[cfg(feature = "zk")]
impl<F: JoltField> StageVerifyResult<F> {
    fn new(
        challenges: Vec<F::Challenge>,
        batched_output_constraint: Option<OutputClaimConstraint>,
        output_constraint_challenge_values: Vec<F>,
        batched_input_constraint: InputClaimConstraint,
        input_constraint_challenge_values: Vec<F>,
        oc_block_ids: Vec<Vec<OpeningId>>,
    ) -> Self {
        Self {
            challenges,
            batched_output_constraint,
            output_constraint_challenge_values,
            batched_input_constraint,
            input_constraint_challenge_values,
            uniskip_input_constraint: None,
            uniskip_input_constraint_challenge_values: Vec::new(),
            uniskip_output_constraint: None,
            uniskip_output_constraint_challenge_values: Vec::new(),
            oc_block_ids,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn with_uniskip(
        challenges: Vec<F::Challenge>,
        batched_output_constraint: Option<OutputClaimConstraint>,
        output_constraint_challenge_values: Vec<F>,
        batched_input_constraint: InputClaimConstraint,
        input_constraint_challenge_values: Vec<F>,
        uniskip_input_constraint: InputClaimConstraint,
        uniskip_input_constraint_challenge_values: Vec<F>,
        uniskip_output_constraint: Option<OutputClaimConstraint>,
        uniskip_output_constraint_challenge_values: Vec<F>,
        oc_block_ids: Vec<Vec<OpeningId>>,
    ) -> Self {
        Self {
            challenges,
            batched_output_constraint,
            output_constraint_challenge_values,
            batched_input_constraint,
            input_constraint_challenge_values,
            uniskip_input_constraint: Some(uniskip_input_constraint),
            uniskip_input_constraint_challenge_values,
            uniskip_output_constraint,
            uniskip_output_constraint_challenge_values,
            oc_block_ids,
        }
    }
}

#[cfg(feature = "zk")]
fn batch_output_constraints<F: JoltField, A: AbstractVerifierOpeningAccumulator<F>>(
    instances: &[&dyn SumcheckInstanceVerifier<F, A>],
) -> Option<OutputClaimConstraint> {
    let constraints: Vec<Option<OutputClaimConstraint>> = instances
        .iter()
        .map(|instance| instance.get_params().output_claim_constraint())
        .collect();
    OutputClaimConstraint::batch(&constraints)
}

#[cfg(feature = "zk")]
fn batch_input_constraints<F: JoltField, A: AbstractVerifierOpeningAccumulator<F>>(
    instances: &[&dyn SumcheckInstanceVerifier<F, A>],
) -> InputClaimConstraint {
    let constraints: Vec<InputClaimConstraint> = instances
        .iter()
        .map(|instance| instance.get_params().input_claim_constraint())
        .collect();
    InputClaimConstraint::batch_required(&constraints, instances.len())
}

#[cfg(feature = "zk")]
fn scale_batching_coefficients<F: JoltField, A: AbstractVerifierOpeningAccumulator<F>>(
    batching_coefficients: &[F],
    instances: &[&dyn SumcheckInstanceVerifier<F, A>],
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
use jolt_transcript::{verifier_transcript, DuplexSpongeInterface, VerifierState};
use tracer::JoltDevice;

pub struct JoltVerifier<
    'a,
    F: JoltField,
    C: JoltCurve<F = F>,
    PCS: CommitmentScheme<Field = F>,
    H: DuplexSpongeInterface,
> {
    pub trusted_advice_commitment: Option<PCS::Commitment>,
    pub program_io: JoltDevice,
    pub proof: JoltProof<F, C, PCS, H>,
    /// Witness-polynomial commitments, decoded from the NARG in `verify_inner`
    /// (one `read_slice` frame) before stage 1. Populated at FS time, not at
    /// construction (the NARG isn't read until `verify_inner`).
    commitments: Vec<PCS::Commitment>,
    /// Untrusted-advice commitment, decoded from the NARG presence frame
    /// (length-0/1) in `verify_inner`. `None` if absent.
    untrusted_advice_commitment: Option<PCS::Commitment>,
    pub preprocessing: &'a JoltVerifierPreprocessing<F, C, PCS>,
    pub opening_accumulator: VerifierOpeningAccumulator<F>,
    /// The advice claim reduction sumcheck effectively spans two stages (6 and 7).
    /// Cache the verifier state here between stages.
    advice_reduction_verifier_trusted: Option<AdviceClaimReductionVerifier<F>>,
    /// The advice claim reduction sumcheck effectively spans two stages (6 and 7).
    /// Cache the verifier state here between stages.
    advice_reduction_verifier_untrusted: Option<AdviceClaimReductionVerifier<F>>,
    /// Bytecode claim reduction spans stages 6b and 7 in committed mode.
    bytecode_reduction_verifier: Option<BytecodeClaimReductionVerifier<F>>,
    /// Program-image claim reduction spans stages 6b and 7 in committed mode.
    program_image_reduction_verifier: Option<ProgramImageClaimReductionVerifier<F>>,
    pub spartan_key: UniformSpartanKey<F>,
    pub one_hot_params: OneHotParams,
    /// ZK sumcheck values (round commitments, per-round degrees, output-claim commitments)
    /// read back from the NARG during each stage's `BatchedSumcheck::verify`, indexed by
    /// stage (0 = stage1 … 7 = stage7, where 5 = stage6a and 6 = stage6b). Since the proof
    /// structs are now data-free, stage 8 (BlindFold) is fed from here instead. Populated
    /// only in ZK mode.
    #[cfg(feature = "zk")]
    zk_sumcheck_readback: Vec<Option<ZkSumcheckReadback<C>>>,
    /// ZK uni-skip values (commitment, degree, output-claim commitments) read back from the
    /// NARG during stages 1–2, indexed by uni-skip stage (0 = stage1, 1 = stage2). Populated
    /// only in ZK mode.
    #[cfg(feature = "zk")]
    zk_uniskip_readback: Vec<Option<ZkUniSkipReadback<C>>>,
}

#[derive(Clone, Debug)]
#[cfg_attr(not(feature = "zk"), allow(dead_code))]
struct Stage8VerifyData<F: JoltField> {
    opening_ids: Vec<OpeningId>,
    constraint_coeffs: Vec<F>,
}

impl<
        'a,
        F: JoltField,
        C: JoltCurve<F = F>,
        PCS: CommitmentScheme<Field = F> + ZkEvalCommitment<C>,
        H: DuplexSpongeInterface<U = u8> + Default,
    > JoltVerifier<'a, F, C, PCS, H>
where
    for<'b> VerifierState<'b, H>: VerifierFs<F>,
{
    #[inline]
    fn main_total_vars(&self) -> usize {
        let trace_log_t = self.proof.trace_length.log_2();
        let log_k_chunk = self.one_hot_params.log_k_chunk;
        JoltSharedPreprocessing::<PCS>::max_total_vars_from_candidates(
            trace_log_t + log_k_chunk,
            self.preprocessing.shared.precommitted_candidate_total_vars(
                self.preprocessing.shared.program.is_committed(),
                self.trusted_advice_commitment.is_some(),
                self.untrusted_advice_commitment.is_some(),
            ),
        )
    }

    pub fn new(
        preprocessing: &'a JoltVerifierPreprocessing<F, C, PCS>,
        proof: JoltProof<F, C, PCS, H>,
        mut program_io: JoltDevice,
        trusted_advice_commitment: Option<PCS::Commitment>,
        _debug_info: Option<ProverDebugInfo<F, H, PCS>>,
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

        // Validate trace_length: must be a power of 2 and within the preprocessed bound
        if !proof.trace_length.is_power_of_two()
            || proof.trace_length > preprocessing.shared.max_padded_trace_length
        {
            return Err(ProofVerifyError::InvalidTraceLength(
                proof.trace_length,
                preprocessing.shared.max_padded_trace_length,
            ));
        }

        // Truncate trailing zero bytes from outputs. Both prover and verifier
        // apply the same truncation so the proof is internally consistent.
        // WARNING: callers reading `program_io.outputs` directly after verification
        // will see truncated data. The SDK re-pads to `max_output_size` before
        // deserialization, but direct consumers must account for this.
        program_io.outputs.truncate(
            program_io
                .outputs
                .iter()
                .rposition(|&b| b != 0)
                .map_or(0, |pos| pos + 1),
        );

        let zk_mode = proof.zk_mode;
        // `zk_mode` is an attacker-controlled proof field, but the verifier's ability to
        // check a given mode is fixed at compile time: a `zk` build's `JoltProof` has no
        // `opening_claims` field (so non-ZK proofs cannot be verified), and a non-`zk`
        // build has no BlindFold code (so ZK proofs cannot be verified). Reject the
        // mismatch here, explicitly, instead of failing somewhere downstream with an
        // empty opening accumulator or a missing-stage error.
        if zk_mode != cfg!(feature = "zk") {
            return Err(ProofVerifyError::ZkModeMismatch);
        }
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

        // The verifier transcript (VerifierState) borrows the proof's NARG, so it is
        // built locally in `verify_inner`, not stored as a field (D3). NARG consistency
        // is inherent (the verifier replays the prover's NARG in lock-step).
        #[cfg(test)]
        {
            if let Some(debug_info) = _debug_info {
                opening_accumulator.compare_to(debug_info.opening_accumulator);
            }
        }

        let spartan_key = UniformSpartanKey::new(proof.trace_length.next_power_of_two());

        // Validate configs from the proof
        proof
            .one_hot_config
            .validate()
            .map_err(ProofVerifyError::InvalidOneHotConfig)?;

        let ram_preprocessing = crate::zkvm::ram::RAMPreprocessing {
            min_bytecode_address: preprocessing.shared.program_meta.min_bytecode_address,
            bytecode_words: vec![0; preprocessing.shared.program_meta.program_image_len_words],
        };
        let min_ram_K = compute_min_ram_K(&ram_preprocessing, &preprocessing.shared.memory_layout);
        let max_ram_K = compute_max_ram_K(&preprocessing.shared.memory_layout);
        if !proof.ram_K.is_power_of_two() || proof.ram_K < min_ram_K || proof.ram_K > max_ram_K {
            return Err(ProofVerifyError::InvalidRamK {
                got: proof.ram_K,
                min: min_ram_K,
                max: max_ram_K,
            });
        }

        proof
            .rw_config
            .validate(proof.trace_length.log_2(), proof.ram_K.log_2())
            .map_err(ProofVerifyError::InvalidReadWriteConfig)?;

        // Construct full params from the validated config.
        let bytecode_K = preprocessing.shared.bytecode_size();
        let one_hot_params =
            OneHotParams::from_config(&proof.one_hot_config, bytecode_K, proof.ram_K);

        Ok(Self {
            trusted_advice_commitment,
            program_io,
            proof,
            commitments: Vec::new(),
            untrusted_advice_commitment: None,
            preprocessing,
            opening_accumulator,
            advice_reduction_verifier_trusted: None,
            advice_reduction_verifier_untrusted: None,
            bytecode_reduction_verifier: None,
            program_image_reduction_verifier: None,
            spartan_key,
            one_hot_params,
            #[cfg(feature = "zk")]
            zk_sumcheck_readback: (0..8).map(|_| None).collect(),
            #[cfg(feature = "zk")]
            zk_uniskip_readback: (0..2).map(|_| None).collect(),
        })
    }

    #[tracing::instrument(skip_all)]
    pub fn verify(self) -> Result<(), ProofVerifyError> {
        // In test/debug builds, let panics propagate for full backtraces.
        // In release builds, catch panics from malformed proofs (e.g., missing
        // opening claims) and convert them to clean error returns.
        #[cfg(any(test, debug_assertions))]
        {
            self.verify_inner()
        }
        #[cfg(not(any(test, debug_assertions)))]
        {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.verify_inner()))
                .unwrap_or_else(|payload| {
                    let msg = payload
                        .downcast_ref::<&str>()
                        .map(|s| s.to_string())
                        .or_else(|| payload.downcast_ref::<String>().cloned())
                        .unwrap_or_else(|| "unknown panic".to_string());
                    tracing::error!("Verifier panicked on malformed proof: {msg}");
                    Err(ProofVerifyError::InternalError)
                })
        }
    }

    #[cfg_attr(not(feature = "zk"), allow(unused_variables))]
    fn verify_inner(mut self) -> Result<(), ProofVerifyError> {
        let _pprof_verify = pprof_scope!("verify");
        let zk_mode = self.opening_accumulator.zk_mode;

        // Build the verifier transcript locally over the proof's NARG (D3). A.1: the
        // public statement is bound into the `instance` digest (`Blake2b(statement)`),
        // recomputed from the proof's public tail — byte-identical to the prover (O1).
        let preprocessing_digest = self.preprocessing.shared.digest();
        let instance = fiat_shamir_instance(
            &self.program_io,
            self.proof.ram_K,
            self.proof.trace_length,
            self.preprocessing.shared.program_meta.entry_address,
            &self.proof.rw_config,
            &self.proof.one_hot_config,
            self.proof.dory_layout,
            &preprocessing_digest,
        );
        let narg = std::mem::take(&mut self.proof.narg);
        let mut ts = verifier_transcript(b"Jolt", instance, H::default(), &narg);
        let transcript = &mut ts;

        // Initialize DoryGlobals with the layout from the proof
        // This ensures the verifier uses the same layout as the prover
        let _guard = DoryGlobals::initialize_context(
            1 << self.one_hot_params.log_k_chunk,
            self.proof.trace_length.next_power_of_two(),
            DoryContext::Main,
            Some(self.proof.dory_layout),
        );

        // Read the witness-polynomial commitments back from the NARG as ONE frame
        // (matching the prover's single `write_slice`), absorbing them in the process.
        self.commitments = transcript
            .read_slice()
            .map_err(|_| ProofVerifyError::SumcheckVerificationError)?;
        // Read the untrusted-advice presence frame (length-0/1 vec) and reconstruct
        // the Option. The prover ALWAYS writes this frame, so the read position is the
        // same in the Some and None cases.
        let untrusted_advice: Vec<PCS::Commitment> = transcript
            .read_slice()
            .map_err(|_| ProofVerifyError::SumcheckVerificationError)?;
        // The presence frame is length 0 (None) or 1 (Some); reject any over-long frame
        // rather than silently dropping extra entries via `.next()`.
        if untrusted_advice.len() > 1 {
            return Err(ProofVerifyError::SumcheckVerificationError);
        }
        self.untrusted_advice_commitment = untrusted_advice.into_iter().next();
        // Append trusted advice commitment to transcript
        if let Some(ref trusted_advice_commitment) = self.trusted_advice_commitment {
            transcript.absorb(trusted_advice_commitment);
        }
        if let Some(trusted_bytecode) = self.preprocessing.shared.program.bytecode_commitments() {
            for commitment in &trusted_bytecode.commitments {
                transcript.absorb(commitment);
            }
        }
        if self.preprocessing.shared.program.is_committed() {
            let trusted = self.preprocessing.shared.program.as_committed()?;
            transcript.absorb(&trusted.program_image_commitment);
        }

        let (stage1_result, uniskip_challenge1) = self
            .verify_stage1(transcript)
            .inspect_err(|e| tracing::error!("Stage 1: {e}"))?;
        let (stage2_result, uniskip_challenge2) = self
            .verify_stage2(transcript)
            .inspect_err(|e| tracing::error!("Stage 2: {e}"))?;
        let stage3_result = self
            .verify_stage3(transcript)
            .inspect_err(|e| tracing::error!("Stage 3: {e}"))?;
        let stage4_result = self
            .verify_stage4(transcript)
            .inspect_err(|e| tracing::error!("Stage 4: {e}"))?;
        let stage5_result = self
            .verify_stage5(transcript)
            .inspect_err(|e| tracing::error!("Stage 5: {e}"))?;
        let (stage6a_result, stage6b_result) = self
            .verify_stage6(transcript)
            .inspect_err(|e| tracing::error!("Stage 6: {e}"))?;
        let stage7_result = self
            .verify_stage7(transcript)
            .inspect_err(|e| tracing::error!("Stage 7: {e}"))?;
        let stage8_data = self
            .verify_stage8(transcript)
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
                    stage6a_result.challenges.clone(),
                    stage6b_result.challenges.clone(),
                    stage7_result.challenges.clone(),
                ];
                let uniskip_challenges = [uniskip_challenge1, uniskip_challenge2];

                let stage_output_constraints = [
                    stage1_result.batched_output_constraint,
                    stage2_result.batched_output_constraint,
                    stage3_result.batched_output_constraint,
                    stage4_result.batched_output_constraint,
                    stage5_result.batched_output_constraint,
                    stage6a_result.batched_output_constraint,
                    stage6b_result.batched_output_constraint,
                    stage7_result.batched_output_constraint,
                ];

                let stage_input_constraints = [
                    stage1_result.uniskip_input_constraint.clone().unwrap(),
                    stage2_result.uniskip_input_constraint.clone().unwrap(),
                    stage3_result.batched_input_constraint.clone(),
                    stage4_result.batched_input_constraint.clone(),
                    stage5_result.batched_input_constraint.clone(),
                    stage6a_result.batched_input_constraint.clone(),
                    stage6b_result.batched_input_constraint.clone(),
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
                    stage6a_result.input_constraint_challenge_values.clone(),
                    stage6b_result.input_constraint_challenge_values.clone(),
                    stage7_result.input_constraint_challenge_values.clone(),
                ];

                let output_constraint_challenge_values: [Vec<F>; 8] = [
                    stage1_result.output_constraint_challenge_values.clone(),
                    stage2_result.output_constraint_challenge_values.clone(),
                    stage3_result.output_constraint_challenge_values.clone(),
                    stage4_result.output_constraint_challenge_values.clone(),
                    stage5_result.output_constraint_challenge_values.clone(),
                    stage6a_result.output_constraint_challenge_values.clone(),
                    stage6b_result.output_constraint_challenge_values.clone(),
                    stage7_result.output_constraint_challenge_values.clone(),
                ];

                let mut oc_blocks: Vec<Vec<OpeningId>> = Vec::new();
                oc_blocks.extend(stage1_result.oc_block_ids);
                oc_blocks.extend(stage2_result.oc_block_ids);
                oc_blocks.extend(stage3_result.oc_block_ids);
                oc_blocks.extend(stage4_result.oc_block_ids);
                oc_blocks.extend(stage5_result.oc_block_ids);
                oc_blocks.extend(stage6a_result.oc_block_ids);
                oc_blocks.extend(stage6b_result.oc_block_ids);
                oc_blocks.extend(stage7_result.oc_block_ids);

                let uniskip_output_constraints = [
                    stage1_result.uniskip_output_constraint.clone(),
                    stage2_result.uniskip_output_constraint.clone(),
                ];
                let uniskip_output_challenge_values = [
                    stage1_result
                        .uniskip_output_constraint_challenge_values
                        .clone(),
                    stage2_result
                        .uniskip_output_constraint_challenge_values
                        .clone(),
                ];

                self.verify_blindfold(
                    transcript,
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
                    &uniskip_output_constraints,
                    &uniskip_output_challenge_values,
                    &stage8_data,
                    oc_blocks,
                )?;
            }
            #[cfg(not(feature = "zk"))]
            return Err(ProofVerifyError::ZkFeatureRequired);
        }

        // Soundness (malleability guard): the NARG must be fully consumed — reject any
        // trailing/garbage bytes. Error paths above already returned before reaching here.
        ts.check_eof()
            .map_err(|_| ProofVerifyError::SumcheckVerificationError)?;
        Ok(())
    }

    #[cfg_attr(not(feature = "zk"), allow(unused_variables))]
    fn verify_stage1(
        &mut self,
        transcript: &mut impl VerifierFs<F>,
    ) -> Result<(StageVerifyResult<F>, F::Challenge), ProofVerifyError> {
        let (uni_skip_params, uni_skip_challenge, zk_uniskip_readback) =
            verify_stage1_uni_skip::<F, C, _, _>(
                self.proof.zk_mode,
                &self.spartan_key,
                &mut self.opening_accumulator,
                transcript,
            )?;

        // Drain uniskip OC block IDs (pending_claims were drained inside verify_transcript)
        #[cfg(feature = "zk")]
        let uniskip_oc_ids = self.opening_accumulator.take_pending_claim_ids();

        let spartan_outer_remaining = OuterRemainingSumcheckVerifier::new(
            self.spartan_key,
            self.proof.trace_length,
            &uni_skip_params,
            &self.opening_accumulator,
        );

        let instances: Vec<&dyn SumcheckInstanceVerifier<F, VerifierOpeningAccumulator<F>>> =
            vec![&spartan_outer_remaining];

        let (batching_coefficients, r_stage1, zk_sumcheck_readback) =
            BatchedSumcheck::verify::<F, C>(
                self.proof.zk_mode,
                instances.clone(),
                &mut self.opening_accumulator,
                transcript,
            )?;

        #[cfg(feature = "zk")]
        {
            let regular_oc_ids = self.opening_accumulator.take_pending_claim_ids();

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
            let uniskip_output_constraint = uni_skip_params.output_claim_constraint();
            let uniskip_output_constraint_challenge_values =
                uni_skip_params.output_constraint_challenge_values(&[uni_skip_challenge]);

            let stage_result = StageVerifyResult::with_uniskip(
                r_stage1,
                batched_output_constraint,
                output_constraint_challenge_values,
                batched_input_constraint,
                input_constraint_challenge_values,
                uniskip_input_constraint,
                uniskip_input_constraint_challenge_values,
                uniskip_output_constraint,
                uniskip_output_constraint_challenge_values,
                vec![uniskip_oc_ids, regular_oc_ids],
            );

            self.zk_uniskip_readback[0] = zk_uniskip_readback;
            self.zk_sumcheck_readback[0] = zk_sumcheck_readback;

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
    fn verify_stage2(
        &mut self,
        transcript: &mut impl VerifierFs<F>,
    ) -> Result<(StageVerifyResult<F>, F::Challenge), ProofVerifyError> {
        let (uni_skip_params, uni_skip_challenge, zk_uniskip_readback) =
            verify_stage2_uni_skip::<F, C, _, _>(
                self.proof.zk_mode,
                &mut self.opening_accumulator,
                transcript,
            )?;

        #[cfg(feature = "zk")]
        let uniskip_oc_ids = self.opening_accumulator.take_pending_claim_ids();

        let ram_read_write_checking = RamReadWriteCheckingVerifier::new(
            &self.opening_accumulator,
            transcript,
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
            transcript,
        );

        let ram_raf_evaluation = RamRafEvaluationSumcheckVerifier::new(
            &self.program_io.memory_layout,
            &self.one_hot_params,
            self.proof.trace_length,
            &self.proof.rw_config,
            &self.opening_accumulator,
        );

        let ram_output_check = OutputSumcheckVerifier::new(
            self.proof.ram_K,
            &self.program_io,
            transcript,
            self.proof.trace_length,
            &self.proof.rw_config,
        );

        let instances: Vec<&dyn SumcheckInstanceVerifier<F, VerifierOpeningAccumulator<F>>> = vec![
            &ram_read_write_checking,
            &spartan_product_virtual_remainder,
            &instruction_claim_reduction,
            &ram_raf_evaluation,
            &ram_output_check,
        ];

        let (batching_coefficients, r_stage2, zk_sumcheck_readback) =
            BatchedSumcheck::verify::<F, C>(
                self.proof.zk_mode,
                instances.clone(),
                &mut self.opening_accumulator,
                transcript,
            )?;

        #[cfg(feature = "zk")]
        {
            let regular_oc_ids = self.opening_accumulator.take_pending_claim_ids();

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
            let uniskip_output_constraint = uni_skip_params.output_claim_constraint();
            let uniskip_output_constraint_challenge_values =
                uni_skip_params.output_constraint_challenge_values(&[uni_skip_challenge]);

            let stage_result = StageVerifyResult::with_uniskip(
                r_stage2,
                batched_output_constraint,
                output_constraint_challenge_values,
                batched_input_constraint,
                input_constraint_challenge_values,
                uniskip_input_constraint,
                uniskip_input_constraint_challenge_values,
                uniskip_output_constraint,
                uniskip_output_constraint_challenge_values,
                vec![uniskip_oc_ids, regular_oc_ids],
            );

            self.zk_uniskip_readback[1] = zk_uniskip_readback;
            self.zk_sumcheck_readback[1] = zk_sumcheck_readback;

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

    #[cfg_attr(not(feature = "zk"), allow(unused_variables))]
    fn verify_stage3(
        &mut self,
        transcript: &mut impl VerifierFs<F>,
    ) -> Result<StageVerifyResult<F>, ProofVerifyError> {
        let spartan_shift = ShiftSumcheckVerifier::new(
            self.proof.trace_length.log_2(),
            &self.opening_accumulator,
            transcript,
        );
        let spartan_instruction_input =
            InstructionInputSumcheckVerifier::new(&self.opening_accumulator, transcript);
        let spartan_registers_claim_reduction = RegistersClaimReductionSumcheckVerifier::new(
            self.proof.trace_length,
            &self.opening_accumulator,
            transcript,
        );

        let instances: Vec<&dyn SumcheckInstanceVerifier<F, VerifierOpeningAccumulator<F>>> = vec![
            &spartan_shift,
            &spartan_instruction_input,
            &spartan_registers_claim_reduction,
        ];

        let (batching_coefficients, r_stage3, zk_sumcheck_readback) =
            BatchedSumcheck::verify::<F, C>(
                self.proof.zk_mode,
                instances.clone(),
                &mut self.opening_accumulator,
                transcript,
            )?;

        #[cfg(feature = "zk")]
        {
            let regular_oc_ids = self.opening_accumulator.take_pending_claim_ids();
            let batched_output_constraint = batch_output_constraints(&instances);
            let batched_input_constraint = batch_input_constraints(&instances);
            let max_num_rounds = instances.iter().map(|i| i.num_rounds()).max().unwrap();
            let mut output_constraint_challenge_values: Vec<F> = batching_coefficients.clone();
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
            let stage_result = StageVerifyResult::new(
                r_stage3,
                batched_output_constraint,
                output_constraint_challenge_values,
                batched_input_constraint,
                input_constraint_challenge_values,
                vec![regular_oc_ids],
            );
            self.zk_sumcheck_readback[2] = zk_sumcheck_readback;
            Ok(stage_result)
        }
        #[cfg(not(feature = "zk"))]
        Ok(StageVerifyResult {
            challenges: r_stage3,
        })
    }

    #[cfg_attr(not(feature = "zk"), allow(unused_variables))]
    fn verify_stage4(
        &mut self,
        transcript: &mut impl VerifierFs<F>,
    ) -> Result<StageVerifyResult<F>, ProofVerifyError> {
        let registers_read_write_checking = RegistersReadWriteCheckingVerifier::new(
            self.proof.trace_length,
            &self.opening_accumulator,
            transcript,
            &self.proof.rw_config,
        );
        verifier_accumulate_advice::<F, VerifierOpeningAccumulator<F>>(
            self.proof.ram_K,
            &self.program_io,
            self.untrusted_advice_commitment.is_some(),
            self.trusted_advice_commitment.is_some(),
            &mut self.opening_accumulator,
        );
        if self.preprocessing.shared.program.is_committed() {
            verifier_accumulate_program_image::<F>(self.proof.ram_K, &mut self.opening_accumulator);
        }
        // Domain-separate the batching challenge.
        let ram_val_check_gamma: F = transcript.challenge_field();
        let initial_ram_state = if self.preprocessing.shared.program.is_full() {
            crate::zkvm::ram::gen_ram_initial_memory_state::<F>(
                self.proof.ram_K,
                &self.preprocessing.shared.program.as_full()?.ram,
                &self.program_io,
            )
        } else {
            vec![0u64; self.proof.ram_K]
        };
        let ram_preprocessing = if self.preprocessing.shared.program.is_full() {
            self.preprocessing.shared.program.as_full()?.ram.clone()
        } else {
            crate::zkvm::ram::RAMPreprocessing {
                min_bytecode_address: self.preprocessing.shared.program_meta.min_bytecode_address,
                bytecode_words: vec![
                    0;
                    self.preprocessing
                        .shared
                        .program_meta
                        .program_image_len_words
                ],
            }
        };
        let ram_val_check = RamValCheckSumcheckVerifier::new(
            &initial_ram_state,
            &self.program_io,
            &ram_preprocessing,
            self.proof.trace_length,
            self.proof.ram_K,
            &self.proof.rw_config,
            ram_val_check_gamma,
            &self.opening_accumulator,
            self.preprocessing.shared.program.is_committed(),
        );

        let instances: Vec<&dyn SumcheckInstanceVerifier<F, VerifierOpeningAccumulator<F>>> =
            vec![&registers_read_write_checking, &ram_val_check];

        let (batching_coefficients, r_stage4, zk_sumcheck_readback) =
            BatchedSumcheck::verify::<F, C>(
                self.proof.zk_mode,
                instances.clone(),
                &mut self.opening_accumulator,
                transcript,
            )?;

        #[cfg(feature = "zk")]
        {
            let regular_oc_ids = self.opening_accumulator.take_pending_claim_ids();
            let batched_output_constraint = batch_output_constraints(&instances);
            let batched_input_constraint = batch_input_constraints(&instances);
            let max_num_rounds = instances.iter().map(|i| i.num_rounds()).max().unwrap();
            let mut output_constraint_challenge_values: Vec<F> = batching_coefficients.clone();
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
            let stage_result = StageVerifyResult::new(
                r_stage4,
                batched_output_constraint,
                output_constraint_challenge_values,
                batched_input_constraint,
                input_constraint_challenge_values,
                vec![regular_oc_ids],
            );
            self.zk_sumcheck_readback[3] = zk_sumcheck_readback;
            Ok(stage_result)
        }
        #[cfg(not(feature = "zk"))]
        Ok(StageVerifyResult {
            challenges: r_stage4,
        })
    }

    #[cfg_attr(not(feature = "zk"), allow(unused_variables))]
    fn verify_stage5(
        &mut self,
        transcript: &mut impl VerifierFs<F>,
    ) -> Result<StageVerifyResult<F>, ProofVerifyError> {
        let n_cycle_vars = self.proof.trace_length.log_2();

        let lookups_read_raf = InstructionReadRafSumcheckVerifier::new(
            n_cycle_vars,
            &self.one_hot_params,
            &self.opening_accumulator,
            transcript,
        );
        let ram_ra_reduction = RamRaClaimReductionSumcheckVerifier::new(
            self.proof.trace_length,
            &self.one_hot_params,
            &self.opening_accumulator,
            transcript,
        );
        let registers_val_evaluation =
            RegistersValEvaluationSumcheckVerifier::new(&self.opening_accumulator);

        let instances: Vec<&dyn SumcheckInstanceVerifier<F, VerifierOpeningAccumulator<F>>> = vec![
            &lookups_read_raf,
            &ram_ra_reduction,
            &registers_val_evaluation,
        ];

        let (batching_coefficients, r_stage5, zk_sumcheck_readback) =
            BatchedSumcheck::verify::<F, C>(
                self.proof.zk_mode,
                instances.clone(),
                &mut self.opening_accumulator,
                transcript,
            )?;

        #[cfg(feature = "zk")]
        {
            let regular_oc_ids = self.opening_accumulator.take_pending_claim_ids();
            let batched_output_constraint = batch_output_constraints(&instances);
            let batched_input_constraint = batch_input_constraints(&instances);
            let max_num_rounds = instances.iter().map(|i| i.num_rounds()).max().unwrap();
            let mut output_constraint_challenge_values: Vec<F> = batching_coefficients.clone();
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
            let stage_result = StageVerifyResult::new(
                r_stage5,
                batched_output_constraint,
                output_constraint_challenge_values,
                batched_input_constraint,
                input_constraint_challenge_values,
                vec![regular_oc_ids],
            );
            self.zk_sumcheck_readback[4] = zk_sumcheck_readback;
            Ok(stage_result)
        }
        #[cfg(not(feature = "zk"))]
        Ok(StageVerifyResult {
            challenges: r_stage5,
        })
    }

    #[cfg_attr(not(feature = "zk"), allow(unused_variables))]
    fn verify_stage6(
        &mut self,
        transcript: &mut impl VerifierFs<F>,
    ) -> Result<(StageVerifyResult<F>, StageVerifyResult<F>), ProofVerifyError> {
        let _ = DoryGlobals::initialize_main_with_log_embedding(
            self.one_hot_params.k_chunk,
            self.proof.trace_length,
            self.main_total_vars(),
            Some(self.proof.dory_layout),
        );
        let (bytecode_read_raf_params, booleanity_params, stage6a_result) =
            self.verify_stage6a(transcript)?;
        let stage6b_result =
            self.verify_stage6b(transcript, bytecode_read_raf_params, booleanity_params)?;
        Ok((stage6a_result, stage6b_result))
    }

    #[cfg_attr(not(feature = "zk"), allow(unused_variables))]
    fn verify_stage6a(
        &mut self,
        transcript: &mut impl VerifierFs<F>,
    ) -> Result<Stage6aVerifyResult<F>, ProofVerifyError> {
        let n_cycle_vars = self.proof.trace_length.log_2();
        let bytecode_read_raf = BytecodeReadRafAddressSumcheckVerifier::new(
            &self.preprocessing.shared.program,
            n_cycle_vars,
            &self.one_hot_params,
            &self.opening_accumulator,
            transcript,
        );
        let booleanity = BooleanityAddressSumcheckVerifier::new(BooleanitySumcheckParams::new(
            n_cycle_vars,
            &self.one_hot_params,
            &self.opening_accumulator,
            transcript,
        ));
        let instances: Vec<&dyn SumcheckInstanceVerifier<F, VerifierOpeningAccumulator<F>>> =
            vec![&bytecode_read_raf, &booleanity];
        let (_batching_coefficients, r_stage6a, zk_sumcheck_readback) =
            BatchedSumcheck::verify::<F, C>(
                self.proof.zk_mode,
                instances.clone(),
                &mut self.opening_accumulator,
                transcript,
            )
            .inspect_err(|err| tracing::error!("Stage 6a: {err}"))?;
        #[cfg(feature = "zk")]
        {
            let regular_oc_ids = self.opening_accumulator.take_pending_claim_ids();
            let batched_output_constraint = batch_output_constraints(&instances);
            let batched_input_constraint = batch_input_constraints(&instances);
            let max_num_rounds = instances.iter().map(|i| i.num_rounds()).max().unwrap();
            let mut output_constraint_challenge_values: Vec<F> = _batching_coefficients.clone();
            let mut input_constraint_challenge_values: Vec<F> =
                scale_batching_coefficients(&_batching_coefficients, &instances);
            for instance in &instances {
                let num_rounds = instance.num_rounds();
                let offset = instance.round_offset(max_num_rounds);
                let r_slice = &r_stage6a[offset..offset + num_rounds];
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
            let stage_result = StageVerifyResult::new(
                r_stage6a,
                batched_output_constraint,
                output_constraint_challenge_values,
                batched_input_constraint,
                input_constraint_challenge_values,
                vec![regular_oc_ids],
            );
            self.zk_sumcheck_readback[5] = zk_sumcheck_readback;
            Ok((
                bytecode_read_raf.into_params(),
                booleanity.into_params(),
                stage_result,
            ))
        }
        #[cfg(not(feature = "zk"))]
        Ok((
            bytecode_read_raf.into_params(),
            booleanity.into_params(),
            StageVerifyResult {
                challenges: r_stage6a,
            },
        ))
    }

    #[cfg_attr(not(feature = "zk"), allow(unused_variables))]
    fn verify_stage6b(
        &mut self,
        transcript: &mut impl VerifierFs<F>,
        bytecode_read_raf_params: BytecodeReadRafSumcheckParams<F>,
        booleanity_params: BooleanitySumcheckParams<F>,
    ) -> Result<StageVerifyResult<F>, ProofVerifyError> {
        let ram_hamming_booleanity =
            HammingBooleanitySumcheckVerifier::new(&self.opening_accumulator);
        let booleanity =
            BooleanityCycleSumcheckVerifier::new(booleanity_params, &self.opening_accumulator);
        let ram_ra_virtual = RamRaVirtualSumcheckVerifier::new(
            self.proof.trace_length,
            &self.one_hot_params,
            &self.opening_accumulator,
            transcript,
        );
        let lookups_ra_virtual = LookupsRaSumcheckVerifier::new(
            &self.one_hot_params,
            &self.opening_accumulator,
            transcript,
        );
        let inc_reduction = IncClaimReductionSumcheckVerifier::new(
            self.proof.trace_length,
            &self.opening_accumulator,
            transcript,
        );

        let main_total_vars = self.proof.trace_length.log_2() + self.one_hot_params.log_k_chunk;
        let precommitted_candidates = self.preprocessing.shared.precommitted_candidate_total_vars(
            self.preprocessing.shared.program.is_committed(),
            self.trusted_advice_commitment.is_some(),
            self.untrusted_advice_commitment.is_some(),
        );
        let precommitted_scheduling_reference =
            PrecommittedClaimReduction::<F>::scheduling_reference(
                main_total_vars,
                &precommitted_candidates,
            );

        // Advice claim reduction (Phase 1 in Stage 6b): trusted and untrusted are separate instances.
        if self.trusted_advice_commitment.is_some() {
            self.advice_reduction_verifier_trusted = Some(AdviceClaimReductionVerifier::new(
                AdviceKind::Trusted,
                self.program_io.memory_layout.max_trusted_advice_size as usize,
                precommitted_scheduling_reference,
                &self.opening_accumulator,
            ));
        }
        if self.untrusted_advice_commitment.is_some() {
            self.advice_reduction_verifier_untrusted = Some(AdviceClaimReductionVerifier::new(
                AdviceKind::Untrusted,
                self.program_io.memory_layout.max_untrusted_advice_size as usize,
                precommitted_scheduling_reference,
                &self.opening_accumulator,
            ));
        }
        if self.preprocessing.shared.program.is_committed() {
            let bytecode_chunk_count = self.preprocessing.shared.bytecode_chunk_count;
            let bytecode_reduction_params = BytecodeClaimReductionParams::new(
                bytecode_read_raf_params.stage_gammas(),
                self.preprocessing.shared.bytecode_size(),
                bytecode_chunk_count,
                precommitted_scheduling_reference,
                &self.opening_accumulator,
                transcript,
            );
            self.bytecode_reduction_verifier = Some(BytecodeClaimReductionVerifier::new(
                bytecode_reduction_params,
            ));

            let padded_len_words = self
                .preprocessing
                .shared
                .program_meta
                .committed_program_image_num_words(&self.program_io.memory_layout);
            let program_image_reduction_params = ProgramImageClaimReductionParams::new(
                &self.program_io,
                self.preprocessing.shared.program_meta.min_bytecode_address,
                padded_len_words,
                self.proof.ram_K,
                precommitted_scheduling_reference,
                &self.opening_accumulator,
            );
            self.program_image_reduction_verifier = Some(ProgramImageClaimReductionVerifier::new(
                program_image_reduction_params,
            ));
        }

        let bytecode_read_raf = BytecodeReadRafCycleSumcheckVerifier::new(
            bytecode_read_raf_params,
            &self.opening_accumulator,
        );

        let mut instances: Vec<&dyn SumcheckInstanceVerifier<F, VerifierOpeningAccumulator<F>>> = vec![
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
        if let Some(ref reduction) = self.bytecode_reduction_verifier {
            instances.push(reduction);
        }
        if let Some(ref reduction) = self.program_image_reduction_verifier {
            instances.push(reduction);
        }

        let (batching_coefficients, r_stage6b, zk_sumcheck_readback) =
            BatchedSumcheck::verify::<F, C>(
                self.proof.zk_mode,
                instances.clone(),
                &mut self.opening_accumulator,
                transcript,
            )
            .inspect_err(|err| tracing::error!("Stage 6b: {err}"))?;

        #[cfg(feature = "zk")]
        {
            let regular_oc_ids = self.opening_accumulator.take_pending_claim_ids();
            let batched_output_constraint = batch_output_constraints(&instances);
            let batched_input_constraint = batch_input_constraints(&instances);
            let max_num_rounds = instances.iter().map(|i| i.num_rounds()).max().unwrap();
            let mut output_constraint_challenge_values: Vec<F> = batching_coefficients.clone();
            let mut input_constraint_challenge_values: Vec<F> =
                scale_batching_coefficients(&batching_coefficients, &instances);
            for instance in &instances {
                let num_rounds = instance.num_rounds();
                let offset = instance.round_offset(max_num_rounds);
                let r_slice = &r_stage6b[offset..offset + num_rounds];
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
            let stage_result = StageVerifyResult::new(
                r_stage6b,
                batched_output_constraint,
                output_constraint_challenge_values,
                batched_input_constraint,
                input_constraint_challenge_values,
                vec![regular_oc_ids],
            );
            self.zk_sumcheck_readback[6] = zk_sumcheck_readback;
            Ok(stage_result)
        }
        #[cfg(not(feature = "zk"))]
        Ok(StageVerifyResult {
            challenges: r_stage6b,
        })
    }

    #[cfg(feature = "zk")]
    #[allow(clippy::too_many_arguments)]
    fn verify_blindfold(
        &mut self,
        transcript: &mut impl VerifierFs<F>,
        sumcheck_challenges: &[Vec<F::Challenge>; 8],
        uniskip_challenges: [F::Challenge; 2],
        stage_output_constraints: &[Option<OutputClaimConstraint>; 8],
        output_constraint_challenge_values: &[Vec<F>; 8],
        stage_input_constraints: &[InputClaimConstraint; 8],
        input_constraint_challenge_values: &[Vec<F>; 8],
        // For stages 0-1: batched input constraint for regular rounds (different from uni-skip)
        stage1_batched_input: &InputClaimConstraint,
        stage2_batched_input: &InputClaimConstraint,
        stage1_batched_input_values: &[F],
        stage2_batched_input_values: &[F],
        uniskip_output_constraints: &[Option<OutputClaimConstraint>; 2],
        uniskip_output_challenge_values: &[Vec<F>; 2],
        stage8_data: &Stage8VerifyData<F>,
        oc_blocks: Vec<Vec<OpeningId>>,
    ) -> Result<(), ProofVerifyError> {
        // Build stage configurations including uni-skip rounds.
        // Uni-skip rounds are the first round of stages 1 and 2 (indices 0 and 1).
        //
        // The ZK commitments/degrees that used to live in the (now data-free) proof structs
        // were read back from the NARG during each stage's verification; we feed BlindFold
        // from those caches (`zk_sumcheck_readback` / `zk_uniskip_readback`) instead.
        let zk_sumcheck_readback: Vec<&ZkSumcheckReadback<C>> = self
            .zk_sumcheck_readback
            .iter()
            .map(|r| {
                r.as_ref()
                    .ok_or(ProofVerifyError::SumcheckVerificationError)
            })
            .collect::<Result<_, _>>()?;
        let zk_uniskip_readback: Vec<&ZkUniSkipReadback<C>> = self
            .zk_uniskip_readback
            .iter()
            .map(|r| r.as_ref().ok_or(ProofVerifyError::UniSkipVerificationError))
            .collect::<Result<_, _>>()?;

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
        let mut regular_first_round_indices: Vec<usize> = Vec::new(); // 8 elements for all stages
        let mut last_round_indices: Vec<usize> = Vec::new();

        for (stage_idx, sumcheck_readback) in zk_sumcheck_readback.iter().enumerate() {
            // For stages 0 and 1 (Jolt stages 1 and 2), add uni-skip config first
            if stage_idx < 2 {
                let poly_degree = zk_uniskip_readback[stage_idx].poly_degree;

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

            // The per-round degrees come from the NARG read-back (one per round); the round
            // count is derived from its length, not from any (now data-free) proof struct.
            let round_poly_degrees = sumcheck_readback.poly_degrees.clone();
            stage_configs.push(StageConfig::new_chain_with_round_degrees(
                round_poly_degrees,
            ));

            // Record the last round index for output constraint
            last_round_indices.push(stage_configs.len() - 1);
        }

        // Add final_output configurations using the batched constraints from verifier instances
        for (stage_idx, constraint) in stage_output_constraints.iter().enumerate() {
            if let Some(batched) = constraint {
                let last_round_idx = last_round_indices[stage_idx];
                stage_configs[last_round_idx].final_output =
                    Some(ClaimBindingConfig::with_constraint(batched.clone()));
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
                Some(ClaimBindingConfig::with_constraint(constraint.clone()));
        }

        // Add final_output configurations for uni-skip stages (stages 0-1)
        for (i, constraint) in uniskip_output_constraints.iter().enumerate() {
            if let Some(oc) = constraint {
                let idx = uniskip_indices[i];
                stage_configs[idx].final_output =
                    Some(ClaimBindingConfig::with_constraint(oc.clone()));
            }
        }

        // Add initial_input configurations for regular first rounds (all 8 stages)
        // These use the batched input constraints from the stage results
        let regular_constraints = [
            stage1_batched_input.clone(),       // Stage 0 regular
            stage2_batched_input.clone(),       // Stage 1 regular
            stage_input_constraints[2].clone(), // Stage 2
            stage_input_constraints[3].clone(), // Stage 3
            stage_input_constraints[4].clone(), // Stage 4
            stage_input_constraints[5].clone(), // Stage 5 (6a)
            stage_input_constraints[6].clone(), // Stage 6 (6b)
            stage_input_constraints[7].clone(), // Stage 7
        ];
        for (i, constraint) in regular_constraints.iter().enumerate() {
            let idx = regular_first_round_indices[i];
            stage_configs[idx].initial_input =
                Some(ClaimBindingConfig::with_constraint(constraint.clone()));
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

        let all_input_challenge_values: [&[F]; 10] = [
            &input_constraint_challenge_values[0],
            stage1_batched_input_values,
            &input_constraint_challenge_values[1],
            stage2_batched_input_values,
            &input_constraint_challenge_values[2],
            &input_constraint_challenge_values[3],
            &input_constraint_challenge_values[4],
            &input_constraint_challenge_values[5],
            &input_constraint_challenge_values[6],
            &input_constraint_challenge_values[7],
        ];
        let mut baked_input_challenges: Vec<F> = Vec::new();
        for expected_values in all_input_challenge_values.iter() {
            baked_input_challenges.extend_from_slice(expected_values);
        }

        let mut baked_output_challenges: Vec<F> = Vec::new();
        for (stage_idx, expected_values) in output_constraint_challenge_values.iter().enumerate() {
            if stage_idx < 2 && uniskip_output_constraints[stage_idx].is_some() {
                baked_output_challenges
                    .extend_from_slice(&uniskip_output_challenge_values[stage_idx]);
            }
            baked_output_challenges.extend_from_slice(expected_values);
        }

        // Count chains — ConstantInitialClaim paths index into this vector.
        // Only chain 0 (outer uni-skip, initial_claim = zero) uses ConstantInitialClaim;
        // all others use InitialClaimVar and won't read from here.
        let num_chains = stage_configs
            .iter()
            .enumerate()
            .filter(|(i, c)| *i == 0 || c.starts_new_chain)
            .count();
        let baked = BakedPublicInputs {
            challenges: baked_challenges,
            initial_claims: vec![F::zero(); num_chains],
            batching_coefficients: Vec::new(),
            output_constraint_challenges: baked_output_challenges,
            input_constraint_challenges: baked_input_challenges,
            extra_constraint_challenges: stage8_data.constraint_coeffs.clone(),
        };

        // Assemble the BlindFold input from the NARG read-back caches, in the same stage
        // order as before (uni-skip first for stages 0–1, then the stage's round commitments).
        let mut round_commitments: Vec<C::G1> = Vec::new();
        let mut oc_row_commitments: Vec<C::G1> = Vec::new();
        for (stage_idx, sumcheck_readback) in zk_sumcheck_readback.iter().enumerate() {
            if stage_idx < 2 {
                let zk_uniskip = zk_uniskip_readback[stage_idx];
                round_commitments.push(zk_uniskip.commitment);
                oc_row_commitments.extend_from_slice(&zk_uniskip.output_claims_commitments);
            }
            round_commitments.extend(sumcheck_readback.round_commitments.iter().cloned());
            oc_row_commitments.extend_from_slice(&sumcheck_readback.output_claims_commitments);
        }

        let builder = VerifierR1CSBuilder::new_with_extra(
            &stage_configs,
            &extra_constraints,
            &baked,
            oc_blocks,
            self.opening_accumulator.aliases.clone(),
        );
        let r1cs = builder.build();

        let eval_commitment = PCS::eval_commitment(&self.proof.joint_opening_proof)
            .ok_or(ProofVerifyError::InvalidOpeningProof)?;
        let eval_commitments = vec![eval_commitment];

        let verifier_input = BlindFoldVerifierInput {
            round_commitments,
            output_claims_row_commitments: oc_row_commitments,
            eval_commitments,
        };

        let pedersen_generator_count = pedersen_generator_count_for_r1cs(&r1cs);
        let pedersen_generators = self
            .preprocessing
            .pedersen_generators(pedersen_generator_count);
        let eval_commitment_gens =
            PCS::eval_commitment_gens_verifier(&self.preprocessing.generators);
        let verifier =
            BlindFoldVerifier::<_, _>::new(&pedersen_generators, &r1cs, eval_commitment_gens);

        verifier
            .verify(&verifier_input, transcript)
            .map_err(|e| ProofVerifyError::BlindFoldError(format!("{e:?}")))?;

        tracing::debug!(
            "BlindFold verification passed: {} R1CS constraints",
            r1cs.num_constraints
        );

        Ok(())
    }

    #[cfg_attr(not(feature = "zk"), allow(unused_variables))]
    fn verify_stage7(
        &mut self,
        transcript: &mut impl VerifierFs<F>,
    ) -> Result<StageVerifyResult<F>, ProofVerifyError> {
        // Create verifier for HammingWeightClaimReduction
        // (r_cycle and r_addr_bool are extracted from Booleanity opening internally)
        let hw_verifier = HammingWeightClaimReductionVerifier::new(
            &self.one_hot_params,
            &self.opening_accumulator,
            transcript,
        );

        let mut instances: Vec<&dyn SumcheckInstanceVerifier<F, VerifierOpeningAccumulator<F>>> =
            vec![&hw_verifier];
        if let Some(advice_reduction_verifier_trusted) =
            self.advice_reduction_verifier_trusted.as_mut()
        {
            if advice_reduction_verifier_trusted
                .params
                .precommitted
                .num_address_phase_rounds()
                > 0
            {
                // Transition phase
                advice_reduction_verifier_trusted
                    .params
                    .transition_to_address_phase(&self.opening_accumulator);
                instances.push(advice_reduction_verifier_trusted);
            }
        }
        if let Some(advice_reduction_verifier_untrusted) =
            self.advice_reduction_verifier_untrusted.as_mut()
        {
            if advice_reduction_verifier_untrusted
                .params
                .precommitted
                .num_address_phase_rounds()
                > 0
            {
                // Transition phase
                advice_reduction_verifier_untrusted
                    .params
                    .transition_to_address_phase(&self.opening_accumulator);
                instances.push(advice_reduction_verifier_untrusted);
            }
        }
        if let Some(bytecode_reduction_verifier) = self.bytecode_reduction_verifier.as_mut() {
            if bytecode_reduction_verifier
                .params
                .precommitted
                .num_address_phase_rounds()
                > 0
            {
                bytecode_reduction_verifier
                    .params
                    .transition_to_address_phase(&self.opening_accumulator);
                instances.push(bytecode_reduction_verifier);
            }
        }
        if let Some(program_image_reduction_verifier) =
            self.program_image_reduction_verifier.as_mut()
        {
            if program_image_reduction_verifier
                .params
                .precommitted
                .num_address_phase_rounds()
                > 0
            {
                program_image_reduction_verifier
                    .params
                    .transition_to_address_phase(&self.opening_accumulator);
                instances.push(program_image_reduction_verifier);
            }
        }

        let (batching_coefficients, r_stage7, zk_sumcheck_readback) =
            BatchedSumcheck::verify::<F, C>(
                self.proof.zk_mode,
                instances.clone(),
                &mut self.opening_accumulator,
                transcript,
            )?;

        #[cfg(feature = "zk")]
        {
            let regular_oc_ids = self.opening_accumulator.take_pending_claim_ids();
            let batched_output_constraint = batch_output_constraints(&instances);
            let batched_input_constraint = batch_input_constraints(&instances);
            let max_num_rounds = instances.iter().map(|i| i.num_rounds()).max().unwrap();
            let mut output_constraint_challenge_values: Vec<F> = batching_coefficients.clone();
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
            let stage_result = StageVerifyResult::new(
                r_stage7,
                batched_output_constraint,
                output_constraint_challenge_values,
                batched_input_constraint,
                input_constraint_challenge_values,
                vec![regular_oc_ids],
            );
            self.zk_sumcheck_readback[7] = zk_sumcheck_readback;
            Ok(stage_result)
        }
        #[cfg(not(feature = "zk"))]
        Ok(StageVerifyResult {
            challenges: r_stage7,
        })
    }

    fn verify_stage8(
        &mut self,
        transcript: &mut impl VerifierFs<F>,
    ) -> Result<Stage8VerifyData<F>, ProofVerifyError> {
        let native_main_vars = self.proof.trace_length.log_2() + self.one_hot_params.log_k_chunk;
        let opening_point = compute_final_opening_point(
            &self.opening_accumulator,
            native_main_vars,
            self.one_hot_params.log_k_chunk,
            self.proof.dory_layout,
            if self.preprocessing.shared.program.is_committed() {
                ProgramMode::Committed
            } else {
                ProgramMode::Full
            },
            self.preprocessing.shared.bytecode_chunk_count,
        )?;

        // 1. Collect all (polynomial, claim) pairs
        let mut polynomial_claims = Vec::new();
        let mut scaling_factors = Vec::new();

        // Dense polynomials: RamInc and RdInc (from IncClaimReduction in Stage 6)
        let (ram_inc_point, ram_inc_claim) =
            self.opening_accumulator.get_committed_polynomial_opening(
                CommittedPolynomial::RamInc,
                SumcheckId::IncClaimReduction,
            );
        let (rd_inc_point, rd_inc_claim) =
            self.opening_accumulator.get_committed_polynomial_opening(
                CommittedPolynomial::RdInc,
                SumcheckId::IncClaimReduction,
            );
        let ram_inc_lagrange = compute_lagrange_factor::<F>(&opening_point.r, &ram_inc_point.r);
        let rd_inc_lagrange = compute_lagrange_factor::<F>(&opening_point.r, &rd_inc_point.r);
        polynomial_claims.push((
            CommittedPolynomial::RamInc,
            ram_inc_claim * ram_inc_lagrange,
        ));
        scaling_factors.push(ram_inc_lagrange);
        polynomial_claims.push((CommittedPolynomial::RdInc, rd_inc_claim * rd_inc_lagrange));
        scaling_factors.push(rd_inc_lagrange);

        // Sparse polynomials: all RA polys (from HammingWeightClaimReduction)
        for i in 0..self.one_hot_params.instruction_d {
            let (ra_point, claim) = self.opening_accumulator.get_committed_polynomial_opening(
                CommittedPolynomial::InstructionRa(i),
                SumcheckId::HammingWeightClaimReduction,
            );
            let lagrange = compute_lagrange_factor::<F>(&opening_point.r, &ra_point.r);
            polynomial_claims.push((CommittedPolynomial::InstructionRa(i), claim * lagrange));
            scaling_factors.push(lagrange);
        }
        for i in 0..self.one_hot_params.bytecode_d {
            let (ra_point, claim) = self.opening_accumulator.get_committed_polynomial_opening(
                CommittedPolynomial::BytecodeRa(i),
                SumcheckId::HammingWeightClaimReduction,
            );
            let lagrange = compute_lagrange_factor::<F>(&opening_point.r, &ra_point.r);
            polynomial_claims.push((CommittedPolynomial::BytecodeRa(i), claim * lagrange));
            scaling_factors.push(lagrange);
        }
        for i in 0..self.one_hot_params.ram_d {
            let (ra_point, claim) = self.opening_accumulator.get_committed_polynomial_opening(
                CommittedPolynomial::RamRa(i),
                SumcheckId::HammingWeightClaimReduction,
            );
            let lagrange = compute_lagrange_factor::<F>(&opening_point.r, &ra_point.r);
            polynomial_claims.push((CommittedPolynomial::RamRa(i), claim * lagrange));
            scaling_factors.push(lagrange);
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
            let lagrange_factor = compute_lagrange_factor::<F>(&opening_point.r, &advice_point.r);
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
            let lagrange_factor = compute_lagrange_factor::<F>(&opening_point.r, &advice_point.r);
            polynomial_claims.push((
                CommittedPolynomial::UntrustedAdvice,
                advice_claim * lagrange_factor,
            ));
            scaling_factors.push(lagrange_factor);
            include_untrusted_advice = true;
        }

        if self.preprocessing.shared.program.is_committed() {
            let chunk_count = self.preprocessing.shared.bytecode_chunk_count;
            for chunk_idx in 0..chunk_count {
                let (chunk_point, chunk_claim) =
                    self.opening_accumulator.get_committed_polynomial_opening(
                        CommittedPolynomial::BytecodeChunk(chunk_idx),
                        SumcheckId::BytecodeClaimReduction,
                    );
                let lagrange_factor =
                    compute_lagrange_factor::<F>(&opening_point.r, &chunk_point.r);
                polynomial_claims.push((
                    CommittedPolynomial::BytecodeChunk(chunk_idx),
                    chunk_claim * lagrange_factor,
                ));
                scaling_factors.push(lagrange_factor);
            }
        }
        if self.preprocessing.shared.program.is_committed() {
            let (program_point, program_claim) =
                self.opening_accumulator.get_committed_polynomial_opening(
                    CommittedPolynomial::ProgramImageInit,
                    SumcheckId::ProgramImageClaimReduction,
                );
            let lagrange_factor = compute_lagrange_factor::<F>(&opening_point.r, &program_point.r);
            polynomial_claims.push((
                CommittedPolynomial::ProgramImageInit,
                program_claim * lagrange_factor,
            ));
            scaling_factors.push(lagrange_factor);
        }

        // 2. Sample gamma and compute powers for RLC
        let claims: Vec<F> = polynomial_claims.iter().map(|(_, c)| *c).collect();
        // In non-ZK mode, absorb claims before sampling gamma for Fiat-Shamir binding.
        // In ZK mode, claims are secret; binding comes from BlindFold constraints instead.
        #[cfg(not(feature = "zk"))]
        transcript.absorb(&claims);
        let gamma_powers: Vec<F> = transcript.challenge_powers(polynomial_claims.len());
        let constraint_coeffs: Vec<F> = gamma_powers
            .iter()
            .zip(&scaling_factors)
            .map(|(gamma, scale)| *gamma * *scale)
            .collect();

        let opening_ids = stage8_opening_ids(
            &self.one_hot_params,
            include_trusted_advice,
            include_untrusted_advice,
            if self.preprocessing.shared.program.is_committed() {
                ProgramMode::Committed
            } else {
                ProgramMode::Full
            },
            self.preprocessing.shared.bytecode_chunk_count,
        );
        let joint_claim: F = gamma_powers
            .iter()
            .zip(claims.iter())
            .map(|(gamma, claim)| *gamma * claim)
            .sum();

        // Build state for computing joint commitment/claim
        let state = DoryOpeningState {
            opening_point: opening_point.r.clone(),
            gamma_powers: gamma_powers.clone(),
            polynomial_claims,
        };

        // Build commitments map
        let mut commitments_map = HashMap::new();
        let expected_polynomials = all_committed_polynomials(&self.one_hot_params);
        if expected_polynomials.len() != self.commitments.len() {
            return Err(ProofVerifyError::InvalidInputLength(
                expected_polynomials.len(),
                self.commitments.len(),
            ));
        }
        for (polynomial, commitment) in expected_polynomials.into_iter().zip(&self.commitments) {
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
        if let Some(ref commitment) = self.untrusted_advice_commitment {
            if state
                .polynomial_claims
                .iter()
                .any(|(p, _)| *p == CommittedPolynomial::UntrustedAdvice)
            {
                commitments_map.insert(CommittedPolynomial::UntrustedAdvice, commitment.clone());
            }
        }
        if let Some(trusted_bytecode) = self.preprocessing.shared.program.bytecode_commitments() {
            for (chunk_idx, commitment) in trusted_bytecode.commitments.iter().enumerate() {
                if state
                    .polynomial_claims
                    .iter()
                    .any(|(p, _)| *p == CommittedPolynomial::BytecodeChunk(chunk_idx))
                {
                    commitments_map.insert(
                        CommittedPolynomial::BytecodeChunk(chunk_idx),
                        commitment.clone(),
                    );
                }
            }
        }
        if let Ok(trusted_program) = self.preprocessing.shared.program.as_committed() {
            if state
                .polynomial_claims
                .iter()
                .any(|(p, _)| *p == CommittedPolynomial::ProgramImageInit)
            {
                commitments_map.insert(
                    CommittedPolynomial::ProgramImageInit,
                    trusted_program.program_image_commitment.clone(),
                );
            }
        }

        let joint_commitment = self.compute_joint_commitment(&mut commitments_map, &state)?;

        let zk_mode = self.opening_accumulator.zk_mode;
        if zk_mode {
            PCS::verify(
                &self.proof.joint_opening_proof,
                &self.preprocessing.generators,
                transcript,
                &opening_point.r,
                &F::zero(),
                &joint_commitment,
            )?;

            #[cfg(feature = "zk")]
            {
                let y_com: C::G1 = PCS::eval_commitment(&self.proof.joint_opening_proof)
                    .ok_or(ProofVerifyError::InvalidOpeningProof)?;
                bind_opening_inputs_zk::<F, C, _>(transcript, &opening_point.r, &y_com);
            }
            #[cfg(not(feature = "zk"))]
            {
                return Err(ProofVerifyError::ZkFeatureRequired);
            }
        } else {
            PCS::verify(
                &self.proof.joint_opening_proof,
                &self.preprocessing.generators,
                transcript,
                &opening_point.r,
                &joint_claim,
                &joint_commitment,
            )?;

            bind_opening_inputs::<F, _>(transcript, &opening_point.r, &joint_claim);
        }

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
                commitment_map.remove(&k).map(|c| (v, c)).ok_or_else(|| {
                    ProofVerifyError::DoryError(format!(
                        "missing commitment for Stage 8 polynomial {:?}",
                        k
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .unzip();

        Ok(PCS::combine_commitments(&commitments, &coeffs))
    }
}

#[derive(Debug, Clone)]
pub struct JoltSharedPreprocessing<
    PCS: CommitmentScheme = crate::poly::commitment::dory::DoryCommitmentScheme,
> {
    pub program: ProgramPreprocessing<PCS>,
    pub program_meta: ProgramMetadata,
    pub memory_layout: MemoryLayout,
    pub max_padded_trace_length: usize,
    pub bytecode_chunk_count: usize,
}

impl<PCS: CommitmentScheme> JoltSharedPreprocessing<PCS>
where
    PCS::Commitment: CanonicalSerialize,
{
    /// Blake2b-256 digest of the serialized preprocessing, used to bind
    /// the program identity to the Fiat-Shamir transcript.
    pub fn digest(&self) -> [u8; 32] {
        use ark_serialize::CanonicalSerialize;
        use blake2::{digest::consts::U32, Blake2b, Digest};
        let mut buf = Vec::new();
        self.serialize_compressed(&mut buf)
            .expect("serialization cannot fail for in-memory buffer");
        Blake2b::<U32>::digest(&buf).into()
    }
}

impl<PCS: CommitmentScheme> CanonicalSerialize for JoltSharedPreprocessing<PCS>
where
    PCS::Commitment: CanonicalSerialize,
{
    fn serialize_with_mode<W: std::io::Write>(
        &self,
        mut writer: W,
        compress: ark_serialize::Compress,
    ) -> Result<(), ark_serialize::SerializationError> {
        self.program.serialize_with_mode(&mut writer, compress)?;
        self.program_meta
            .serialize_with_mode(&mut writer, compress)?;
        self.memory_layout
            .serialize_with_mode(&mut writer, compress)?;
        self.max_padded_trace_length
            .serialize_with_mode(&mut writer, compress)?;
        self.bytecode_chunk_count
            .serialize_with_mode(&mut writer, compress)?;
        Ok(())
    }

    fn serialized_size(&self, compress: ark_serialize::Compress) -> usize {
        self.program.serialized_size(compress)
            + self.program_meta.serialized_size(compress)
            + self.memory_layout.serialized_size(compress)
            + self.max_padded_trace_length.serialized_size(compress)
            + self.bytecode_chunk_count.serialized_size(compress)
    }
}

impl<PCS: CommitmentScheme> CanonicalDeserialize for JoltSharedPreprocessing<PCS>
where
    PCS::Commitment: CanonicalDeserialize,
{
    fn deserialize_with_mode<R: std::io::Read>(
        mut reader: R,
        compress: ark_serialize::Compress,
        validate: ark_serialize::Validate,
    ) -> Result<Self, ark_serialize::SerializationError> {
        let program = ProgramPreprocessing::deserialize_with_mode(&mut reader, compress, validate)?;
        let program_meta = ProgramMetadata::deserialize_with_mode(&mut reader, compress, validate)?;
        let memory_layout = MemoryLayout::deserialize_with_mode(&mut reader, compress, validate)?;
        let max_padded_trace_length =
            usize::deserialize_with_mode(&mut reader, compress, validate)?;
        let bytecode_chunk_count = usize::deserialize_with_mode(&mut reader, compress, validate)?;
        let shared = Self {
            program,
            program_meta,
            memory_layout,
            max_padded_trace_length,
            bytecode_chunk_count,
        };
        if matches!(validate, ark_serialize::Validate::Yes) {
            ark_serialize::Valid::check(&shared)?;
        }
        Ok(shared)
    }
}

impl<PCS: CommitmentScheme> ark_serialize::Valid for JoltSharedPreprocessing<PCS>
where
    PCS::Commitment: ark_serialize::Valid,
{
    fn check(&self) -> Result<(), ark_serialize::SerializationError> {
        self.program.check()?;
        self.program_meta.check()?;
        self.memory_layout.check()?;
        if self.program.is_committed()
            && !is_valid_committed_bytecode_chunking_for_len(
                self.program.bytecode_len(),
                self.bytecode_chunk_count,
            )
        {
            return Err(ark_serialize::SerializationError::InvalidData);
        }
        Ok(())
    }
}

impl<PCS: CommitmentScheme> JoltSharedPreprocessing<PCS> {
    #[tracing::instrument(skip_all, name = "JoltSharedPreprocessing::new")]
    pub fn new(
        program: ProgramPreprocessing<PCS>,
        memory_layout: MemoryLayout,
        max_padded_trace_length: usize,
    ) -> JoltSharedPreprocessing<PCS> {
        Self {
            program_meta: program.meta(),
            program,
            memory_layout,
            max_padded_trace_length,
            bytecode_chunk_count: DEFAULT_COMMITTED_BYTECODE_CHUNK_COUNT,
        }
    }

    #[tracing::instrument(skip_all, name = "JoltSharedPreprocessing::new_committed")]
    pub fn new_committed(
        program: ProgramPreprocessing<PCS>,
        memory_layout: MemoryLayout,
        max_padded_trace_length: usize,
        bytecode_chunk_count: usize,
    ) -> (
        JoltSharedPreprocessing<PCS>,
        CommittedProgramProverData<PCS>,
        PCS::ProverSetup,
    ) {
        let bytecode_len = program.bytecode_len();
        assert!(
            is_valid_committed_bytecode_chunking_for_len(bytecode_len, bytecode_chunk_count),
            "bytecode chunk count ({bytecode_chunk_count}) must be non-zero, a power of two, at \
             most {}, and divide bytecode size ({bytecode_len})",
            crate::zkvm::bytecode::chunks::MAX_COMMITTED_BYTECODE_CHUNK_COUNT,
        );
        let mut shared = Self {
            program_meta: program.meta(),
            program,
            memory_layout,
            max_padded_trace_length,
            bytecode_chunk_count,
        };
        let (max_total_vars, max_log_k_chunk) = shared.compute_max_total_vars(true);
        let generators = PCS::setup_prover(max_total_vars);
        let (committed_program, prover_data) = shared.program.commit(
            &shared.memory_layout,
            &generators,
            shared.bytecode_chunk_count,
            max_log_k_chunk,
        );
        shared.program = committed_program;
        shared.program_meta = shared.program.meta();
        (shared, prover_data, generators)
    }

    pub fn is_committed_mode(&self) -> bool {
        self.program.is_committed()
    }

    pub fn bytecode_size(&self) -> usize {
        self.program_meta.bytecode_len
    }

    #[inline]
    pub fn committed_program_image_num_words(&self) -> usize {
        self.program_meta
            .committed_program_image_num_words(&self.memory_layout)
    }

    #[inline]
    pub(crate) fn precommitted_candidate_total_vars(
        &self,
        include_committed: bool,
        include_trusted_advice: bool,
        include_untrusted_advice: bool,
    ) -> Vec<usize> {
        let mut candidates = Vec::with_capacity(
            include_committed as usize * 2
                + include_trusted_advice as usize
                + include_untrusted_advice as usize,
        );

        if include_trusted_advice {
            let (trusted_sigma, trusted_nu) = DoryGlobals::advice_sigma_nu_from_max_bytes(
                self.memory_layout.max_trusted_advice_size as usize,
            );
            candidates.push(trusted_sigma + trusted_nu);
        }

        if include_untrusted_advice {
            let (untrusted_sigma, untrusted_nu) = DoryGlobals::advice_sigma_nu_from_max_bytes(
                self.memory_layout.max_untrusted_advice_size as usize,
            );
            candidates.push(untrusted_sigma + untrusted_nu);
        }

        if include_committed {
            let chunk_cycle_log_t = (self.bytecode_size() / self.bytecode_chunk_count)
                .next_power_of_two()
                .log_2();
            candidates.push(committed_lanes().log_2() + chunk_cycle_log_t);
            candidates.push(self.committed_program_image_num_words().log_2());
        }

        candidates
    }

    #[inline]
    pub(crate) fn max_total_vars_from_candidates(
        main_total_vars: usize,
        candidates: impl IntoIterator<Item = usize>,
    ) -> usize {
        let mut max_total_vars = main_total_vars;
        for total_vars in candidates {
            max_total_vars = max_total_vars.max(total_vars);
        }
        max_total_vars
    }

    #[inline]
    pub(crate) fn compute_max_total_vars(&self, include_committed: bool) -> (usize, usize) {
        use common::constants::ONEHOT_CHUNK_THRESHOLD_LOG_T;
        let max_t_any = self.max_padded_trace_length.next_power_of_two();
        let max_log_t = max_t_any.log_2();
        let max_log_k_chunk = if max_log_t < ONEHOT_CHUNK_THRESHOLD_LOG_T {
            4
        } else {
            8
        };

        let max_total_vars = Self::max_total_vars_from_candidates(
            max_log_k_chunk + max_log_t,
            self.precommitted_candidate_total_vars(include_committed, true, true),
        );

        (max_total_vars, max_log_k_chunk)
    }
}

/// Serializable wrapper around [`PedersenGenerators`] for ZK setup transfer.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct BlindfoldSetup<C: JoltCurve>(pub PedersenGenerators<C>);

impl<C: JoltCurve> std::ops::Deref for BlindfoldSetup<C> {
    type Target = PedersenGenerators<C>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<C: JoltCurve> From<BlindfoldSetup<C>> for PedersenGenerators<C> {
    fn from(setup: BlindfoldSetup<C>) -> Self {
        setup.0
    }
}

#[derive(Debug, Clone)]
pub struct JoltVerifierPreprocessing<F, C, PCS>
where
    F: JoltField,
    C: JoltCurve<F = F>,
    PCS: CommitmentScheme<Field = F>,
{
    _curve: std::marker::PhantomData<C>,
    pub generators: PCS::VerifierSetup,
    pub shared: JoltSharedPreprocessing<PCS>,
    pub blindfold_setup: Option<BlindfoldSetup<C>>,
}

impl<F, C, PCS> CanonicalSerialize for JoltVerifierPreprocessing<F, C, PCS>
where
    F: JoltField,
    C: JoltCurve<F = F>,
    PCS: CommitmentScheme<Field = F>,
    PCS::VerifierSetup: CanonicalSerialize,
    PCS::Commitment: CanonicalSerialize,
{
    fn serialize_with_mode<W: std::io::Write>(
        &self,
        mut writer: W,
        compress: ark_serialize::Compress,
    ) -> Result<(), ark_serialize::SerializationError> {
        self.generators.serialize_with_mode(&mut writer, compress)?;
        self.shared.serialize_with_mode(&mut writer, compress)?;
        self.blindfold_setup
            .serialize_with_mode(&mut writer, compress)?;
        Ok(())
    }

    fn serialized_size(&self, compress: ark_serialize::Compress) -> usize {
        self.generators.serialized_size(compress)
            + self.shared.serialized_size(compress)
            + self.blindfold_setup.serialized_size(compress)
    }
}

impl<F, C, PCS> CanonicalDeserialize for JoltVerifierPreprocessing<F, C, PCS>
where
    F: JoltField,
    C: JoltCurve<F = F>,
    PCS: CommitmentScheme<Field = F>,
    PCS::VerifierSetup: CanonicalDeserialize,
    PCS::Commitment: CanonicalDeserialize,
{
    fn deserialize_with_mode<R: std::io::Read>(
        mut reader: R,
        compress: ark_serialize::Compress,
        validate: ark_serialize::Validate,
    ) -> Result<Self, ark_serialize::SerializationError> {
        Ok(Self {
            _curve: std::marker::PhantomData,
            generators: PCS::VerifierSetup::deserialize_with_mode(&mut reader, compress, validate)?,
            shared: JoltSharedPreprocessing::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
            )?,
            blindfold_setup: Option::<BlindfoldSetup<C>>::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
            )?,
        })
    }
}

impl<F, C, PCS> ark_serialize::Valid for JoltVerifierPreprocessing<F, C, PCS>
where
    F: JoltField,
    C: JoltCurve<F = F>,
    PCS: CommitmentScheme<Field = F>,
    PCS::VerifierSetup: ark_serialize::Valid,
    PCS::Commitment: ark_serialize::Valid,
{
    fn check(&self) -> Result<(), ark_serialize::SerializationError> {
        self.generators.check()?;
        self.shared.check()?;
        self.blindfold_setup.check()?;
        Ok(())
    }
}

impl<F, C, PCS> Serializable for JoltVerifierPreprocessing<F, C, PCS>
where
    F: JoltField,
    C: JoltCurve<F = F>,
    PCS: CommitmentScheme<Field = F>,
    PCS::VerifierSetup: CanonicalSerialize + CanonicalDeserialize,
    PCS::Commitment: CanonicalSerialize + CanonicalDeserialize,
{
}

impl<F, C, PCS> JoltVerifierPreprocessing<F, C, PCS>
where
    F: JoltField,
    C: JoltCurve<F = F>,
    PCS: CommitmentScheme<Field = F>,
    PCS::VerifierSetup: CanonicalSerialize + CanonicalDeserialize,
    PCS::Commitment: CanonicalSerialize + CanonicalDeserialize,
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

impl<F: JoltField, C: JoltCurve<F = F>, PCS: CommitmentScheme<Field = F>>
    JoltVerifierPreprocessing<F, C, PCS>
{
    #[tracing::instrument(skip_all, name = "JoltVerifierPreprocessing::new")]
    pub fn new(
        mut shared: JoltSharedPreprocessing<PCS>,
        generators: PCS::VerifierSetup,
        blindfold_setup: Option<BlindfoldSetup<C>>,
    ) -> Self {
        shared.program = shared.program.to_verifier_program();
        Self {
            _curve: std::marker::PhantomData,
            generators,
            shared,
            blindfold_setup,
        }
    }

    #[cfg(feature = "zk")]
    pub fn pedersen_generators(&self, count: usize) -> PedersenGenerators<C> {
        let gens = &self
            .blindfold_setup
            .as_ref()
            .expect("BlindfoldSetup required for ZK mode")
            .0;
        assert!(
            count <= gens.message_generators.len(),
            "Requested {count} Pedersen generators but BlindfoldSetup only has {}",
            gens.message_generators.len()
        );
        PedersenGenerators::new(
            gens.message_generators[..count].to_vec(),
            gens.blinding_generator,
        )
    }
}

#[cfg(feature = "prover")]
impl<F: JoltField, C: JoltCurve<F = F>, PCS: CommitmentScheme<Field = F> + ZkEvalCommitment<C>>
    From<&JoltProverPreprocessing<F, C, PCS>> for JoltVerifierPreprocessing<F, C, PCS>
{
    fn from(prover_preprocessing: &JoltProverPreprocessing<F, C, PCS>) -> Self {
        let shared = prover_preprocessing.shared.clone();
        let generators = PCS::setup_verifier(&prover_preprocessing.generators);
        #[cfg(not(feature = "zk"))]
        let blindfold_setup = None;
        #[cfg(feature = "zk")]
        let blindfold_setup = Some(prover_preprocessing.blindfold_setup());
        Self::new(shared, generators, blindfold_setup)
    }
}
