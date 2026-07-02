//! Top-level verifier entry point.

use common::jolt_device::JoltDevice;
use jolt_claims::protocols::jolt::{
    JoltOneHotConfig, JoltReadWriteConfig, JoltRelationId, TracePolynomialOrder,
};
use jolt_crypto::{HomomorphicCommitment, VectorCommitment};
use jolt_field::{Field, RingAccumulator, WithAccumulator};
use jolt_openings::{AdditivelyHomomorphic, CommitmentScheme, ZkOpeningScheme};
use jolt_program::preprocess::{compute_max_ram_k, compute_min_ram_k};
use jolt_sumcheck::SumcheckProof;
use jolt_transcript::{AppendToTranscript, Label, LabelWithCount, Transcript, U64Word};

use crate::{
    config::{validate_proof_config, JoltProtocolConfig},
    preprocessing::JoltVerifierPreprocessing,
    proof::{JoltCommitments, JoltProof},
    stages::{
        stage1, stage2, stage3, stage4, stage5, stage6, stage7, stage8,
        zk::{blindfold, committed, inputs::BlindFoldInputs, outputs::zk_stage_outputs},
        CommittedProgramSchedule, PrecommittedSchedule,
    },
    VerifierError,
};

/// Proof-derived configuration that participates in the Fiat-Shamir preamble.
/// Bundles the parameters [`absorb_transcript_preamble`] reads from the proof so
/// the modular prover can seed an identical transcript before it has a finished
/// [`JoltProof`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProofTranscriptConfig {
    pub rw_config: JoltReadWriteConfig,
    pub one_hot_config: JoltOneHotConfig,
    pub trace_polynomial_order: TracePolynomialOrder,
}

impl ProofTranscriptConfig {
    pub const fn new(
        rw_config: JoltReadWriteConfig,
        one_hot_config: JoltOneHotConfig,
        trace_polynomial_order: TracePolynomialOrder,
    ) -> Self {
        Self {
            rw_config,
            one_hot_config,
            trace_polynomial_order,
        }
    }
}

pub fn verify<F, PCS, VC, T>(
    preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    public_io: &JoltDevice,
    proof: &JoltProof<PCS, VC>,
    trusted_advice_commitment: Option<&PCS::Output>,
    zk: bool,
) -> Result<(), VerifierError>
where
    F: Field + AppendToTranscript,
    PCS: CommitmentScheme<Field = F>
        + AdditivelyHomomorphic
        + ZkOpeningScheme<HidingCommitment = VC::Output>,
    PCS::Output: AppendToTranscript + HomomorphicCommitment<F>,
    VC: VectorCommitment<Field = F>,
    VC::Output: Copy + HomomorphicCommitment<F> + AppendToTranscript,
    T: Transcript<Challenge = F>,
    <F as WithAccumulator>::Accumulator: RingAccumulator<Element = F>,
{
    let checked = validate_inputs(
        preprocessing,
        public_io,
        proof,
        trusted_advice_commitment.is_some(),
        zk,
    )?;
    validate_proof_consistency(proof, checked.zk)?;
    validate_proof_config(&JoltProtocolConfig::for_zk(checked.zk), proof)?;

    let mut transcript = T::new(b"Jolt");
    absorb_preamble(&checked, proof, &mut transcript);
    absorb_commitments(
        preprocessing,
        proof,
        trusted_advice_commitment,
        &mut transcript,
    );

    // Built once for the whole verification and shared by the stages that read the
    // RA layout (5–8), instead of each rebuilding the same dimensions.
    let formula_dimensions = crate::stages::build_formula_dimensions(
        proof,
        preprocessing,
        &checked,
        checked.trace_length.ilog2() as usize,
        JoltRelationId::InstructionReadRaf,
    )?;

    let stage1 = stage1::verify(&checked, proof, &mut transcript)?;
    let stage2 = stage2::verify(&checked, proof, &mut transcript, &stage1)?;
    let stage3 = stage3::verify(&checked, proof, &mut transcript, &stage1, &stage2)?;
    let stage4 = stage4::verify(
        &checked,
        preprocessing,
        proof,
        &mut transcript,
        &stage2,
        &stage3,
    )?;
    let stage5 = stage5::verify(
        &checked,
        proof,
        &formula_dimensions,
        &mut transcript,
        &stage2,
        &stage4,
    )?;
    let stage6 = stage6::verify(
        &checked,
        preprocessing,
        proof,
        &formula_dimensions,
        &mut transcript,
        &stage1,
        &stage2,
        &stage3,
        &stage4,
        &stage5,
    )?;
    let stage7 = stage7::verify(
        &checked,
        proof,
        &formula_dimensions,
        &mut transcript,
        &stage4,
        &stage6,
    )?;
    let stage8 = stage8::verify(
        &checked,
        preprocessing,
        proof,
        &formula_dimensions,
        trusted_advice_commitment,
        &mut transcript,
        &stage6,
        &stage7,
    )?;

    if checked.zk {
        let zk_stages = zk_stage_outputs::<PCS, VC>(
            &stage1, &stage2, &stage3, &stage4, &stage5, &stage6, &stage7, &stage8,
        )?;
        let blindfold = blindfold::build(BlindFoldInputs {
            checked: &checked,
            preprocessing,
            proof,
            stage1: zk_stages.stage1,
            stage2: zk_stages.stage2,
            stage3: zk_stages.stage3,
            stage4: zk_stages.stage4,
            stage5: zk_stages.stage5,
            stage6: zk_stages.stage6,
            stage7: zk_stages.stage7,
            stage8: zk_stages.stage8,
        })?;
        let vc_setup = preprocessing
            .vc_setup
            .as_ref()
            .ok_or(VerifierError::MissingVectorCommitmentSetup)?;
        transcript.append(&Label(b"BlindFold"));
        blindfold
            .protocol
            .verify::<VC, T>(proof.blindfold_proof()?, vc_setup, &mut transcript)
            .map_err(|error| VerifierError::BlindFoldVerificationFailed {
                reason: error.to_string(),
            })?;
        return Ok(());
    }

    let stage8::Stage8Output::Clear(_stage8) = stage8 else {
        return Err(VerifierError::ExpectedClearProof { field: "stage8" });
    };

    Ok(())
}

/// Verifier state captured immediately before stage 1, after input validation
/// and the preamble/commitment absorption. The modular prover drives the staged
/// verification itself, so it reuses this entry point to obtain the checked
/// inputs and a transcript seeded identically to [`verify`].
#[derive(Debug)]
pub struct PreStage1VerifierState<T> {
    pub checked: CheckedInputs,
    pub transcript: T,
}

/// Runs the [`verify`] preamble — input validation, proof-consistency and
/// config checks, then preamble + commitment absorption — and returns the
/// pre-stage-1 state. WARNING: the absorption order here must stay identical to
/// [`verify`] or the prover and verifier transcripts diverge.
pub fn verify_until_stage1<PCS, VC, T, ZkProof>(
    preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    public_io: &JoltDevice,
    proof: &JoltProof<PCS, VC, ZkProof>,
    trusted_advice_commitment: Option<&PCS::Output>,
    zk: bool,
) -> Result<PreStage1VerifierState<T>, VerifierError>
where
    PCS: CommitmentScheme,
    PCS::Output: AppendToTranscript,
    VC: VectorCommitment<Field = PCS::Field>,
    VC::Output: AppendToTranscript,
    T: Transcript<Challenge = PCS::Field>,
{
    let checked = validate_inputs(
        preprocessing,
        public_io,
        proof,
        trusted_advice_commitment.is_some(),
        zk,
    )?;
    validate_proof_consistency(proof, checked.zk)?;
    validate_proof_config(&JoltProtocolConfig::for_zk(checked.zk), proof)?;

    let mut transcript = T::new(b"Jolt");
    absorb_preamble(&checked, proof, &mut transcript);
    absorb_commitments(
        preprocessing,
        proof,
        trusted_advice_commitment,
        &mut transcript,
    );

    Ok(PreStage1VerifierState {
        checked,
        transcript,
    })
}

#[expect(
    non_snake_case,
    reason = "Matches current jolt-prover-legacy proof field name."
)]
#[derive(Clone, Debug, PartialEq)]
pub struct CheckedInputs {
    pub public_io: JoltDevice,
    pub zk: bool,
    pub trace_length: usize,
    pub ram_K: usize,
    pub entry_address: u64,
    pub preprocessing_digest: [u8; 32],
    pub trusted_advice_commitment_present: bool,
    pub vc_capacity: Option<usize>,
    pub precommitted: PrecommittedSchedule,
}

pub fn validate_inputs<PCS, VC, ZkProof>(
    preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    public_io: &JoltDevice,
    proof: &JoltProof<PCS, VC, ZkProof>,
    trusted_advice_commitment_present: bool,
    zk: bool,
) -> Result<CheckedInputs, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    validate_inputs_from_parts(
        preprocessing,
        public_io,
        proof.trace_length,
        proof.ram_K,
        proof.trace_polynomial_order,
        proof.one_hot_config,
        trusted_advice_commitment_present,
        proof.untrusted_advice_commitment.is_some(),
        zk,
    )
}

/// Validates the verifier inputs from the individual proof-derived parameters,
/// rather than a finished [`JoltProof`]. The modular prover calls this before it
/// has assembled a proof, so it must supply `trace_length`, `ram_k`,
/// `trace_polynomial_order`, and `one_hot_config` directly, along with whether
/// the trusted/untrusted advice commitments are present.
///
/// [`validate_inputs`] delegates here, so the two paths produce identical
/// [`CheckedInputs`] — including the [`PrecommittedSchedule`] — given matching
/// parameters.
#[expect(
    clippy::too_many_arguments,
    reason = "Mirrors the proof-derived inputs validate_inputs threads through; bundling them would obscure the FS-critical parameter set."
)]
pub fn validate_inputs_from_parts<PCS, VC>(
    preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    public_io: &JoltDevice,
    trace_length: usize,
    ram_k: usize,
    trace_polynomial_order: TracePolynomialOrder,
    one_hot_config: JoltOneHotConfig,
    trusted_advice_commitment_present: bool,
    untrusted_advice_commitment_present: bool,
    zk: bool,
) -> Result<CheckedInputs, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    let memory_layout = preprocessing.program.memory_layout();
    if &public_io.memory_layout != memory_layout {
        return Err(VerifierError::MemoryLayoutMismatch);
    }

    let max_input_size = memory_layout.max_input_size as usize;
    if public_io.inputs.len() > max_input_size {
        return Err(VerifierError::InputTooLarge {
            got: public_io.inputs.len(),
            max: max_input_size,
        });
    }

    let max_output_size = memory_layout.max_output_size as usize;
    if public_io.outputs.len() > max_output_size {
        return Err(VerifierError::OutputTooLarge {
            got: public_io.outputs.len(),
            max: max_output_size,
        });
    }

    if !trace_length.is_power_of_two()
        || trace_length > preprocessing.program.max_padded_trace_length()
    {
        return Err(VerifierError::InvalidTraceLength {
            got: trace_length,
            max: preprocessing.program.max_padded_trace_length(),
        });
    }

    let min_ram_k = compute_min_ram_k(
        preprocessing.program.min_bytecode_address(),
        preprocessing.program.program_image_len_words(),
        memory_layout,
    )
    .map_err(|error| VerifierError::InvalidMemoryLayout {
        reason: error.to_string(),
    })?;
    let max_ram_k =
        compute_max_ram_k(memory_layout).map_err(|error| VerifierError::InvalidMemoryLayout {
            reason: error.to_string(),
        })?;
    if !ram_k.is_power_of_two() || ram_k < min_ram_k || ram_k > max_ram_k {
        return Err(VerifierError::InvalidRamK {
            got: ram_k,
            min: min_ram_k,
            max: max_ram_k,
        });
    }

    let vc_capacity = if zk {
        Some(validate_zk_vector_commitment_setup::<PCS, VC>(
            preprocessing,
        )?)
    } else {
        None
    };

    let mut normalized_public_io = public_io.clone();
    normalized_public_io.outputs.truncate(
        normalized_public_io
            .outputs
            .iter()
            .rposition(|&byte| byte != 0)
            .map_or(0, |position| position + 1),
    );

    let committed_program = preprocessing
        .program
        .committed()
        .map(|committed| {
            let program_image_start_index = memory_layout
                .remapped_word_address(committed.meta.min_bytecode_address)
                .map_err(|error| VerifierError::InvalidCommittedProgram {
                    reason: error.to_string(),
                })?;
            if committed.meta.entry_bytecode_index >= committed.meta.bytecode_len {
                return Err(VerifierError::InvalidCommittedProgram {
                    reason: format!(
                        "entry bytecode index {} is out of range for bytecode length {}",
                        committed.meta.entry_bytecode_index, committed.meta.bytecode_len
                    ),
                });
            }
            Ok(CommittedProgramSchedule {
                bytecode_len: committed.meta.bytecode_len,
                bytecode_chunk_count: committed.bytecode_chunk_count(),
                program_image_len_words: committed.meta.program_image_len_words,
                program_image_start_index: program_image_start_index as usize,
            })
        })
        .transpose()?;
    let precommitted = PrecommittedSchedule::new(
        trace_polynomial_order,
        trace_length.ilog2() as usize,
        one_hot_config.committed_chunk_bits(),
        trusted_advice_commitment_present.then_some(memory_layout.max_trusted_advice_size as usize),
        untrusted_advice_commitment_present
            .then_some(memory_layout.max_untrusted_advice_size as usize),
        committed_program,
    )
    .map_err(|error| VerifierError::InvalidPrecommittedSchedule {
        reason: error.to_string(),
    })?;

    Ok(CheckedInputs {
        public_io: normalized_public_io,
        zk,
        trace_length,
        ram_K: ram_k,
        entry_address: preprocessing.program.entry_address(),
        preprocessing_digest: preprocessing.preprocessing_digest,
        trusted_advice_commitment_present,
        vc_capacity,
        precommitted,
    })
}

fn validate_zk_vector_commitment_setup<PCS, VC>(
    preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
) -> Result<usize, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    let setup = preprocessing
        .vc_setup
        .as_ref()
        .ok_or(VerifierError::MissingVectorCommitmentSetup)?;
    let required = committed::zk_vector_commitment_capacity_requirement();
    let got = VC::capacity(setup);
    if got < required {
        return Err(VerifierError::InvalidVectorCommitmentCapacity { required, got });
    }

    Ok(got)
}

pub(crate) fn absorb_preamble<PCS, VC, ZkProof, T>(
    checked: &CheckedInputs,
    proof: &JoltProof<PCS, VC, ZkProof>,
    transcript: &mut T,
) where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
    T: Transcript<Challenge = PCS::Field>,
{
    absorb_transcript_preamble(
        checked,
        ProofTranscriptConfig::new(
            proof.rw_config,
            proof.one_hot_config,
            proof.trace_polynomial_order,
        ),
        transcript,
    );
}

/// Absorbs the Jolt Fiat-Shamir preamble: the preprocessing digest, public I/O
/// metadata, and the proof-derived structural parameters carried by
/// [`ProofTranscriptConfig`]. WARNING: the byte order here is consensus-critical
/// — it must stay identical to the order [`verify`] uses (via [`absorb_preamble`])
/// or prover and verifier transcripts diverge.
pub fn absorb_transcript_preamble<T>(
    checked: &CheckedInputs,
    config: ProofTranscriptConfig,
    transcript: &mut T,
) where
    T: Transcript,
{
    let public_io = &checked.public_io;
    absorb_labeled_bytes(
        transcript,
        b"preprocessing_digest",
        &checked.preprocessing_digest,
    );
    absorb_labeled_u64(
        transcript,
        b"max_input_size",
        public_io.memory_layout.max_input_size,
    );
    absorb_labeled_u64(
        transcript,
        b"max_output_size",
        public_io.memory_layout.max_output_size,
    );
    absorb_labeled_u64(transcript, b"heap_size", public_io.memory_layout.heap_size);
    absorb_labeled_bytes(transcript, b"inputs", &public_io.inputs);
    absorb_labeled_bytes(transcript, b"outputs", &public_io.outputs);
    absorb_labeled_u64(transcript, b"panic", public_io.panic as u64);
    absorb_labeled_u64(transcript, b"ram_K", checked.ram_K as u64);
    absorb_labeled_u64(transcript, b"trace_length", checked.trace_length as u64);
    absorb_labeled_u64(transcript, b"entry_address", checked.entry_address);
    absorb_labeled_u64(
        transcript,
        b"ram_rw_phase1_num_rounds",
        config.rw_config.ram_rw_phase1_num_rounds as u64,
    );
    absorb_labeled_u64(
        transcript,
        b"ram_rw_phase2_num_rounds",
        config.rw_config.ram_rw_phase2_num_rounds as u64,
    );
    absorb_labeled_u64(
        transcript,
        b"registers_rw_phase1_num_rounds",
        config.rw_config.registers_rw_phase1_num_rounds as u64,
    );
    absorb_labeled_u64(
        transcript,
        b"registers_rw_phase2_num_rounds",
        config.rw_config.registers_rw_phase2_num_rounds as u64,
    );
    absorb_labeled_u64(
        transcript,
        b"log_k_chunk",
        config.one_hot_config.log_k_chunk as u64,
    );
    absorb_labeled_u64(
        transcript,
        b"lookups_ra_virtual_log_k_chunk",
        config.one_hot_config.lookups_ra_virtual_log_k_chunk as u64,
    );
    absorb_labeled_u64(
        transcript,
        b"dory_layout",
        config.trace_polynomial_order.transcript_scalar(),
    );
}

pub(crate) fn absorb_commitments<PCS, VC, ZkProof, T>(
    preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    proof: &JoltProof<PCS, VC, ZkProof>,
    trusted_advice_commitment: Option<&PCS::Output>,
    transcript: &mut T,
) where
    PCS: CommitmentScheme,
    PCS::Output: AppendToTranscript,
    VC: VectorCommitment<Field = PCS::Field>,
    T: Transcript<Challenge = PCS::Field>,
{
    absorb_transcript_commitments(
        &proof.commitments,
        proof.untrusted_advice_commitment.as_ref(),
        trusted_advice_commitment,
        transcript,
    );
    if let Some(committed) = preprocessing.program.committed() {
        for commitment in &committed.bytecode_chunk_commitments {
            append_payload_label(transcript, b"bytecode_chunk_commit", commitment);
            transcript.append(commitment);
        }
        append_payload_label(
            transcript,
            b"program_image_commitment",
            &committed.program_image_commitment,
        );
        transcript.append(&committed.program_image_commitment);
    }
}

/// Absorbs the proof-derived polynomial commitments (increment, one-hot `ra`,
/// and optional advice commitments) in the consensus-critical order. WARNING:
/// this covers only the commitments carried by the proof itself; committed
/// program-image commitments live in the preprocessing and are absorbed
/// separately by [`absorb_commitments`] immediately after this call.
pub fn absorb_transcript_commitments<C, T>(
    commitments: &JoltCommitments<C>,
    untrusted_advice_commitment: Option<&C>,
    trusted_advice_commitment: Option<&C>,
    transcript: &mut T,
) where
    C: AppendToTranscript,
    T: Transcript,
{
    let mut absorb_commitment = |commitment: &C| {
        append_payload_label(transcript, b"commitment", commitment);
        transcript.append(commitment);
    };
    absorb_commitment(&commitments.rd_inc);
    absorb_commitment(&commitments.ram_inc);
    for commitment in &commitments.ra.instruction {
        absorb_commitment(commitment);
    }
    for commitment in &commitments.ra.ram {
        absorb_commitment(commitment);
    }
    for commitment in &commitments.ra.bytecode {
        absorb_commitment(commitment);
    }
    if let Some(untrusted_advice_commitment) = untrusted_advice_commitment {
        append_payload_label(transcript, b"untrusted_advice", untrusted_advice_commitment);
        transcript.append(untrusted_advice_commitment);
    }
    if let Some(trusted_advice_commitment) = trusted_advice_commitment {
        append_payload_label(transcript, b"trusted_advice", trusted_advice_commitment);
        transcript.append(trusted_advice_commitment);
    }
}

fn append_payload_label<T, A>(transcript: &mut T, label: &'static [u8], payload: &A)
where
    T: Transcript,
    A: AppendToTranscript,
{
    if let Some(len) = payload.transcript_payload_len() {
        transcript.append(&LabelWithCount(label, len));
    } else {
        transcript.append(&Label(label));
    }
}

fn absorb_labeled_bytes<T: Transcript>(transcript: &mut T, label: &'static [u8], bytes: &[u8]) {
    transcript.append(&LabelWithCount(label, bytes.len() as u64));
    transcript.append_bytes(bytes);
}

fn absorb_labeled_u64<T: Transcript>(transcript: &mut T, label: &'static [u8], value: u64) {
    transcript.append(&Label(label));
    transcript.append(&U64Word(value));
}

pub fn validate_proof_consistency<PCS, VC, ZkProof>(
    proof: &JoltProof<PCS, VC, ZkProof>,
    zk: bool,
) -> Result<(), VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    validate_sumcheck_representation(
        &proof.stages.stage1_uni_skip_first_round_proof,
        "stage1_uni_skip_first_round_proof",
        zk,
    )?;
    validate_sumcheck_representation(
        &proof.stages.stage1_sumcheck_proof,
        "stage1_sumcheck_proof",
        zk,
    )?;
    validate_sumcheck_representation(
        &proof.stages.stage2_uni_skip_first_round_proof,
        "stage2_uni_skip_first_round_proof",
        zk,
    )?;
    validate_sumcheck_representation(
        &proof.stages.stage2_sumcheck_proof,
        "stage2_sumcheck_proof",
        zk,
    )?;
    validate_sumcheck_representation(
        &proof.stages.stage3_sumcheck_proof,
        "stage3_sumcheck_proof",
        zk,
    )?;
    validate_sumcheck_representation(
        &proof.stages.stage4_sumcheck_proof,
        "stage4_sumcheck_proof",
        zk,
    )?;
    validate_sumcheck_representation(
        &proof.stages.stage5_sumcheck_proof,
        "stage5_sumcheck_proof",
        zk,
    )?;
    validate_sumcheck_representation(
        &proof.stages.stage6a_sumcheck_proof,
        "stage6a_sumcheck_proof",
        zk,
    )?;
    validate_sumcheck_representation(
        &proof.stages.stage6b_sumcheck_proof,
        "stage6b_sumcheck_proof",
        zk,
    )?;
    validate_sumcheck_representation(
        &proof.stages.stage7_sumcheck_proof,
        "stage7_sumcheck_proof",
        zk,
    )?;

    match (&proof.claims, zk) {
        (crate::proof::JoltProofClaims::Clear(_), false)
        | (crate::proof::JoltProofClaims::Zk { .. }, true) => {}
        (crate::proof::JoltProofClaims::Clear(_), true) => {
            return Err(VerifierError::UnexpectedOpeningClaims);
        }
        (crate::proof::JoltProofClaims::Zk { .. }, false) => {
            return Err(VerifierError::UnexpectedBlindFoldProof);
        }
    }
    Ok(())
}

fn validate_sumcheck_representation<F, RoundCommitment>(
    proof: &SumcheckProof<F, RoundCommitment>,
    field: &'static str,
    zk: bool,
) -> Result<(), VerifierError>
where
    F: Field,
{
    if proof.is_committed() == zk {
        return Ok(());
    }

    if zk {
        Err(VerifierError::ExpectedCommittedProof { field })
    } else {
        Err(VerifierError::ExpectedClearProof { field })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proof::{ClearProofClaims, JoltProofClaims, JoltStageProofs};
    use common::jolt_device::{JoltDevice, MemoryConfig};
    use jolt_claims::protocols::jolt::{JoltOneHotConfig, JoltReadWriteConfig};
    use jolt_crypto::{Bn254G1, Commitment, Pedersen, PedersenSetup, VectorCommitmentOpening};
    use jolt_field::Fr;
    use jolt_openings::{CommitmentScheme, OpeningsError};
    use jolt_poly::{MultilinearPoly, Polynomial};
    use jolt_program::preprocess::{
        BytecodePreprocessing, JoltProgramPreprocessing, RAMPreprocessing,
    };
    use jolt_sumcheck::{
        ClearProof, ClearSumcheckProof, CommittedSumcheckProof, CompressedSumcheckProof,
    };
    use jolt_transcript::Transcript;
    use num_traits::Zero;

    #[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
    struct TestPcs;

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
    struct TestCommitment;

    impl Commitment for TestPcs {
        type Output = TestCommitment;
    }

    impl CommitmentScheme for TestPcs {
        type Field = Fr;
        type Proof = ();
        type ProverSetup = ();
        type VerifierSetup = ();
        type Polynomial = Polynomial<Fr>;
        type OpeningHint = ();
        type SetupParams = ();

        fn setup(
            _params: Self::SetupParams,
        ) -> Result<(Self::ProverSetup, Self::VerifierSetup), OpeningsError> {
            Ok(((), ()))
        }

        fn verifier_setup(_prover_setup: &Self::ProverSetup) -> Self::VerifierSetup {}

        fn commit<P: MultilinearPoly<Self::Field> + ?Sized>(
            _poly: &P,
            _setup: &Self::ProverSetup,
        ) -> Result<(Self::Output, Self::OpeningHint), OpeningsError> {
            Ok((TestCommitment, ()))
        }

        fn open<P: MultilinearPoly<Self::Field> + ?Sized>(
            _poly: &P,
            _point: &[Self::Field],
            _eval: Self::Field,
            _setup: &Self::ProverSetup,
            _hint: Option<Self::OpeningHint>,
            _transcript: &mut impl Transcript<Challenge = Self::Field>,
        ) -> Result<Self::Proof, OpeningsError> {
            Ok(())
        }

        fn verify(
            _commitment: &Self::Output,
            _point: &[Self::Field],
            _eval: Self::Field,
            _proof: &Self::Proof,
            _setup: &Self::VerifierSetup,
            _transcript: &mut impl Transcript<Challenge = Self::Field>,
        ) -> Result<(), OpeningsError> {
            Ok(())
        }
    }

    impl jolt_transcript::AppendToTranscript for TestCommitment {
        fn append_to_transcript<T: Transcript>(&self, _transcript: &mut T) {}
    }

    type TestProof = JoltProof<TestPcs, Pedersen<Bn254G1>>;
    type TestClaims = JoltProofClaims<Fr, jolt_blindfold::BlindFoldProof<Fr, Bn254G1>>;

    #[test]
    fn proof_wrapper_uses_modular_trait_bounds() {
        fn assert_proof_traits<T>()
        where
            T: Clone
                + std::fmt::Debug
                + PartialEq
                + Eq
                + Send
                + Sync
                + 'static
                + serde::Serialize
                + serde::de::DeserializeOwned,
        {
        }

        assert_proof_traits::<TestProof>();
    }

    #[test]
    fn accepts_standard_proof_consistency() {
        let proof = proof_with_zk(false, clear_claims());

        assert!(validate_proof_consistency(&proof, false).is_ok());
    }

    #[test]
    fn accepts_zk_proof_consistency() {
        let proof = proof_with_zk(true, zk_claims());

        assert!(validate_proof_consistency(&proof, true).is_ok());
    }

    #[test]
    fn rejects_wrong_stage_representation() {
        let mut proof = proof_with_zk(false, clear_claims());
        proof.stages.stage5_sumcheck_proof =
            SumcheckProof::Committed(CommittedSumcheckProof::default());

        assert!(matches!(
            validate_proof_consistency(&proof, false),
            Err(VerifierError::ExpectedClearProof {
                field: "stage5_sumcheck_proof",
            })
        ));
    }

    #[test]
    fn rejects_wrong_verifier_zk_flag() {
        let proof = proof_with_zk(false, clear_claims());

        assert!(matches!(
            validate_proof_consistency(&proof, true),
            Err(VerifierError::ExpectedCommittedProof {
                field: "stage1_uni_skip_first_round_proof",
            })
        ));
    }

    #[test]
    fn checks_payload_for_selected_zk_flag() {
        assert!(matches!(
            validate_proof_consistency(&proof_with_zk(false, zk_claims()), false),
            Err(VerifierError::UnexpectedBlindFoldProof)
        ));
        assert!(matches!(
            validate_proof_consistency(&proof_with_zk(true, clear_claims()), true),
            Err(VerifierError::UnexpectedOpeningClaims)
        ));
    }

    #[test]
    fn validate_inputs_normalizes_public_output() {
        let preprocessing = test_preprocessing();
        let mut public_io = JoltDevice {
            memory_layout: preprocessing.program.memory_layout().clone(),
            inputs: vec![1, 2],
            outputs: vec![3, 0, 0],
            ..JoltDevice::default()
        };
        let proof = proof_with_zk(false, clear_claims());

        let checked = validate_inputs(&preprocessing, &public_io, &proof, false, false);
        assert!(checked.is_ok());
        let Ok(checked) = checked else {
            return;
        };

        assert_eq!(checked.public_io.inputs, vec![1, 2]);
        assert_eq!(checked.public_io.outputs, vec![3]);
        assert_eq!(checked.trace_length, proof.trace_length);
        assert_eq!(checked.ram_K, proof.ram_K);

        public_io.outputs = vec![0, 0];
        let checked = validate_inputs(&preprocessing, &public_io, &proof, false, false);
        assert!(checked.is_ok());
        let Ok(checked) = checked else {
            return;
        };
        assert!(checked.public_io.outputs.is_empty());
    }

    #[test]
    fn validate_inputs_rejects_public_io_layout_mismatch() {
        let preprocessing = test_preprocessing();
        let public_io = JoltDevice::default();
        let proof = proof_with_zk(false, clear_claims());

        assert!(matches!(
            validate_inputs(&preprocessing, &public_io, &proof, false, false),
            Err(VerifierError::MemoryLayoutMismatch)
        ));
    }

    #[test]
    fn validate_inputs_rejects_ram_domain_below_layout_minimum() {
        let preprocessing = test_preprocessing();
        let public_io = JoltDevice {
            memory_layout: preprocessing.program.memory_layout().clone(),
            ..JoltDevice::default()
        };
        let mut proof = proof_with_zk(false, clear_claims());
        proof.ram_K = 2;

        assert!(matches!(
            validate_inputs(&preprocessing, &public_io, &proof, false, false),
            Err(VerifierError::InvalidRamK { got: 2, min: 4, .. })
        ));
    }

    #[test]
    fn validate_inputs_rejects_ram_domain_above_layout_maximum() {
        let preprocessing = test_preprocessing();
        let public_io = JoltDevice {
            memory_layout: preprocessing.program.memory_layout().clone(),
            ..JoltDevice::default()
        };
        let mut proof = proof_with_zk(false, clear_claims());
        proof.ram_K = 1 << 20;

        assert!(matches!(
            validate_inputs(&preprocessing, &public_io, &proof, false, false),
            Err(VerifierError::InvalidRamK {
                got,
                min: 4,
                max,
            }) if got == 1 << 20 && max < got
        ));
    }

    #[test]
    fn validate_inputs_rejects_missing_zk_vector_commitment_setup() {
        let preprocessing = test_preprocessing();
        let public_io = JoltDevice {
            memory_layout: preprocessing.program.memory_layout().clone(),
            ..JoltDevice::default()
        };
        let proof = proof_with_zk(true, zk_claims());

        assert!(matches!(
            validate_inputs(&preprocessing, &public_io, &proof, false, true),
            Err(VerifierError::MissingVectorCommitmentSetup)
        ));
    }

    #[test]
    fn validate_inputs_rejects_small_zk_vector_commitment_setup() {
        let mut preprocessing = test_preprocessing();
        preprocessing.vc_setup = Some(PedersenSetup::new(
            vec![Bn254G1::default()],
            Bn254G1::default(),
        ));
        let public_io = JoltDevice {
            memory_layout: preprocessing.program.memory_layout().clone(),
            ..JoltDevice::default()
        };
        let proof = proof_with_zk(true, zk_claims());

        assert!(matches!(
            validate_inputs(&preprocessing, &public_io, &proof, false, true),
            Err(VerifierError::InvalidVectorCommitmentCapacity { got: 1, .. })
        ));
    }

    fn proof_with_zk(is_zk: bool, claims: TestClaims) -> TestProof {
        JoltProof::new(
            test_commitments(),
            stage_proofs(is_zk),
            (),
            None,
            claims,
            1,
            4,
            JoltReadWriteConfig {
                ram_rw_phase1_num_rounds: 0,
                ram_rw_phase2_num_rounds: 0,
                registers_rw_phase1_num_rounds: 0,
                registers_rw_phase2_num_rounds: 0,
            },
            JoltOneHotConfig {
                log_k_chunk: 0,
                lookups_ra_virtual_log_k_chunk: 0,
            },
            crate::proof::TracePolynomialOrder::CycleMajor,
        )
    }

    fn test_commitments() -> crate::proof::JoltCommitments<TestCommitment> {
        crate::proof::JoltCommitments::new(
            TestCommitment,
            TestCommitment,
            crate::proof::JoltRaCommitments::new(
                Vec::<TestCommitment>::new(),
                Vec::<TestCommitment>::new(),
                Vec::<TestCommitment>::new(),
            ),
        )
    }

    fn clear_claims() -> TestClaims {
        let zero = Fr::zero();

        JoltProofClaims::Clear(ClearProofClaims {
            stage1: stage1::outputs::Stage1OutputClaims {
                uniskip_output_claim: zero,
                outer: empty_spartan_outer_claims(),
            },
            stage2: stage2::outputs::Stage2OutputClaims {
                product_uniskip_output_claim: zero,
                batch_outputs: stage2::outputs::Stage2BatchOutputClaims {
                    ram_read_write: stage2::outputs::RamReadWriteOutputClaims {
                        val: zero,
                        ra: zero,
                        inc: zero,
                    },
                    product_remainder: stage2::outputs::ProductRemainderOutputClaims {
                        left_instruction_input: zero,
                        right_instruction_input: zero,
                        jump_flag: zero,
                        write_lookup_output_to_rd: zero,
                        lookup_output: zero,
                        branch_flag: zero,
                        next_is_noop: zero,
                        virtual_instruction: zero,
                    },
                    instruction_claim_reduction:
                        stage2::outputs::InstructionClaimReductionOutputClaims {
                            lookup_output: None,
                            left_lookup_operand: zero,
                            right_lookup_operand: zero,
                            left_instruction_input: None,
                            right_instruction_input: None,
                        },
                    ram_raf_evaluation: stage2::outputs::RamRafEvaluationOutputClaims {
                        ram_ra: zero,
                    },
                    ram_output_check: stage2::outputs::RamOutputCheckOutputClaims {
                        val_final: zero,
                    },
                },
            },
            stage3: stage3::outputs::Stage3OutputClaims {
                shift: stage3::outputs::SpartanShiftOutputClaims {
                    unexpanded_pc: zero,
                    pc: zero,
                    is_virtual: zero,
                    is_first_in_sequence: zero,
                    is_noop: zero,
                },
                instruction_input: stage3::outputs::InstructionInputOutputClaims {
                    left_operand_is_rs1: zero,
                    rs1_value: zero,
                    left_operand_is_pc: zero,
                    unexpanded_pc: zero,
                    right_operand_is_rs2: zero,
                    rs2_value: zero,
                    right_operand_is_imm: zero,
                    imm: zero,
                },
                registers_claim_reduction: stage3::outputs::RegistersClaimReductionOutputClaims {
                    rd_write_value: zero,
                    rs1_value: zero,
                    rs2_value: zero,
                },
            },
            stage4: stage4::outputs::Stage4OutputClaims {
                advice: stage4::RamValCheckAdviceClaims {
                    untrusted: None,
                    trusted: None,
                },
                program_image_contribution: None,
                registers_read_write: stage4::RegistersReadWriteOutputClaims {
                    registers_val: zero,
                    rs1_ra: zero,
                    rs2_ra: zero,
                    rd_wa: zero,
                    rd_inc: zero,
                },
                ram_val_check: stage4::RamValCheckOutputClaims {
                    ram_ra: zero,
                    ram_inc: zero,
                },
            },
            stage5: stage5::outputs::Stage5OutputClaims {
                instruction_read_raf: stage5::InstructionReadRafOutputClaims {
                    lookup_table_flags: Vec::new(),
                    instruction_ra: Vec::new(),
                    instruction_raf_flag: zero,
                },
                ram_ra_claim_reduction: stage5::RamRaClaimReductionOutputClaims { ram_ra: zero },
                registers_val_evaluation: stage5::RegistersValEvaluationOutputClaims {
                    rd_inc: zero,
                    rd_wa: zero,
                },
            },
            stage6: stage6::outputs::Stage6OutputClaims {
                address_phase: stage6::outputs::Stage6AddressPhaseClaims {
                    bytecode_read_raf: zero,
                    booleanity: zero,
                    bytecode_val_stages: None,
                },
                bytecode_read_raf: stage6::outputs::BytecodeReadRafOutputClaims {
                    bytecode_ra: Vec::new(),
                },
                booleanity: stage6::outputs::BooleanityOutputClaims {
                    instruction_ra: Vec::new(),
                    bytecode_ra: Vec::new(),
                    ram_ra: Vec::new(),
                },
                ram_hamming_booleanity: stage6::outputs::RamHammingBooleanityOutputClaims {
                    ram_hamming_weight: zero,
                },
                ram_ra_virtualization: stage6::outputs::RamRaVirtualizationOutputClaims {
                    ram_ra: Vec::new(),
                },
                instruction_ra_virtualization:
                    stage6::outputs::InstructionRaVirtualizationOutputClaims {
                        committed_instruction_ra: Vec::new(),
                    },
                inc_claim_reduction: stage6::outputs::IncClaimReductionOutputClaims {
                    ram_inc: zero,
                    rd_inc: zero,
                },
                advice_cycle_phase: stage6::outputs::Stage6AdviceCyclePhaseClaims {
                    trusted: None,
                    untrusted: None,
                },
                bytecode_claim_reduction: None,
                program_image_claim_reduction: None,
            },
            stage7: stage7::outputs::Stage7OutputClaims {
                hamming_weight_claim_reduction:
                    stage7::hamming_weight_claim_reduction::HammingWeightClaimReductionOutputClaims {
                        instruction_ra: Vec::new(),
                        bytecode_ra: Vec::new(),
                        ram_ra: Vec::new(),
                    },
                advice_address_phase:
                    stage7::advice_address_phase::AdviceAddressPhaseOutputClaims {
                        trusted: None,
                        untrusted: None,
                    },
                bytecode_address_phase: None,
                program_image_address_phase: None,
            },
        })
    }

    fn empty_spartan_outer_claims() -> stage1::outputs::SpartanOuterClaims<Fr> {
        let zero = Fr::zero();

        stage1::outputs::SpartanOuterClaims {
            left_instruction_input: zero,
            right_instruction_input: zero,
            product: zero,
            should_branch: zero,
            pc: zero,
            unexpanded_pc: zero,
            imm: zero,
            ram_address: zero,
            rs1_value: zero,
            rs2_value: zero,
            rd_write_value: zero,
            ram_read_value: zero,
            ram_write_value: zero,
            left_lookup_operand: zero,
            right_lookup_operand: zero,
            next_unexpanded_pc: zero,
            next_pc: zero,
            next_is_virtual: zero,
            next_is_first_in_sequence: zero,
            lookup_output: zero,
            should_jump: zero,
            flags: stage1::outputs::SpartanOuterFlagClaims {
                add_operands: zero,
                subtract_operands: zero,
                multiply_operands: zero,
                load: zero,
                store: zero,
                jump: zero,
                write_lookup_output_to_rd: zero,
                virtual_instruction: zero,
                assert: zero,
                do_not_update_unexpanded_pc: zero,
                advice: zero,
                is_compressed: zero,
                is_first_in_sequence: zero,
                is_last_in_sequence: zero,
            },
        }
    }

    fn zk_claims() -> TestClaims {
        JoltProofClaims::Zk {
            blindfold_proof: empty_blindfold_proof(),
        }
    }

    fn empty_blindfold_proof() -> jolt_blindfold::BlindFoldProof<Fr, Bn254G1> {
        jolt_blindfold::BlindFoldProof {
            auxiliary_row_commitments: Vec::new(),
            random_round_commitments: Vec::new(),
            random_output_claim_row_commitments: Vec::new(),
            random_auxiliary_row_commitments: Vec::new(),
            random_error_row_commitments: Vec::new(),
            random_eval_commitments: Vec::new(),
            random_u: Fr::zero(),
            cross_term_error_row_commitments: Vec::new(),
            outer_sumcheck: CompressedSumcheckProof::default(),
            az_rx: Fr::zero(),
            bz_rx: Fr::zero(),
            cz_rx: Fr::zero(),
            inner_sumcheck: CompressedSumcheckProof::default(),
            witness_opening: VectorCommitmentOpening {
                combined_vector: Vec::new(),
                combined_blinding: Fr::zero(),
            },
            error_opening: VectorCommitmentOpening {
                combined_vector: Vec::new(),
                combined_blinding: Fr::zero(),
            },
            folded_eval_outputs: Vec::new(),
            folded_eval_blindings: Vec::new(),
            folded_eval_output_openings: Vec::new(),
            folded_eval_blinding_openings: Vec::new(),
        }
    }

    fn stage_proofs(is_zk: bool) -> JoltStageProofs<Fr, Pedersen<Bn254G1>> {
        JoltStageProofs {
            stage1_uni_skip_first_round_proof: uniskip_proof(is_zk),
            stage1_sumcheck_proof: sumcheck_proof(is_zk),
            stage2_uni_skip_first_round_proof: uniskip_proof(is_zk),
            stage2_sumcheck_proof: sumcheck_proof(is_zk),
            stage3_sumcheck_proof: sumcheck_proof(is_zk),
            stage4_sumcheck_proof: sumcheck_proof(is_zk),
            stage5_sumcheck_proof: sumcheck_proof(is_zk),
            stage6a_sumcheck_proof: sumcheck_proof(is_zk),
            stage6b_sumcheck_proof: sumcheck_proof(is_zk),
            stage7_sumcheck_proof: sumcheck_proof(is_zk),
        }
    }

    fn uniskip_proof(is_zk: bool) -> SumcheckProof<Fr, Bn254G1> {
        if is_zk {
            SumcheckProof::Committed(CommittedSumcheckProof::default())
        } else {
            SumcheckProof::Clear(ClearProof::Full(ClearSumcheckProof::default()))
        }
    }

    fn sumcheck_proof(is_zk: bool) -> SumcheckProof<Fr, Bn254G1> {
        if is_zk {
            SumcheckProof::Committed(CommittedSumcheckProof::default())
        } else {
            SumcheckProof::Clear(ClearProof::Compressed(CompressedSumcheckProof::default()))
        }
    }

    fn test_preprocessing() -> JoltVerifierPreprocessing<TestPcs, Pedersen<Bn254G1>> {
        let memory_layout = common::jolt_device::MemoryLayout::new(&MemoryConfig {
            program_size: Some(1024),
            max_trusted_advice_size: 0,
            max_untrusted_advice_size: 0,
            max_input_size: 8,
            max_output_size: 8,
            stack_size: 8,
            heap_size: 8,
        });
        JoltVerifierPreprocessing::new(
            crate::preprocessing::ProgramPreprocessing::Full(JoltProgramPreprocessing {
                bytecode: BytecodePreprocessing::default(),
                ram: RAMPreprocessing::default(),
                memory_layout,
                max_padded_trace_length: 16,
            }),
            [7; 32],
            (),
            None,
        )
    }
}
