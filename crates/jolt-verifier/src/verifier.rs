//! Top-level verifier entry point.

use common::constants::MAX_BLINDFOLD_GENERATORS;
use common::jolt_device::JoltDevice;
use jolt_claims::protocols::field_inline::FieldInlineRepresentation;
use jolt_claims::protocols::jolt::JoltRelationId;
use jolt_claims::protocols::jolt::{JoltOneHotConfig, JoltReadWriteConfig};
#[cfg(not(feature = "akita"))]
use jolt_crypto::HomomorphicCommitment;
use jolt_crypto::VectorCommitment;
use jolt_field::JoltField;
use jolt_openings::CommitmentScheme;
#[cfg(feature = "akita")]
use jolt_openings::GroupBatchOpening;
#[cfg(not(feature = "akita"))]
use jolt_openings::{AdditivelyHomomorphic, ZkOpeningScheme};
use jolt_program::preprocess::{compute_max_ram_k, compute_min_ram_k};
use jolt_sumcheck::SumcheckProof;
#[cfg(feature = "akita")]
use jolt_transcript::VerifierState;
use jolt_transcript::{
    protocol_tag, BytesMsg, Encoding, FieldTranscript, Transcript, VerifierInit, TAG_BYTES,
};
#[cfg(feature = "akita")]
use std::borrow::BorrowMut;

#[cfg(not(feature = "akita"))]
use crate::proof::JoltCommitments;
#[cfg(feature = "akita")]
use crate::stages::stage8::OneHotTraceCommitmentMetadata;
#[cfg(not(feature = "akita"))]
use crate::stages::stage8::Stage8Output;
use crate::{
    config::{
        validate_proof_config, CommitmentConfig, JoltProtocolConfig, ZkConfig,
        JOLT_VERIFIER_CONFIG, JOLT_VERIFIER_INSTRUCTION_PROFILE, MIN_PADDED_TRACE_LENGTH,
    },
    num,
    preprocessing::JoltVerifierPreprocessing,
    proof::{JoltProof, TracePolynomialOrder},
    stages::{
        stage1, stage2, stage3, stage4, stage5, stage6a, stage6b, stage7, stage8,
        CommittedProgramSchedule, PrecommittedSchedule,
    },
    VerifierError,
};

#[cfg(not(feature = "akita"))]
pub fn verify<'a, F, PCS, VC, T>(
    preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    public_io: &JoltDevice,
    proof: &'a JoltProof<PCS, VC>,
    trusted_advice_commitment: Option<&PCS::Output>,
) -> Result<(), VerifierError>
where
    F: JoltField + Encoding,
    PCS: CommitmentScheme<Field = F>
        + AdditivelyHomomorphic
        + ZkOpeningScheme<HidingCommitment = VC::Output>,
    PCS::Output: Encoding + HomomorphicCommitment<F>,
    VC: VectorCommitment<Field = F>,
    VC::Output: Copy + HomomorphicCommitment<F> + Encoding,
    T: FieldTranscript<F> + VerifierInit<'a>,
{
    use crate::stages::zk::{blindfold, inputs::BlindFoldInputs};

    let (checked, mut transcript) = validate_and_seed_transcript(
        preprocessing,
        public_io,
        proof,
        trusted_advice_commitment,
        |statement| T::init(&JOLT_SESSION, statement, &proof.narg),
    )?;

    // Built once for the whole verification and shared by the stages that read
    // the RA layout (5-8), instead of each rebuilding the same dimensions.
    let formula_dimensions = crate::stages::build_formula_dimensions(
        proof,
        preprocessing,
        &checked,
        num::ilog2(checked.trace_length),
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
    let stage6a = stage6a::verify(
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
    let stage6b = stage6b::verify(
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
        &stage6a,
    )?;
    let stage7 = stage7::verify(
        &checked,
        proof,
        &formula_dimensions,
        &mut transcript,
        &stage4,
        &stage6b,
    )?;
    let stage8 = stage8::verify(
        &checked,
        preprocessing,
        proof,
        &formula_dimensions,
        trusted_advice_commitment,
        &mut transcript,
        &stage6b,
        &stage7,
    )?;

    if checked.zk {
        let blindfold = blindfold::build(BlindFoldInputs {
            checked: &checked,
            preprocessing,
            proof,
            stage1: stage1.zk()?,
            stage2: stage2.zk()?,
            stage3: stage3.zk()?,
            stage4: stage4.zk()?,
            stage5: stage5.zk()?,
            stage6a: stage6a.zk()?,
            stage6b: stage6b.zk()?,
            stage7: stage7.zk()?,
            stage8: stage8.zk()?,
        })?;
        let vc_setup = preprocessing
            .vc_setup
            .as_ref()
            .ok_or(VerifierError::MissingVectorCommitmentSetup)?;
        jolt_verifier_derive::fs_scope_guard!(BlindFold);

        blindfold
            .verify::<VC, T>(proof.blindfold_proof()?, vc_setup, &mut transcript)
            .map_err(|error| VerifierError::BlindFoldVerificationFailed {
                reason: error.to_string(),
            })?;
    } else {
        let Stage8Output::Clear(_) = stage8 else {
            return Err(VerifierError::ExpectedClearProof { field: "stage8" });
        };
    }

    transcript
        .check_eof()
        .map_err(|_| VerifierError::NargNotConsumed)
}

/// The Akita verification path: the same stage spine, with ordered native
/// trace-column claims and one batch opening for the trace, advice,
/// and committed-program objects. No homomorphism bounds and no ZK tail.
#[cfg(feature = "akita")]
pub fn verify<'a, F, PCS, VC, T>(
    preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    public_io: &JoltDevice,
    proof: &'a JoltProof<PCS, VC>,
    trusted_advice_commitment: Option<&PCS::Output>,
) -> Result<(), VerifierError>
where
    F: JoltField + Encoding,
    PCS: CommitmentScheme<Field = F> + GroupBatchOpening,
    PCS::Output: Clone + Encoding + OneHotTraceCommitmentMetadata,
    PCS::VerifierSetup: stage8::OneHotTraceSetupMetadata,
    VC: VectorCommitment<Field = F>,
    VC::Output: Copy + Encoding,
    T: FieldTranscript<F>
        + VerifierInit<'a>
        + BorrowMut<VerifierState<'a, PCS::Sponge>>
        + Into<VerifierState<'a, PCS::Sponge>>,
{
    let (checked, mut transcript) = validate_and_seed_transcript(
        preprocessing,
        public_io,
        proof,
        trusted_advice_commitment,
        |statement| T::init(&JOLT_SESSION, statement, &proof.narg),
    )?;

    // Built once for the whole verification and shared by the stages that read
    // the RA layout (5-8), instead of each rebuilding the same dimensions.
    let formula_dimensions = crate::stages::build_formula_dimensions(
        proof,
        preprocessing,
        &checked,
        num::ilog2(checked.trace_length),
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
    let stage6a = stage6a::verify(
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
    let stage6b = stage6b::verify(
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
        &stage6a,
    )?;
    let stage7 = stage7::verify(
        &checked,
        proof,
        &formula_dimensions,
        &mut transcript,
        &stage4,
        &stage6b,
    )?;
    // Akita writes its argument into the same string; its end-of-proof check
    // is the whole proof's.
    let pending = stage8::verify(
        &checked,
        preprocessing,
        proof,
        &formula_dimensions,
        trusted_advice_commitment,
        &mut transcript,
        &stage4,
        &stage6b,
        &stage7,
    )?;

    PCS::check_eof(pending, transcript.into()).map_err(|_| VerifierError::NargNotConsumed)
}

/// The session every Jolt transcript is bound to: a fixed 32-byte tag.
pub const JOLT_SESSION: [u8; TAG_BYTES] = protocol_tag(b"Jolt");

/// The statement every Jolt transcript is bound to before its first message:
/// the protocol configuration, the preprocessing digest, the public I/O, and
/// the proof's structural parameters, all validated before the transcript is
/// built. Prover and verifier encode it through this one type.
///
/// The encoding is positional: 19 configuration words and the advice presence
/// word, each one `u64` little-endian word, then the 32-byte preprocessing
/// digest, then the inputs and the outputs, each a length word, the bytes and
/// zero padding to a word. Every later absorb whose presence depends on advice
/// is fixed by the presence word.
#[derive(Clone, Copy, Debug)]
pub struct JoltStatement<'a> {
    pub checked: &'a CheckedInputs,
    pub config: ProofTranscriptConfig,
}

impl Encoding for JoltStatement<'_> {
    fn encode(&self) -> impl AsRef<[u8]> {
        let checked = self.checked;
        let public_io = &checked.public_io;
        let protocol = JoltProtocolConfig::for_zk(checked.zk);
        let mut out = Vec::new();
        let mut word = |value: u64| out.extend_from_slice(&value.to_le_bytes());
        word(match protocol.zk {
            ZkConfig::Transparent => 0,
            ZkConfig::BlindFold => 1,
        });
        word(match protocol.commitment {
            CommitmentConfig::Homomorphic => 0,
            CommitmentConfig::Akita => 1,
        });
        word(u64::from(protocol.field_inline.enabled));
        word(num::u64_from_usize(
            protocol.field_inline.field_register_log_k,
        ));
        word(match protocol.field_inline.representation {
            FieldInlineRepresentation::NativeFieldElement => 0,
        });
        word(public_io.memory_layout.max_input_size);
        word(public_io.memory_layout.max_output_size);
        word(public_io.memory_layout.heap_size);
        word(u64::from(public_io.panic));
        word(num::u64_from_usize(checked.ram_K));
        word(num::u64_from_usize(checked.trace_length));
        word(checked.entry_address);
        let rw = self.config.rw_config;
        word(u64::from(rw.ram_rw_phase1_num_rounds));
        word(u64::from(rw.ram_rw_phase2_num_rounds));
        word(u64::from(rw.registers_rw_phase1_num_rounds));
        word(u64::from(rw.registers_rw_phase2_num_rounds));
        let one_hot = self.config.one_hot_config;
        word(u64::from(one_hot.log_k_chunk));
        word(u64::from(one_hot.lookups_ra_virtual_log_k_chunk));
        word(self.config.trace_polynomial_order.transcript_scalar());
        word(advice_presence(checked));
        out.extend_from_slice(&checked.preprocessing_digest);
        for bytes in [&public_io.inputs, &public_io.outputs] {
            let message = BytesMsg(bytes.as_slice());
            out.extend_from_slice(message.encode().as_ref());
        }
        out
    }
}

/// The advice presence word: bit 0 when the proof carries an untrusted-advice
/// commitment, bit 1 when the verifier was given a trusted-advice commitment.
/// It is computed on every build, so the advice commitments that follow are
/// fixed by bound data.
fn advice_presence(checked: &CheckedInputs) -> u64 {
    u64::from(checked.untrusted_advice_commitment_present)
        | (u64::from(checked.trusted_advice_commitment_present) << 1)
}

/// Validates the inputs and proof shape, builds the Fiat-Shamir transcript
/// with `new_transcript` from the [`JoltStatement`], then absorbs the
/// commitments. Every consumer of the staged verification — [`verify`] on
/// both builds, the ZK prover's replay and the zk audit harness — starts here,
/// so there is exactly one absorption order.
pub fn validate_and_seed_transcript<PCS, VC, T, ZkProof>(
    preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    public_io: &JoltDevice,
    proof: &JoltProof<PCS, VC, ZkProof>,
    trusted_advice_commitment: Option<&PCS::Output>,
    new_transcript: impl FnOnce(&JoltStatement<'_>) -> T,
) -> Result<(CheckedInputs, T), VerifierError>
where
    PCS: CommitmentScheme,
    PCS::Output: Encoding,
    VC: VectorCommitment<Field = PCS::Field>,
    VC::Output: Encoding,
    T: Transcript,
{
    validate_proof_config(&JOLT_VERIFIER_CONFIG, proof.protocol)?;
    validate_proof_consistency(proof, JOLT_VERIFIER_CONFIG.zk == ZkConfig::BlindFold)?;
    let checked = validate_inputs(
        preprocessing,
        public_io,
        proof,
        trusted_advice_commitment.is_some(),
    )?;

    let mut transcript = new_transcript(&JoltStatement {
        checked: &checked,
        config: ProofTranscriptConfig {
            rw_config: proof.rw_config,
            one_hot_config: proof.one_hot_config,
            trace_polynomial_order: proof.trace_polynomial_order,
        },
    });
    absorb_commitments(
        preprocessing,
        proof,
        trusted_advice_commitment,
        &mut transcript,
    );

    Ok((checked, transcript))
}

#[expect(non_snake_case, reason = "Preserves the deployed proof field name.")]
#[derive(Clone, Debug, PartialEq)]
pub struct CheckedInputs {
    pub public_io: JoltDevice,
    pub zk: bool,
    pub trace_length: usize,
    pub ram_K: usize,
    pub entry_address: u64,
    pub preprocessing_digest: [u8; 32],
    pub untrusted_advice_commitment_present: bool,
    pub trusted_advice_commitment_present: bool,
    pub vc_capacity: Option<usize>,
    pub precommitted: PrecommittedSchedule,
}

pub fn validate_inputs<PCS, VC, ZkProof>(
    preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    public_io: &JoltDevice,
    proof: &JoltProof<PCS, VC, ZkProof>,
    trusted_advice_commitment_present: bool,
) -> Result<CheckedInputs, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    let trace_length = proof.trace_length;
    let ram_k = proof.ram_K;
    let trace_polynomial_order = proof.trace_polynomial_order;
    let one_hot_config = proof.one_hot_config;
    let untrusted_advice_commitment_present = proof.untrusted_advice_commitment.is_some();
    let zk = matches!(JOLT_VERIFIER_CONFIG.zk, ZkConfig::BlindFold);
    let vc_capacity = if zk {
        Some(validate_zk_vector_commitment_setup::<PCS, VC>(
            preprocessing,
        )?)
    } else {
        None
    };
    let program = &preprocessing.program;
    let memory_layout = program.memory_layout();
    if &public_io.memory_layout != memory_layout {
        return Err(VerifierError::MemoryLayoutMismatch);
    }
    validate_ram_remap_base(memory_layout)?;

    if num::u64_from_usize(public_io.inputs.len()) > memory_layout.max_input_size {
        return Err(VerifierError::InputTooLarge {
            got: public_io.inputs.len(),
            // The failed comparison bounds the maximum below a usize length.
            max: usize::try_from(memory_layout.max_input_size).unwrap_or(usize::MAX),
        });
    }

    if num::u64_from_usize(public_io.outputs.len()) > memory_layout.max_output_size {
        return Err(VerifierError::OutputTooLarge {
            got: public_io.outputs.len(),
            // The failed comparison bounds the maximum below a usize length.
            max: usize::try_from(memory_layout.max_output_size).unwrap_or(usize::MAX),
        });
    }

    validate_trace_length(trace_length, program.max_padded_trace_length())?;

    let min_ram_k = compute_min_ram_k(
        program.min_bytecode_address(),
        program.program_image_len_words(),
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

    // This build proves exactly one instruction profile. A full program carrying an
    // instruction the build has no constraints for (a field-inline bridge row on a verifier
    // without field-inline, whose rd write no RV64 row pins) rejects here rather than
    // verifying against the base rows alone. Committed programs carry no rows to scan; with
    // field-inline enabled, the full-program requirement below rejects them.
    if let Some(full) = program.as_full() {
        if let Some(row) = full
            .bytecode
            .bytecode
            .iter()
            .find(|row| !JOLT_VERIFIER_INSTRUCTION_PROFILE.supports_jolt(row.instruction_kind))
        {
            return Err(VerifierError::UnsupportedInstruction {
                kind: row.instruction_kind,
            });
        }
    }
    #[cfg(feature = "field-inline")]
    {
        let full = program
            .as_full()
            .ok_or_else(stage6b::field_inline::committed_program_rejection)?;
        for row in &full.bytecode.bytecode {
            jolt_program::field_inline::validate_field_inline_instruction(row).map_err(
                |error| VerifierError::InvalidFieldInlineBytecode {
                    reason: error.to_string(),
                },
            )?;
        }
    }

    let mut normalized_public_io = public_io.clone();
    normalized_public_io.outputs.truncate(
        normalized_public_io
            .outputs
            .iter()
            .rposition(|&byte| byte != 0)
            .map_or(0, |position| position.saturating_add(1)),
    );

    let committed_program =
        program
            .committed()
            .map(|committed| {
                if committed.trace_order != trace_polynomial_order {
                    return Err(VerifierError::InvalidCommittedProgram {
                        reason: "committed-program trace order disagrees with the proof".to_owned(),
                    });
                }
                let meta = &committed.meta;
                let program_image_start_index = memory_layout
                    .remapped_word_address(meta.min_bytecode_address)
                    .map_err(|error| VerifierError::InvalidCommittedProgram {
                        reason: error.to_string(),
                    })?;
                if meta.entry_bytecode_index >= meta.bytecode_len {
                    return Err(VerifierError::InvalidCommittedProgram {
                        reason: format!(
                            "entry bytecode index {} is out of range for bytecode length {}",
                            meta.entry_bytecode_index, meta.bytecode_len
                        ),
                    });
                }
                let program_image_start_index = usize::try_from(program_image_start_index)
                    .map_err(|_| VerifierError::InvalidCommittedProgram {
                        reason: format!(
                        "program image start index {program_image_start_index} does not fit usize"
                    ),
                    })?;
                Ok(CommittedProgramSchedule {
                    bytecode_len: meta.bytecode_len,
                    program_image_len_words: meta.program_image_len_words,
                    program_image_start_index,
                })
            })
            .transpose()?;
    #[cfg(not(feature = "akita"))]
    let trusted_advice_size = trusted_advice_commitment_present
        .then(|| advice_size_to_usize(memory_layout.max_trusted_advice_size, "trusted"))
        .transpose()?;
    #[cfg(not(feature = "akita"))]
    let untrusted_advice_size = untrusted_advice_commitment_present
        .then(|| advice_size_to_usize(memory_layout.max_untrusted_advice_size, "untrusted"))
        .transpose()?;
    let precommitted = PrecommittedSchedule::new(
        trace_polynomial_order,
        num::ilog2(trace_length),
        one_hot_config.committed_chunk_bits(),
        #[cfg(not(feature = "akita"))]
        trusted_advice_size,
        #[cfg(not(feature = "akita"))]
        untrusted_advice_size,
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
        entry_address: program.entry_address(),
        preprocessing_digest: preprocessing.preprocessing_digest,
        untrusted_advice_commitment_present,
        trusted_advice_commitment_present,
        vc_capacity,
        precommitted,
    })
}

#[cfg(not(feature = "akita"))]
fn advice_size_to_usize(value: u64, kind: &'static str) -> Result<usize, VerifierError> {
    usize::try_from(value).map_err(|_| VerifierError::InvalidMemoryLayout {
        reason: format!("maximum {kind} advice size {value} does not fit usize"),
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
    let required = MAX_BLINDFOLD_GENERATORS;
    let got = VC::capacity(setup);
    if got < required {
        return Err(VerifierError::InvalidVectorCommitmentCapacity { required, got });
    }

    Ok(got)
}

pub fn validate_proof_consistency<PCS, VC, ZkProof>(
    proof: &JoltProof<PCS, VC, ZkProof>,
    zk: bool,
) -> Result<(), VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    // A build with field-inline enabled proves every guest under the composed protocol, so the
    // field-inline committed payload is unconditionally required (absence means a producer
    // without field-inline semantics — reject before any stage logic). On the Akita axis the
    // field-increment commitment slot is equally unconditional: presence is never claim-gated
    // (an all-zero group still commits).
    #[cfg(all(feature = "field-inline", not(feature = "akita")))]
    if proof.commitments.field_inline.is_none() {
        return Err(VerifierError::MissingProofPayload {
            field: "commitments.field_inline",
        });
    }
    #[cfg(all(feature = "field-inline", feature = "akita"))]
    if proof.field_inc_commitment.is_none() {
        return Err(VerifierError::MissingProofPayload {
            field: "field_inc_commitment",
        });
    }

    #[cfg(feature = "akita")]
    if proof.fused_inc_commitment.is_none() {
        return Err(VerifierError::MissingProofPayload {
            field: "fused_inc_commitment",
        });
    }

    let stage_proofs = [
        (
            &proof.stages.stage1_uni_skip_first_round_proof,
            "stage1_uni_skip_first_round_proof",
        ),
        (&proof.stages.stage1_sumcheck_proof, "stage1_sumcheck_proof"),
        (
            &proof.stages.stage2_uni_skip_first_round_proof,
            "stage2_uni_skip_first_round_proof",
        ),
        (&proof.stages.stage2_sumcheck_proof, "stage2_sumcheck_proof"),
        (&proof.stages.stage3_sumcheck_proof, "stage3_sumcheck_proof"),
        (&proof.stages.stage4_sumcheck_proof, "stage4_sumcheck_proof"),
        (&proof.stages.stage5_sumcheck_proof, "stage5_sumcheck_proof"),
        (
            &proof.stages.stage6a_sumcheck_proof,
            "stage6a_sumcheck_proof",
        ),
        (
            &proof.stages.stage6b_sumcheck_proof,
            "stage6b_sumcheck_proof",
        ),
        (&proof.stages.stage7_sumcheck_proof, "stage7_sumcheck_proof"),
    ];
    for (stage_proof, field) in stage_proofs {
        validate_sumcheck_representation(stage_proof, field, zk)?;
    }
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

pub(crate) fn validate_sumcheck_representation<F, RoundCommitment>(
    proof: &SumcheckProof<F, RoundCommitment>,
    field: &'static str,
    zk: bool,
) -> Result<(), VerifierError>
where
    F: JoltField,
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

/// Absorbs the commitments in the consensus-critical order — the proof-carried
/// polynomial commitments, the optional advice commitments, then the committed
/// program's preprocessing-held commitments. WARNING: the prover must absorb
/// identically or the transcripts diverge. On the `akita` build the order is
/// the canonical commitment-object order: `OneHotTrace`, untrusted advice, trusted
/// advice, and direct program objects.
#[jolt_verifier_derive::fs_scope(Commitments)]
pub(crate) fn absorb_commitments<PCS, VC, ZkProof, T>(
    preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    proof: &JoltProof<PCS, VC, ZkProof>,
    trusted_advice_commitment: Option<&PCS::Output>,
    transcript: &mut T,
) where
    PCS: CommitmentScheme,
    PCS::Output: Encoding,
    VC: VectorCommitment<Field = PCS::Field>,
    T: Transcript,
{
    #[cfg(not(feature = "akita"))]
    {
        absorb_transcript_commitments(
            &proof.commitments,
            proof.untrusted_advice_commitment.as_ref(),
            trusted_advice_commitment,
            transcript,
        );
        if let Some(committed) = preprocessing.program.committed() {
            absorb_committed_program_commitments(
                &committed.bytecode_commitment,
                &committed.program_image_commitment,
                transcript,
            );
        }
    }
    #[cfg(feature = "akita")]
    absorb_akita_commitments(
        &proof.commitments,
        proof.fused_inc_commitment.as_ref(),
        proof.untrusted_advice_commitment.as_ref(),
        trusted_advice_commitment,
        #[cfg(feature = "field-inline")]
        proof.field_inc_commitment.as_ref(),
        preprocessing.program.committed().map(|committed| {
            (
                &committed.bytecode_commitment,
                &committed.program_image_commitment,
            )
        }),
        transcript,
    );
}

/// Absorbs the Akita commitment objects in canonical object order: `OneHotTrace`, `FusedInc`,
/// untrusted advice, trusted advice, the field-increment commitment (field-inline builds), then the whole
/// bytecode and program image. Shared verbatim by the Akita prover's stage 0.
#[cfg(feature = "akita")]
pub fn absorb_akita_commitments<C, T>(
    one_hot_trace: &C,
    fused_inc_commitment: Option<&C>,
    untrusted_advice_commitment: Option<&C>,
    trusted_advice_commitment: Option<&C>,
    #[cfg(feature = "field-inline")] field_inc_commitment: Option<&C>,
    program_commitments: Option<(&C, &C)>,
    transcript: &mut T,
) where
    C: Encoding,
    T: Transcript,
{
    transcript.append(one_hot_trace);
    if let Some(commitment) = fused_inc_commitment {
        transcript.append(commitment);
    }
    if let Some(commitment) = untrusted_advice_commitment {
        transcript.append(commitment);
    }
    if let Some(commitment) = trusted_advice_commitment {
        transcript.append(commitment);
    }
    #[cfg(feature = "field-inline")]
    if let Some(commitment) = field_inc_commitment {
        transcript.append(commitment);
    }
    if let Some((bytecode, image)) = program_commitments {
        absorb_akita_program_commitments(bytecode, image, transcript);
    }
}

#[cfg(feature = "akita")]
pub fn absorb_akita_program_commitments<C, T>(bytecode: &C, image: &C, transcript: &mut T)
where
    C: Encoding,
    T: Transcript,
{
    transcript.append(bytecode);
    transcript.append(image);
}

/// Absorbs the preprocessing-held committed-program commitments (whole
/// bytecode, then the program image), immediately after the proof-carried
/// commitments. Shared verbatim by the prover's stage 0.
pub fn absorb_committed_program_commitments<C, T>(
    bytecode_commitment: &C,
    program_image_commitment: &C,
    transcript: &mut T,
) where
    C: Encoding,
    T: Transcript,
{
    transcript.append(bytecode_commitment);
    transcript.append(program_image_commitment);
}

/// Absorbs the proof-derived polynomial commitments (increment, one-hot `ra`,
/// and optional advice commitments) in the consensus-critical order. WARNING:
/// this covers only the commitments carried by the proof itself; committed
/// program-image commitments live in the preprocessing and are absorbed
/// separately by `absorb_commitments` immediately after this call.
#[cfg(not(feature = "akita"))]
pub fn absorb_transcript_commitments<C, T>(
    commitments: &JoltCommitments<C>,
    untrusted_advice_commitment: Option<&C>,
    trusted_advice_commitment: Option<&C>,
    transcript: &mut T,
) where
    C: Encoding,
    T: Transcript,
{
    let mut absorb_commitment = |commitment: &C| transcript.append(commitment);
    absorb_commitment(&commitments.rd_inc);
    absorb_commitment(&commitments.ram_inc);
    for commitment in &commitments.instruction_ra {
        absorb_commitment(commitment);
    }
    for commitment in &commitments.ram_ra {
        absorb_commitment(commitment);
    }
    for commitment in &commitments.bytecode_ra {
        absorb_commitment(commitment);
    }
    // The field-inline payload absorbs after the base commitments and before
    // advice, mirroring its appended-extension position everywhere else in the
    // protocol. The prover's stage 0 must commit in this same order.
    #[cfg(feature = "field-inline")]
    if let Some(field_inline) = &commitments.field_inline {
        transcript.append(&field_inline.field_registers.rd_inc);
    }
    if let Some(untrusted_advice_commitment) = untrusted_advice_commitment {
        transcript.append(untrusted_advice_commitment);
    }
    if let Some(trusted_advice_commitment) = trusted_advice_commitment {
        transcript.append(trusted_advice_commitment);
    }
}

/// The proof's structural parameters in the [`JoltStatement`]. The modular
/// prover builds it from its configuration, before it has a finished
/// [`JoltProof`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProofTranscriptConfig {
    pub rw_config: JoltReadWriteConfig,
    pub one_hot_config: JoltOneHotConfig,
    pub trace_polynomial_order: TracePolynomialOrder,
}

fn validate_trace_length(
    trace_length: usize,
    max_padded_trace_length: usize,
) -> Result<(), VerifierError> {
    if !trace_length.is_power_of_two()
        || trace_length < MIN_PADDED_TRACE_LENGTH
        || trace_length > max_padded_trace_length
    {
        return Err(VerifierError::InvalidTraceLength {
            got: trace_length,
            min: MIN_PADDED_TRACE_LENGTH,
            max: max_padded_trace_length,
        });
    }
    Ok(())
}

/// Fail closed on a RAM remap base that RAF evaluation cannot rely on. Stage
/// 2's RAF-evaluation unmap is `8k + lowest_address`:
/// - `lowest_address > 8` keeps `unmap(0) ≠ 0`, which the lattice digit-zero
///   reconstruction relies on to distinguish "no RAM access" from an access
///   at remapped word zero;
/// - `lowest_address ≡ 0 (mod 8)` makes every admissible `RamAddress` a
///   multiple of 8, which is how RAF evaluation rejects a memory access
///   misaligned for its width (see R1CS constraint 0 in `jolt-claims`).
fn validate_ram_remap_base(
    memory_layout: &common::jolt_device::MemoryLayout,
) -> Result<(), VerifierError> {
    let lowest_address = memory_layout.get_lowest_address();
    if lowest_address <= 8 {
        return Err(VerifierError::InvalidMemoryLayout {
            reason: format!(
                "lowest remapped RAM address {lowest_address:#x} must exceed 8 so the RAF \
                 unmap constant stays clear of the null-address range"
            ),
        });
    }
    if !lowest_address.is_multiple_of(8) {
        return Err(VerifierError::InvalidMemoryLayout {
            reason: format!(
                "lowest remapped RAM address {lowest_address:#x} must be doubleword-aligned so \
                 RAF evaluation rejects misaligned memory accesses"
            ),
        });
    }
    Ok(())
}

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
    validate_ram_remap_base(memory_layout)?;

    if num::u64_from_usize(public_io.inputs.len()) > memory_layout.max_input_size {
        return Err(VerifierError::InputTooLarge {
            got: public_io.inputs.len(),
            // The failed comparison bounds the maximum below a usize length.
            max: usize::try_from(memory_layout.max_input_size).unwrap_or(usize::MAX),
        });
    }

    if num::u64_from_usize(public_io.outputs.len()) > memory_layout.max_output_size {
        return Err(VerifierError::OutputTooLarge {
            got: public_io.outputs.len(),
            // The failed comparison bounds the maximum below a usize length.
            max: usize::try_from(memory_layout.max_output_size).unwrap_or(usize::MAX),
        });
    }

    validate_trace_length(
        trace_length,
        preprocessing.program.max_padded_trace_length(),
    )?;

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
            .map_or(0, |position| position.saturating_add(1)),
    );

    let committed_program =
        preprocessing
            .program
            .committed()
            .map(|committed| {
                if committed.trace_order != trace_polynomial_order {
                    return Err(VerifierError::InvalidCommittedProgram {
                        reason: "committed-program trace order disagrees with the proof".to_owned(),
                    });
                }
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
                let program_image_start_index = usize::try_from(program_image_start_index)
                    .map_err(|_| VerifierError::InvalidCommittedProgram {
                        reason: format!(
                        "program image start index {program_image_start_index} does not fit usize"
                    ),
                    })?;
                Ok(CommittedProgramSchedule {
                    bytecode_len: committed.meta.bytecode_len,
                    program_image_len_words: committed.meta.program_image_len_words,
                    program_image_start_index,
                })
            })
            .transpose()?;
    #[cfg(not(feature = "akita"))]
    let trusted_advice_size = trusted_advice_commitment_present
        .then(|| advice_size_to_usize(memory_layout.max_trusted_advice_size, "trusted"))
        .transpose()?;
    #[cfg(not(feature = "akita"))]
    let untrusted_advice_size = untrusted_advice_commitment_present
        .then(|| advice_size_to_usize(memory_layout.max_untrusted_advice_size, "untrusted"))
        .transpose()?;
    let precommitted = PrecommittedSchedule::new(
        trace_polynomial_order,
        num::ilog2(trace_length),
        one_hot_config.committed_chunk_bits(),
        #[cfg(not(feature = "akita"))]
        trusted_advice_size,
        #[cfg(not(feature = "akita"))]
        untrusted_advice_size,
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
        untrusted_advice_commitment_present,
        trusted_advice_commitment_present,
        vc_capacity,
        precommitted,
    })
}

#[cfg(test)]
#[cfg_attr(
    not(feature = "field-inline"),
    expect(
        clippy::useless_conversion,
        reason = "field-inline selects composed claim and opening types"
    )
)]
mod tests {
    use jolt_claims::protocols::jolt::relations::booleanity::BooleanityOutputClaims;
    use std::sync::Arc;

    use common::constants::RAM_START_ADDRESS;

    use super::*;
    #[cfg(not(feature = "akita"))]
    use crate::proof::JoltCommitments;
    use crate::proof::{ClearProofClaims, JoltProofClaims, JoltStageProofs};
    #[cfg(all(not(feature = "akita"), feature = "field-inline"))]
    use crate::proof::{FieldInlineCommitments, FieldRegistersCommitments};
    use crate::stages::stage1::outputs::Stage1OutputClaims;
    use crate::stages::stage1::OuterRemainderOutputClaims;
    use crate::stages::stage2::outputs::{Stage2BatchOutputClaims, Stage2OutputClaims};
    #[cfg(feature = "field-inline")]
    use crate::stages::{
        stage2::outputs::FieldRegistersClaimReductionOutputClaims,
        stage4::FieldRegistersReadWriteOutputClaims,
        stage5::FieldRegistersValEvaluationOutputClaims,
        stage6b::outputs::FieldRegistersIncClaimReductionOutputClaims,
    };
    #[cfg(any(feature = "field-inline", feature = "zk"))]
    use common::constants::MAX_BLINDFOLD_GENERATORS;
    use common::jolt_device::{JoltDevice, MemoryConfig};
    use jolt_claims::protocols::jolt::{JoltOneHotConfig, JoltReadWriteConfig};
    #[cfg(feature = "zk")]
    use jolt_crypto::PedersenSetup;
    use jolt_crypto::{Bn254G1, Commitment, Pedersen, VectorCommitmentOpening};
    use jolt_field::Fr;
    use jolt_openings::{CommitmentScheme, OpeningsError};
    use jolt_poly::MultilinearPoly;
    use jolt_program::preprocess::JoltProgramPreprocessing;
    use jolt_sumcheck::{
        ClearProof, ClearSumcheckProof, CommittedSumcheckProof, CompressedSumcheckProof,
    };
    use jolt_transcript::FieldTranscript;
    use num_traits::Zero;

    use crate::preprocessing::ProgramPreprocessing;

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
            _transcript: &mut impl FieldTranscript<Self::Field>,
        ) -> Result<Self::Proof, OpeningsError> {
            Ok(())
        }

        fn verify(
            _commitment: &Self::Output,
            _point: &[Self::Field],
            _eval: Self::Field,
            _proof: &Self::Proof,
            _setup: &Self::VerifierSetup,
            _transcript: &mut impl FieldTranscript<Self::Field>,
        ) -> Result<(), OpeningsError> {
            Ok(())
        }
    }

    impl Encoding for TestCommitment {
        fn encode(&self) -> impl AsRef<[u8]> {
            [0u8; 0]
        }
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
    fn protocol_and_payload_reject_before_preprocessing_validation() {
        #[cfg(feature = "field-inline")]
        use jolt_riscv::JoltInstructionKind;
        use jolt_transcript::{verifier_transcript, Keccak};
        #[cfg_attr(not(feature = "field-inline"), expect(unused_mut))]
        let mut preprocessing = test_preprocessing();
        #[cfg(feature = "field-inline")]
        if let ProgramPreprocessing::Full(full) = &mut preprocessing.program {
            let bytecode = &mut Arc::make_mut(full).bytecode.bytecode;
            if let Some(instruction) = bytecode.first_mut() {
                instruction.instruction_kind = JoltInstructionKind::FIELD_ADD;
                instruction.operands.rs1 = Some(u8::MAX);
            }
        }
        let is_zk = JOLT_VERIFIER_CONFIG.zk == ZkConfig::BlindFold;
        let mut proof = proof_with_zk(is_zk, if is_zk { zk_claims() } else { clear_claims() });
        proof.protocol.zk = if is_zk {
            ZkConfig::Transparent
        } else {
            ZkConfig::BlindFold
        };
        let public_io = JoltDevice::default();
        assert!(matches!(
            validate_and_seed_transcript(&preprocessing, &public_io, &proof, None, |statement| {
                verifier_transcript::<Keccak>(&JOLT_SESSION, statement, &[])
            }),
            Err(VerifierError::ProtocolConfigMismatch { .. })
        ));
        proof.protocol = JOLT_VERIFIER_CONFIG;
        proof.stages.stage1_sumcheck_proof = sumcheck_proof(!is_zk);
        assert!(matches!(
            validate_and_seed_transcript(&preprocessing, &public_io, &proof, None, |statement| {
                verifier_transcript::<Keccak>(&JOLT_SESSION, statement, &[])
            }),
            Err(VerifierError::ExpectedClearProof {
                field: "stage1_sumcheck_proof"
            } | VerifierError::ExpectedCommittedProof {
                field: "stage1_sumcheck_proof"
            })
        ));
    }

    /// The statement the transcript is bound to changes with every field of
    /// it the verifier reads, except the fields the verifier binds another
    /// way. Proof-body fields are absorbed as messages where they are read.
    #[test]
    #[expect(
        clippy::unwrap_used,
        clippy::indexing_slicing,
        reason = "the test reads fixed offsets of a statement whose length it asserts first"
    )]
    fn statement_binds_each_field_the_verifier_reads() {
        type Edit = fn(&mut CheckedInputs, &mut ProofTranscriptConfig);

        let preprocessing = test_preprocessing();
        let public_io = JoltDevice {
            memory_layout: preprocessing.program.memory_layout().clone(),
            inputs: vec![1, 2],
            outputs: vec![3],
            ..JoltDevice::default()
        };
        let is_zk = JOLT_VERIFIER_CONFIG.zk == ZkConfig::BlindFold;
        let proof = proof_with_zk(is_zk, if is_zk { zk_claims() } else { clear_claims() });
        let checked = validate_inputs(&preprocessing, &public_io, &proof, false).unwrap();
        let config = ProofTranscriptConfig {
            rw_config: proof.rw_config,
            one_hot_config: proof.one_hot_config,
            trace_polynomial_order: proof.trace_polynomial_order,
        };
        let encode = |edit: Edit| {
            let (mut checked, mut config) = (checked.clone(), config);
            edit(&mut checked, &mut config);
            let statement = JoltStatement {
                checked: &checked,
                config,
            };
            let bytes = statement.encode().as_ref().to_vec();
            bytes
        };
        let honest = encode(|_, _| {});

        // The positional layout: 19 configuration words, the advice presence
        // word, the bare 32-byte digest, then the inputs `[1, 2]` and the
        // outputs `[3]`, each a length word, the bytes and padding to a word.
        let word =
            |index: usize| u64::from_le_bytes(honest[8 * index..8 * index + 8].try_into().unwrap());
        assert_eq!(honest.len(), 20 * 8 + 32 + 2 * 16);
        assert_eq!(word(0), u64::from(is_zk), "zk is the first word");
        assert_eq!(word(19), 0, "no advice is present");
        assert_eq!(&honest[160..192], &checked.preprocessing_digest);
        assert_eq!(
            &honest[192..208],
            &[2, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0]
        );
        assert_eq!(
            &honest[208..224],
            &[1, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0]
        );
        let presence = |untrusted: bool, trusted: bool| {
            let mut checked = checked.clone();
            checked.untrusted_advice_commitment_present = untrusted;
            checked.trusted_advice_commitment_present = trusted;
            let statement = JoltStatement {
                checked: &checked,
                config,
            };
            let bytes = statement.encode().as_ref().to_vec();
            u64::from_le_bytes(bytes[152..160].try_into().unwrap())
        };
        assert_eq!(presence(true, false), 1);
        assert_eq!(presence(false, true), 2);
        assert_eq!(presence(true, true), 3);

        let bound: &[(&str, Edit)] = &[
            ("zk", |c, _| c.zk = !c.zk),
            ("max_input_size", |c, _| {
                c.public_io.memory_layout.max_input_size += 1;
            }),
            ("max_output_size", |c, _| {
                c.public_io.memory_layout.max_output_size += 1;
            }),
            ("heap_size", |c, _| c.public_io.memory_layout.heap_size += 1),
            ("panic", |c, _| c.public_io.panic = !c.public_io.panic),
            ("inputs", |c, _| c.public_io.inputs.push(0)),
            ("outputs", |c, _| c.public_io.outputs.push(1)),
            ("trace_length", |c, _| c.trace_length *= 2),
            ("ram_K", |c, _| c.ram_K *= 2),
            ("entry_address", |c, _| c.entry_address += 4),
            ("preprocessing_digest", |c, _| {
                c.preprocessing_digest[0] ^= 1;
            }),
            ("ram_rw_phase1_num_rounds", |_, k| {
                k.rw_config.ram_rw_phase1_num_rounds += 1;
            }),
            ("ram_rw_phase2_num_rounds", |_, k| {
                k.rw_config.ram_rw_phase2_num_rounds += 1;
            }),
            ("registers_rw_phase1_num_rounds", |_, k| {
                k.rw_config.registers_rw_phase1_num_rounds += 1;
            }),
            ("registers_rw_phase2_num_rounds", |_, k| {
                k.rw_config.registers_rw_phase2_num_rounds += 1;
            }),
            ("log_k_chunk", |_, k| k.one_hot_config.log_k_chunk += 1),
            ("lookups_ra_virtual_log_k_chunk", |_, k| {
                k.one_hot_config.lookups_ra_virtual_log_k_chunk += 1;
            }),
            ("trace_polynomial_order", |_, k| {
                k.trace_polynomial_order = match k.trace_polynomial_order {
                    TracePolynomialOrder::CycleMajor => TracePolynomialOrder::AddressMajor,
                    TracePolynomialOrder::AddressMajor => TracePolynomialOrder::CycleMajor,
                };
            }),
            // The advice commitments are absorbed without labels, so their
            // presence is bound by the presence word on every build.
            ("untrusted_advice_commitment_present", |c, _| {
                c.untrusted_advice_commitment_present = !c.untrusted_advice_commitment_present;
            }),
            ("trusted_advice_commitment_present", |c, _| {
                c.trusted_advice_commitment_present = !c.trusted_advice_commitment_present;
            }),
        ];
        for (field, edit) in bound {
            assert_ne!(encode(*edit), honest, "the statement does not bind {field}");
        }

        let unbound: &[(&str, Edit)] = &[
            // `validate_inputs` rejects a memory layout other than the
            // preprocessing's, and the preprocessing digest binds that one.
            ("program_size", |c, _| {
                c.public_io.memory_layout.program_size += 1;
            }),
            ("max_trusted_advice_size", |c, _| {
                c.public_io.memory_layout.max_trusted_advice_size += 1;
            }),
            ("trusted_advice_start", |c, _| {
                c.public_io.memory_layout.trusted_advice_start += 1;
            }),
            ("trusted_advice_end", |c, _| {
                c.public_io.memory_layout.trusted_advice_end += 1;
            }),
            ("max_untrusted_advice_size", |c, _| {
                c.public_io.memory_layout.max_untrusted_advice_size += 1;
            }),
            ("untrusted_advice_start", |c, _| {
                c.public_io.memory_layout.untrusted_advice_start += 1;
            }),
            ("untrusted_advice_end", |c, _| {
                c.public_io.memory_layout.untrusted_advice_end += 1;
            }),
            ("input_start", |c, _| {
                c.public_io.memory_layout.input_start += 1;
            }),
            ("input_end", |c, _| c.public_io.memory_layout.input_end += 1),
            ("output_start", |c, _| {
                c.public_io.memory_layout.output_start += 1;
            }),
            ("output_end", |c, _| {
                c.public_io.memory_layout.output_end += 1;
            }),
            ("stack_size", |c, _| {
                c.public_io.memory_layout.stack_size += 1;
            }),
            ("stack_end", |c, _| c.public_io.memory_layout.stack_end += 1),
            ("heap_end", |c, _| c.public_io.memory_layout.heap_end += 1),
            ("panic address", |c, _| c.public_io.memory_layout.panic += 1),
            ("termination", |c, _| {
                c.public_io.memory_layout.termination += 1;
            }),
            ("io_end", |c, _| c.public_io.memory_layout.io_end += 1),
            // The capacity of the verifier's own vector-commitment setup.
            ("vc_capacity", |c, _| {
                c.vc_capacity = Some(c.vc_capacity.unwrap_or(0) + 1);
            }),
            // The advice bytes themselves: the verifier reads only their
            // commitments.
            ("trusted_advice", |c, _| c.public_io.trusted_advice.push(1)),
            ("untrusted_advice", |c, _| {
                c.public_io.untrusted_advice.push(1);
            }),
            // Derived during validation from the trace order, trace length,
            // chunk bits and advice presence, which are bound, and from the
            // committed program, which the preprocessing digest binds.
            ("precommitted", |c, _| {
                c.precommitted = PrecommittedSchedule::new(
                    TracePolynomialOrder::CycleMajor,
                    4,
                    4,
                    #[cfg(not(feature = "akita"))]
                    None,
                    #[cfg(not(feature = "akita"))]
                    None,
                    Some(CommittedProgramSchedule {
                        bytecode_len: 16,
                        program_image_len_words: 16,
                        program_image_start_index: 0,
                    }),
                )
                .unwrap();
            }),
        ];
        for (field, edit) in unbound {
            let (mut edited, mut edited_config) = (checked.clone(), config);
            edit(&mut edited, &mut edited_config);
            assert_ne!(edited, checked, "the {field} edit changes nothing");
            assert_eq!(
                encode(*edit),
                honest,
                "the statement now binds {field}; move it to the bound list"
            );
        }
    }

    #[test]
    #[expect(clippy::unwrap_used)]
    fn validate_inputs_normalizes_public_output() {
        let preprocessing = test_preprocessing();
        let mut public_io = JoltDevice {
            memory_layout: preprocessing.program.memory_layout().clone(),
            inputs: vec![1, 2],
            outputs: vec![3, 0, 0],
            ..JoltDevice::default()
        };
        let proof = proof_with_zk(false, clear_claims());

        let checked = validate_inputs(&preprocessing, &public_io, &proof, false).unwrap();

        assert_eq!(checked.public_io.inputs, vec![1, 2]);
        assert_eq!(checked.public_io.outputs, vec![3]);
        assert_eq!(checked.trace_length, proof.trace_length);
        assert_eq!(checked.ram_K, proof.ram_K);

        public_io.outputs = vec![0, 0];
        let checked = validate_inputs(&preprocessing, &public_io, &proof, false).unwrap();
        assert!(checked.public_io.outputs.is_empty());
    }

    #[test]
    fn validate_inputs_rejects_public_io_layout_mismatch() {
        let preprocessing = test_preprocessing();
        let public_io = JoltDevice::default();
        let proof = proof_with_zk(false, clear_claims());

        assert!(matches!(
            validate_inputs(&preprocessing, &public_io, &proof, false),
            Err(VerifierError::MemoryLayoutMismatch)
        ));
    }

    #[test]
    fn validate_inputs_rejects_zero_based_ram_remap() {
        let mut memory_layout = test_memory_layout();
        // A layout whose remap is zero-based: `unmap(0) = lowest_address = 0`
        // would make the RAF identity blind to digit zero.
        memory_layout.trusted_advice_start = 0;
        memory_layout.untrusted_advice_start = 0;
        let preprocessing = test_preprocessing_with_layout(memory_layout);
        let public_io = JoltDevice {
            memory_layout: preprocessing.program.memory_layout().clone(),
            ..JoltDevice::default()
        };
        let proof = proof_with_zk(false, clear_claims());

        assert!(matches!(
            validate_inputs(&preprocessing, &public_io, &proof, false),
            Err(VerifierError::InvalidMemoryLayout { reason })
                if reason.contains("lowest remapped RAM address")
        ));
    }

    #[test]
    fn validate_inputs_rejects_misaligned_ram_remap_base() {
        let mut memory_layout = test_memory_layout();
        memory_layout.trusted_advice_start -= 4;
        memory_layout.untrusted_advice_start -= 4;
        let preprocessing = test_preprocessing_with_layout(memory_layout);
        let public_io = JoltDevice {
            memory_layout: preprocessing.program.memory_layout().clone(),
            ..JoltDevice::default()
        };
        let proof = proof_with_zk(false, clear_claims());

        assert!(matches!(
            validate_inputs(&preprocessing, &public_io, &proof, false),
            Err(VerifierError::InvalidMemoryLayout { reason })
                if reason.contains("doubleword-aligned")
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
            validate_inputs(&preprocessing, &public_io, &proof, false),
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
            validate_inputs(&preprocessing, &public_io, &proof, false),
            Err(VerifierError::InvalidRamK {
                got,
                min: 4,
                max,
            }) if got == 1 << 20 && max < got
        ));
    }

    #[test]
    fn validate_inputs_rejects_trace_length_below_pcs_floor() {
        let preprocessing = test_preprocessing();
        let public_io = JoltDevice {
            memory_layout: preprocessing.program.memory_layout().clone(),
            ..JoltDevice::default()
        };
        let mut proof = proof_with_zk(false, clear_claims());
        // A power of two below the fixture's maximum, so only the floor rejects it.
        let undersized = MIN_PADDED_TRACE_LENGTH >> 1;
        proof.trace_length = undersized;

        assert!(matches!(
            validate_inputs(&preprocessing, &public_io, &proof, false),
            Err(VerifierError::InvalidTraceLength { got, .. }) if got == undersized
        ));
        assert!(matches!(
            validate_inputs_from_parts(
                &preprocessing,
                &public_io,
                undersized,
                proof.ram_K,
                proof.trace_polynomial_order,
                proof.one_hot_config,
                false,
                false,
                false,
            ),
            Err(VerifierError::InvalidTraceLength { got, .. }) if got == undersized
        ));
    }

    /// The field-inline BlindFold generator budget must fit the largest committed round of the
    /// composed protocol: the Spartan outer uni-skip first round (degree
    /// `SPARTAN_OUTER_UNISKIP_FIRST_ROUND_DEGREE`, one coefficient more), or `commit_round`
    /// fails closed at proving time.
    #[cfg(feature = "field-inline")]
    #[test]
    fn blindfold_generator_budget_covers_the_composed_uniskip_rounds() {
        use jolt_claims::protocols::composed::geometry::SPARTAN_PRODUCT_UNISKIP_FIRST_ROUND_DEGREE;
        use jolt_claims::protocols::composed::r1cs::SPARTAN_OUTER_UNISKIP_FIRST_ROUND_DEGREE;

        const {
            assert!(MAX_BLINDFOLD_GENERATORS > SPARTAN_OUTER_UNISKIP_FIRST_ROUND_DEGREE);
        }
        const {
            assert!(MAX_BLINDFOLD_GENERATORS > SPARTAN_PRODUCT_UNISKIP_FIRST_ROUND_DEGREE);
        }
    }

    #[cfg(feature = "zk")]
    #[test]
    fn validate_inputs_rejects_missing_zk_vector_commitment_setup() {
        let mut preprocessing = test_preprocessing();
        preprocessing.vc_setup = None;
        let public_io = JoltDevice {
            memory_layout: preprocessing.program.memory_layout().clone(),
            ..JoltDevice::default()
        };
        let proof = proof_with_zk(true, zk_claims());

        assert!(matches!(
            validate_inputs(&preprocessing, &public_io, &proof, false),
            Err(VerifierError::MissingVectorCommitmentSetup)
        ));
    }

    #[cfg(feature = "zk")]
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
            validate_inputs(&preprocessing, &public_io, &proof, false),
            Err(VerifierError::InvalidVectorCommitmentCapacity { got: 1, .. })
        ));
    }

    fn proof_with_zk(is_zk: bool, claims: TestClaims) -> TestProof {
        JoltProof {
            protocol: crate::config::JoltProtocolConfig::for_zk(claims.is_zk()),
            #[cfg(not(feature = "akita"))]
            commitments: test_commitments(),
            #[cfg(feature = "akita")]
            commitments: TestCommitment,
            stages: stage_proofs(is_zk),
            #[cfg(not(feature = "akita"))]
            joint_opening_proof: (),
            #[cfg(feature = "akita")]
            joint_opening_proof: (),
            untrusted_advice_commitment: None,
            #[cfg(feature = "akita")]
            fused_inc_commitment: Some(TestCommitment),
            #[cfg(all(feature = "akita", feature = "field-inline"))]
            field_inc_commitment: Some(TestCommitment),
            claims,
            trace_length: MIN_PADDED_TRACE_LENGTH,
            ram_K: 4,
            rw_config: JoltReadWriteConfig {
                ram_rw_phase1_num_rounds: 0,
                ram_rw_phase2_num_rounds: 0,
                registers_rw_phase1_num_rounds: 0,
                registers_rw_phase2_num_rounds: 0,
            },
            one_hot_config: JoltOneHotConfig {
                log_k_chunk: 0,
                lookups_ra_virtual_log_k_chunk: 0,
            },
            trace_polynomial_order: crate::proof::TracePolynomialOrder::CycleMajor,
            narg: Vec::new(),
        }
    }

    #[cfg(not(feature = "akita"))]
    fn test_commitments() -> JoltCommitments<TestCommitment> {
        #[cfg(feature = "field-inline")]
        {
            JoltCommitments::new(
                TestCommitment,
                TestCommitment,
                Vec::<TestCommitment>::new(),
                Vec::<TestCommitment>::new(),
                Vec::<TestCommitment>::new(),
            )
            .with_field_inline(FieldInlineCommitments {
                field_registers: FieldRegistersCommitments {
                    rd_inc: TestCommitment,
                },
            })
        }
        #[cfg(not(feature = "field-inline"))]
        JoltCommitments::new(
            TestCommitment,
            TestCommitment,
            Vec::<TestCommitment>::new(),
            Vec::<TestCommitment>::new(),
            Vec::<TestCommitment>::new(),
        )
    }

    fn clear_claims() -> TestClaims {
        let zero = Fr::zero();

        JoltProofClaims::Clear(ClearProofClaims {
            stage1: Stage1OutputClaims::new(zero, empty_spartan_outer_claims()),
            stage2: Stage2OutputClaims::new(
                zero,
                Stage2BatchOutputClaims {
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
                    }.into(),
                    instruction_claim_reduction:
                        stage2::outputs::InstructionClaimReductionOutputClaims {
                            lookup_output: zero,
                            left_lookup_operand: zero,
                            right_lookup_operand: zero,
                            left_instruction_input: zero,
                            right_instruction_input: zero,
                        },
                    #[cfg(feature = "field-inline")]
                    field_registers_claim_reduction:
                        FieldRegistersClaimReductionOutputClaims {
                            rd_value: zero,
                            rs1_value: zero,
                            rs2_value: zero,
                        },
                    ram_raf_evaluation: stage2::outputs::RamRafEvaluationOutputClaims {
                        ram_ra: zero,
                    },
                    ram_output_check: stage2::outputs::RamOutputCheckOutputClaims {
                        val_final: zero,
                    },
                },
            ),
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
                registers_read_write: stage4::RegistersReadWriteOutputClaims {
                    registers_val: zero,
                    rs1_ra: zero,
                    rs2_ra: zero,
                    rd_wa: zero,
                    rd_inc: zero,
                },
                #[cfg(feature = "field-inline")]
                field_registers_read_write: FieldRegistersReadWriteOutputClaims {
                    registers_val: zero,
                    rs1_ra: zero,
                    rs2_ra: zero,
                    rd_wa: zero,
                    rd_inc: zero,
                },
                ram_val_check: stage4::RamValCheckOutputClaims {
                    untrusted_advice: None,
                    trusted_advice: None,
                    program_image: None,
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
                #[cfg(feature = "field-inline")]
                field_registers_val_evaluation: FieldRegistersValEvaluationOutputClaims {
                    rd_inc: zero,
                    rd_wa: zero,
                },
            },
            stage6a: stage6a::outputs::Stage6aOutputClaims {
                bytecode_read_raf: stage6a::outputs::BytecodeReadRafAddressPhaseOutputClaims {
                    intermediate: zero,
                    val_stages: Vec::new(),
                }.into(),
                booleanity: stage6a::outputs::BooleanityAddressPhaseOutputClaims {
                    intermediate: zero,
                },
            },
            stage6b: stage6b::outputs::Stage6bOutputClaims {
                #[cfg(not(feature = "akita"))]
                bytecode_read_raf: stage6b::outputs::BytecodeReadRafOutputClaims {
                    bytecode_ra: Vec::new(),
                },
                #[cfg(feature = "akita")]
                bytecode_read_raf:
                    stage6b::bytecode_read_raf::LatticeBytecodeReadRafOutputClaims {
                        bytecode_ra: Vec::new(),
                        fused_inc: zero,
                    },
                booleanity: BooleanityOutputClaims {
                    instruction_ra: Vec::new(),
                    bytecode_ra: Vec::new(),
                    ram_ra: Vec::new(),
                },
                ram_hamming_booleanity: stage6b::outputs::RamHammingBooleanityOutputClaims {
                    ram_hamming_weight: zero,
                },
                ram_ra_virtualization: stage6b::outputs::RamRaVirtualizationOutputClaims {
                    ram_ra: Vec::new(),
                },
                instruction_ra_virtualization:
                    stage6b::outputs::InstructionRaVirtualizationOutputClaims {
                        committed_instruction_ra: Vec::new(),
                    },
                #[cfg(not(feature = "akita"))]
                inc_claim_reduction: stage6b::outputs::IncClaimReductionOutputClaims {
                    ram_inc: zero,
                    rd_inc: zero,
                },
                #[cfg(feature = "field-inline")]
                field_registers_inc_claim_reduction:
                    FieldRegistersIncClaimReductionOutputClaims { rd_inc: zero },
                #[cfg(not(feature = "akita"))]
                trusted_advice: None,
                #[cfg(not(feature = "akita"))]
                untrusted_advice: None,
                bytecode_reduction: None,
                program_image_reduction: None,
            },
            stage7: stage7::outputs::Stage7OutputClaims {
                hamming_weight_claim_reduction:
                    stage7::hamming_weight_claim_reduction::HammingWeightClaimReductionOutputClaims {
                        instruction_ra: Vec::new(),
                        bytecode_ra: Vec::new(),
                        ram_ra: Vec::new(),
                    },
                #[cfg(not(feature = "akita"))]
                trusted_advice: None,
                #[cfg(not(feature = "akita"))]
                untrusted_advice: None,
                bytecode_address_phase: None,
                program_image_address_phase: None,
            },
        })
    }

    fn empty_spartan_outer_claims() -> stage1::outputs::Stage1BatchOutputClaims<Fr> {
        stage1::outputs::Stage1BatchOutputClaims {
            outer_remainder: OuterRemainderOutputClaims::<Fr>::default().into(),
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

    fn test_memory_layout() -> common::jolt_device::MemoryLayout {
        common::jolt_device::MemoryLayout::new(&MemoryConfig {
            program_size: Some(1024),
            max_trusted_advice_size: 0,
            max_untrusted_advice_size: 0,
            max_input_size: 8,
            max_output_size: 8,
            stack_size: 8,
            heap_size: 8,
        })
    }

    fn test_preprocessing() -> JoltVerifierPreprocessing<TestPcs, Pedersen<Bn254G1>> {
        test_preprocessing_with_layout(test_memory_layout())
    }

    #[expect(clippy::expect_used, reason = "test fixture")]
    fn test_preprocessing_with_layout(
        memory_layout: common::jolt_device::MemoryLayout,
    ) -> JoltVerifierPreprocessing<TestPcs, Pedersen<Bn254G1>> {
        let program = JoltProgramPreprocessing::new(
            Vec::new(),
            Vec::new(),
            memory_layout,
            RAM_START_ADDRESS,
            MIN_PADDED_TRACE_LENGTH,
            JOLT_VERIFIER_INSTRUCTION_PROFILE,
        )
        .expect("test program");
        #[cfg(feature = "zk")]
        let vc_setup = Some(PedersenSetup::new(
            vec![Bn254G1::default(); MAX_BLINDFOLD_GENERATORS],
            Bn254G1::default(),
        ));
        #[cfg(not(feature = "zk"))]
        let vc_setup = None;
        JoltVerifierPreprocessing::new(ProgramPreprocessing::Full(Arc::new(program)), (), vc_setup)
            .expect("test program digest")
    }

    #[test]
    #[expect(clippy::unwrap_used, reason = "test fixture")]
    fn verifier_preprocessing_recomputes_its_digest_on_load() {
        let preprocessing = test_preprocessing();
        let encoded =
            bincode::serde::encode_to_vec(&preprocessing, bincode::config::standard()).unwrap();

        // The digest is not on the wire: a stale in-memory copy encodes
        // identically and decoding rebuilds the digest from the program.
        let mut stale = preprocessing.clone();
        stale.preprocessing_digest = [0xa5; 32];
        assert_eq!(
            bincode::serde::encode_to_vec(&stale, bincode::config::standard()).unwrap(),
            encoded
        );
        let (decoded, consumed): (JoltVerifierPreprocessing<TestPcs, Pedersen<Bn254G1>>, usize) =
            bincode::serde::decode_from_slice(&encoded, bincode::config::standard()).unwrap();
        assert_eq!(consumed, encoded.len());
        assert_eq!(decoded.program, preprocessing.program);
        assert_eq!(
            decoded.preprocessing_digest,
            preprocessing.preprocessing_digest
        );
        assert_eq!(
            decoded.preprocessing_digest,
            preprocessing.program.digest().unwrap()
        );
    }
}
