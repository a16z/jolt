//! Top-level verifier entry point.

use common::constants::MAX_BLINDFOLD_GENERATORS;
use common::jolt_device::JoltDevice;
#[cfg(not(feature = "akita"))]
use jolt_blindfold::BlindFoldProtocol;
use jolt_claims::protocols::jolt::geometry::dimensions::JoltFormulaDimensions;
use jolt_claims::protocols::jolt::{
    JoltOneHotConfig, JoltReadWriteConfig, JoltRelationId, TracePolynomialOrder,
};
#[cfg(not(feature = "akita"))]
use jolt_crypto::HomomorphicCommitment;
use jolt_crypto::VectorCommitment;
use jolt_field::{CanonicalDecode, JoltField};
use jolt_openings::CommitmentScheme;
#[cfg(not(feature = "akita"))]
use jolt_openings::{AdditivelyHomomorphic, ZkOpeningScheme};
use jolt_program::preprocess::{compute_max_ram_k, compute_min_ram_k};
use jolt_transcript::{Channel, ProtocolId, Sponge, VerifierTranscript};

#[cfg(not(feature = "akita"))]
use crate::sites::BLINDFOLD;
use crate::{
    config::{
        validate_proof_config, ZkConfig, JOLT_VERIFIER_CONFIG, JOLT_VERIFIER_INSTRUCTION_PROFILE,
    },
    num,
    preprocessing::JoltVerifierPreprocessing,
    proof::{JoltProof, ProofCommitments, ProofHeader},
    sites::{COMMITMENTS, PREAMBLE},
    stages::{
        build_formula_dimensions, stage1, stage2, stage3, stage4, stage5, stage6a, stage6b, stage7,
        stage8, CommittedProgramSchedule, PrecommittedSchedule,
    },
    VerifierError,
};

/// The sponge Jolt proofs run on. Every prover and verifier entry point is
/// generic over the sponge; this alias is the one place the default is chosen.
pub type JoltSponge = jolt_transcript::Blake2b512;

/// The session every Jolt transcript is bound to.
pub const JOLT_SESSION: &[u8] = b"";

/// The domain separator of a Jolt proof on sponge `H`. The protocol axes are
/// absorbed separately, by [`absorb_public_preamble`].
pub fn jolt_protocol_id<H: Sponge>() -> ProtocolId {
    ProtocolId::new::<H>("jolt/v1")
}

#[cfg(not(feature = "akita"))]
pub fn verify<F, PCS, VC, H>(
    preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    public_io: &JoltDevice,
    proof: &JoltProof,
    trusted_advice_commitment: Option<&PCS::Output>,
) -> Result<(), VerifierError>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>
        + AdditivelyHomomorphic
        + ZkOpeningScheme<HidingCommitment = VC::Output>,
    PCS::Output: HomomorphicCommitment<F>,
    VC: VectorCommitment<Field = F>,
    VC::Output: Copy + HomomorphicCommitment<F> + CanonicalDecode,
    H: Sponge,
{
    validate_proof_config(&JOLT_VERIFIER_CONFIG, proof.protocol)?;
    let mut transcript =
        VerifierTranscript::<H>::new(&jolt_protocol_id::<H>(), JOLT_SESSION, &proof.narg);
    match verify_stages(
        preprocessing,
        public_io,
        trusted_advice_commitment,
        &mut transcript,
    )? {
        VerifiedStages::Clear => {}
        VerifiedStages::Zk {
            protocol: blindfold,
            checkpoint: _,
        } => {
            transcript.site(BLINDFOLD);
            let vc_setup = preprocessing
                .vc_setup
                .as_ref()
                .ok_or(VerifierError::MissingVectorCommitmentSetup)?;
            blindfold
                .verify::<VC, H>(vc_setup, &mut transcript)
                .map_err(|error| VerifierError::BlindFoldVerificationFailed {
                    reason: error.to_string(),
                })?;
        }
    }
    transcript.finish()?;
    Ok(())
}

/// What remains to check after the stage spine: nothing for a clear proof,
/// the BlindFold relation over the committed stage outputs for a ZK proof.
#[cfg(not(feature = "akita"))]
pub enum VerifiedStages<F: JoltField, C> {
    Clear,
    Zk {
        protocol: Box<BlindFoldProtocol<F, C>>,
        /// The 32 bytes squeezed after the stage spine, before BlindFold.
        /// Every ZK transcript draws it; the prover compares it against its
        /// replay of this function, which checks the replay reached the
        /// prover's sponge state without reading that state.
        checkpoint: [u8; 32],
    },
}

/// Runs the stage spine on `transcript`: the seeding messages, stages 1–8,
/// and, for a ZK proof, the lowering of the committed stage outputs into the
/// BlindFold protocol. The ZK prover replays its own argument string through
/// this function to obtain the protocol it proves.
#[cfg(not(feature = "akita"))]
pub fn verify_stages<F, PCS, VC, H>(
    preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    public_io: &JoltDevice,
    trusted_advice_commitment: Option<&PCS::Output>,
    transcript: &mut VerifierTranscript<'_, H>,
) -> Result<VerifiedStages<F, VC::Output>, VerifierError>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>
        + AdditivelyHomomorphic
        + ZkOpeningScheme<HidingCommitment = VC::Output>,
    PCS::Output: HomomorphicCommitment<F>,
    VC: VectorCommitment<Field = F>,
    VC::Output: Copy + HomomorphicCommitment<F> + CanonicalDecode,
    H: Sponge,
{
    use crate::stages::zk::{blindfold, inputs::BlindFoldInputs};

    let SeededTranscript {
        checked,
        commitments,
        formula_dimensions,
    } = seed_transcript(
        preprocessing,
        public_io,
        trusted_advice_commitment,
        transcript,
    )?;

    let stage1 = stage1::verify::<F, VC::Output, H>(&checked, transcript)?;
    let stage2 = stage2::verify(&checked, transcript, &stage1)?;
    let stage3 = stage3::verify(&checked, transcript, &stage1, &stage2)?;
    let stage4 = stage4::verify(&checked, preprocessing, transcript, &stage2, &stage3)?;
    let stage5 = stage5::verify(&checked, &formula_dimensions, transcript, &stage2, &stage4)?;
    let stage6a = stage6a::verify(
        &checked,
        preprocessing,
        &formula_dimensions,
        transcript,
        &stage1,
        &stage2,
        &stage3,
        &stage4,
        &stage5,
    )?;
    let stage6b = stage6b::verify(
        &checked,
        preprocessing,
        &formula_dimensions,
        transcript,
        &stage1,
        &stage2,
        &stage3,
        &stage4,
        &stage5,
        &stage6a,
    )?;
    let stage7 = stage7::verify(&checked, &formula_dimensions, transcript, &stage4, &stage6b)?;
    let stage8 = stage8::verify(
        &checked,
        preprocessing,
        &commitments,
        &formula_dimensions,
        trusted_advice_commitment,
        transcript,
        &stage6b,
        &stage7,
    )?;

    if !checked.zk {
        let stage8::Stage8Output::Clear(_) = stage8 else {
            return Err(VerifierError::ExpectedClearProof { field: "stage8" });
        };
        return Ok(VerifiedStages::Clear);
    }
    let protocol = Box::new(blindfold::build(BlindFoldInputs {
        checked: &checked,
        preprocessing,
        stage1: stage1.zk()?,
        stage2: stage2.zk()?,
        stage3: stage3.zk()?,
        stage4: stage4.zk()?,
        stage5: stage5.zk()?,
        stage6a: stage6a.zk()?,
        stage6b: stage6b.zk()?,
        stage7: stage7.zk()?,
        stage8: stage8.zk()?,
    })?);
    let checkpoint = transcript.challenge_bytes();
    Ok(VerifiedStages::Zk {
        protocol,
        checkpoint,
    })
}

/// The Akita verification path: the same stage spine, with a random-selector
/// reduction of the packed trace and one native opening for the trace, advice,
/// and committed-program objects. No homomorphism bounds and no ZK tail.
#[cfg(feature = "akita")]
pub fn verify<F, PCS, VC, H>(
    preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    public_io: &JoltDevice,
    proof: &JoltProof,
    trusted_advice_commitment: Option<&PCS::Output>,
) -> Result<(), VerifierError>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    PCS::Output: Clone + stage8::OneHotTraceCommitmentMetadata,
    PCS::VerifierSetup: stage8::OneHotTraceSetupMetadata,
    VC: VectorCommitment<Field = F>,
    VC::Output: Copy + CanonicalDecode,
    H: Sponge,
{
    validate_proof_config(&JOLT_VERIFIER_CONFIG, proof.protocol)?;
    let mut transcript =
        VerifierTranscript::<H>::new(&jolt_protocol_id::<H>(), JOLT_SESSION, &proof.narg);
    let SeededTranscript {
        checked,
        commitments,
        formula_dimensions,
    } = seed_transcript(
        preprocessing,
        public_io,
        trusted_advice_commitment,
        &mut transcript,
    )?;

    let stage1 = stage1::verify::<F, VC::Output, H>(&checked, &mut transcript)?;
    let stage2 = stage2::verify(&checked, &mut transcript, &stage1)?;
    let stage3 = stage3::verify(&checked, &mut transcript, &stage1, &stage2)?;
    let stage4 = stage4::verify(&checked, preprocessing, &mut transcript, &stage2, &stage3)?;
    let stage5 = stage5::verify(
        &checked,
        &formula_dimensions,
        &mut transcript,
        &stage2,
        &stage4,
    )?;
    let stage6a = stage6a::verify(
        &checked,
        preprocessing,
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
        &formula_dimensions,
        &mut transcript,
        &stage4,
        &stage6b,
    )?;
    let stage8 = stage8::verify(
        &checked,
        preprocessing,
        &commitments,
        &formula_dimensions,
        trusted_advice_commitment,
        &mut transcript,
        &stage4,
        &stage6b,
        &stage7,
    )?;

    let stage8::Stage8Output::Clear = stage8 else {
        return Err(VerifierError::ExpectedClearProof { field: "stage8" });
    };

    transcript.finish()?;
    Ok(())
}

/// The verifier state after the transcript's leading messages: the validated
/// inputs, the received polynomial commitments, and the formula dimensions the
/// later stages share.
#[derive(Debug)]
pub struct SeededTranscript<C> {
    pub checked: CheckedInputs,
    pub commitments: ProofCommitments<C>,
    pub formula_dimensions: JoltFormulaDimensions,
}

/// Reads and validates the proof header, absorbs the public preamble, reads the
/// proof's commitments at the counts the header and preprocessing fix, and
/// absorbs the public commitments. The prover's stage 0 performs the mirrored
/// sequence (sending where this receives), so there is one order.
pub fn seed_transcript<PCS, VC, H>(
    preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    public_io: &JoltDevice,
    trusted_advice_commitment: Option<&PCS::Output>,
    transcript: &mut VerifierTranscript<'_, H>,
) -> Result<SeededTranscript<PCS::Output>, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
    H: Sponge,
{
    transcript.site(PREAMBLE);
    let header = ProofHeader::receive(transcript)?;
    let checked = validate_inputs(
        preprocessing,
        public_io,
        &header,
        trusted_advice_commitment.is_some(),
    )?;
    absorb_public_preamble(&checked, transcript);
    let formula_dimensions = build_formula_dimensions(
        preprocessing,
        &checked,
        num::ilog2(checked.trace_length),
        JoltRelationId::InstructionReadRaf,
    )?;
    transcript.site(COMMITMENTS);
    let commitments = ProofCommitments::receive::<PCS, H>(
        &preprocessing.pcs_setup,
        &header,
        #[cfg(not(feature = "akita"))]
        formula_dimensions.ra_layout,
        transcript,
    )?;
    absorb_public_commitments(preprocessing, trusted_advice_commitment, transcript);
    Ok(SeededTranscript {
        checked,
        commitments,
        formula_dimensions,
    })
}

#[expect(non_snake_case, reason = "Preserves the deployed proof field name.")]
#[derive(Clone, Debug, PartialEq)]
pub struct CheckedInputs {
    pub public_io: JoltDevice,
    pub zk: bool,
    pub trace_length: usize,
    pub ram_K: usize,
    pub rw_config: JoltReadWriteConfig,
    pub one_hot_config: JoltOneHotConfig,
    pub trace_polynomial_order: TracePolynomialOrder,
    pub untrusted_advice_commitment_present: bool,
    pub entry_address: u64,
    pub preprocessing_digest: [u8; 32],
    pub trusted_advice_commitment_present: bool,
    pub vc_capacity: Option<usize>,
    pub precommitted: PrecommittedSchedule,
}

impl CheckedInputs {
    /// The committed output-claim row length: the vector-commitment capacity.
    /// `validate_zk_vector_commitment_setup` has already checked it is at
    /// least `MAX_BLINDFOLD_GENERATORS`, so it is nonzero.
    pub fn committed_row_len(&self) -> Result<usize, VerifierError> {
        self.vc_capacity
            .ok_or(VerifierError::MissingVectorCommitmentSetup)
    }
}

/// Absorbs the public preamble: the verifier's protocol axes, the
/// preprocessing digest, the public I/O, and the entry address. Both sides run
/// it right after the proof header.
pub fn absorb_public_preamble<C: Channel>(checked: &CheckedInputs, channel: &mut C) {
    let public_io = &checked.public_io;
    channel.public(&JOLT_VERIFIER_CONFIG.transcript_bytes());
    channel.public(&checked.preprocessing_digest);
    channel.public_all(&[
        public_io.memory_layout.max_input_size,
        public_io.memory_layout.max_output_size,
        public_io.memory_layout.heap_size,
    ]);
    channel.public_bytes(&public_io.inputs);
    channel.public_bytes(&public_io.outputs);
    channel.public(&u8::from(public_io.panic));
    channel.public(&checked.entry_address);
}

/// Absorbs the commitments both sides hold, after the proof's own: the trusted
/// advice commitment, then the committed program's preprocessing-held
/// commitments (bytecode chunks, then the program image).
pub fn absorb_public_commitments<PCS, VC, C>(
    preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    trusted_advice_commitment: Option<&PCS::Output>,
    channel: &mut C,
) where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
    C: Channel,
{
    if let Some(commitment) = trusted_advice_commitment {
        PCS::absorb_commitment(commitment, channel);
    }
    let Some(committed) = preprocessing.program.committed() else {
        return;
    };
    #[cfg(not(feature = "akita"))]
    {
        for commitment in &committed.bytecode_chunk_commitments {
            PCS::absorb_commitment(commitment, channel);
        }
        PCS::absorb_commitment(&committed.program_image_commitment, channel);
    }
    #[cfg(feature = "akita")]
    for commitment in &committed.direct_program_commitments {
        PCS::absorb_commitment(commitment, channel);
    }
}

/// Validates the public inputs and the proof header against the preprocessing.
/// Both sides run it: the verifier on the header it received, the prover on the
/// header it is about to send.
pub fn validate_inputs<PCS, VC>(
    preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    public_io: &JoltDevice,
    header: &ProofHeader,
    trusted_advice_commitment_present: bool,
) -> Result<CheckedInputs, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    let trace_length = header.trace_length;
    let ram_k = header.ram_K;
    let trace_polynomial_order = header.trace_polynomial_order;
    let one_hot_config = header.one_hot_config;
    #[cfg(not(feature = "akita"))]
    let untrusted_advice_commitment_present = header.untrusted_advice;
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

    if !trace_length.is_power_of_two() || trace_length > program.max_padded_trace_length() {
        return Err(VerifierError::InvalidTraceLength {
            got: trace_length,
            max: program.max_padded_trace_length(),
        });
    }

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
                #[cfg(feature = "akita")]
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
                    bytecode_chunk_count: committed.bytecode_chunk_count(),
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
        rw_config: header.rw_config,
        one_hot_config,
        trace_polynomial_order,
        untrusted_advice_commitment_present: header.untrusted_advice,
        entry_address: program.entry_address(),
        preprocessing_digest: preprocessing.preprocessing_digest,
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

/// Fail closed on a zero-based RAM remap. Stage 2's RAF-evaluation unmap is
/// `8k + lowest_address`, and the lattice digit-zero reconstruction relies on
/// `unmap(0) = lowest_address ≠ 0` to distinguish "no RAM access" from an
/// access at remapped word zero (see "Where the RAM activation is pinned" in
/// `specs/lattice-claims.md`). Mirrors the prover-side
/// `UnmapRamAddressPolynomial::new` assertion (`start_address > 8`).
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
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use common::constants::RAM_START_ADDRESS;

    use super::*;
    use crate::preprocessing::ProgramPreprocessing;
    use common::jolt_device::{JoltDevice, MemoryConfig};
    #[cfg(feature = "zk")]
    use jolt_crypto::PedersenSetup;
    use jolt_crypto::{Bn254G1, Commitment, Pedersen};
    use jolt_field::Fr;
    use jolt_openings::{CommitmentScheme, OpeningsError};
    use jolt_poly::MultilinearPoly;
    use jolt_program::preprocess::JoltProgramPreprocessing;
    use jolt_transcript::{ProverTranscript, VerifierTranscript};

    #[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
    struct TestPcs;

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
    struct TestCommitment;

    impl Commitment for TestPcs {
        type Output = TestCommitment;
    }

    impl CommitmentScheme for TestPcs {
        type Field = Fr;
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

        fn send_commitment<H: Sponge>(
            _commitment: &Self::Output,
            _transcript: &mut ProverTranscript<H>,
        ) {
        }

        fn receive_commitment<H: Sponge>(
            _setup: &Self::VerifierSetup,
            _transcript: &mut VerifierTranscript<'_, H>,
        ) -> Result<Self::Output, OpeningsError> {
            Ok(TestCommitment)
        }

        fn absorb_commitment<C: Channel>(_commitment: &Self::Output, _channel: &mut C) {}

        fn open<P: MultilinearPoly<Self::Field> + ?Sized, H: Sponge>(
            _poly: &P,
            _point: &[Self::Field],
            _eval: Self::Field,
            _setup: &Self::ProverSetup,
            _hint: Option<Self::OpeningHint>,
            _transcript: &mut ProverTranscript<H>,
        ) -> Result<(), OpeningsError> {
            Ok(())
        }

        fn verify<H: Sponge>(
            _commitment: &Self::Output,
            _point: &[Self::Field],
            _eval: Self::Field,
            _setup: &Self::VerifierSetup,
            _transcript: &mut VerifierTranscript<'_, H>,
        ) -> Result<(), OpeningsError> {
            Ok(())
        }
    }

    // The homomorphic-build entry point's bounds; never reached by the tests,
    // which reject before any commitment arithmetic.
    #[cfg(not(feature = "akita"))]
    impl HomomorphicCommitment<Fr> for TestCommitment {
        fn add(_c1: &Self, _c2: &Self) -> Self {
            Self
        }

        fn linear_combine(_c1: &Self, _c2: &Self, _scalar: &Fr) -> Self {
            Self
        }
    }

    #[cfg(not(feature = "akita"))]
    impl AdditivelyHomomorphic for TestPcs {
        fn combine(_commitments: &[Self::Output], _scalars: &[Self::Field]) -> Self::Output {
            TestCommitment
        }
    }

    #[cfg(not(feature = "akita"))]
    impl ZkOpeningScheme for TestPcs {
        type HidingCommitment = Bn254G1;
        type Blind = ();

        fn commit_zk<P: MultilinearPoly<Self::Field> + ?Sized>(
            _poly: &P,
            _setup: &Self::ProverSetup,
        ) -> Result<(Self::Output, Self::OpeningHint), OpeningsError> {
            Ok((TestCommitment, ()))
        }

        fn open_zk<P: MultilinearPoly<Self::Field> + ?Sized, H: Sponge>(
            _poly: &P,
            _point: &[Self::Field],
            _eval: Self::Field,
            _setup: &Self::ProverSetup,
            _hint: Self::OpeningHint,
            _transcript: &mut ProverTranscript<H>,
        ) -> Result<(Self::HidingCommitment, Self::Blind), OpeningsError> {
            Ok((Bn254G1::default(), ()))
        }

        fn verify_zk<H: Sponge>(
            _commitment: &Self::Output,
            _point: &[Self::Field],
            _setup: &Self::VerifierSetup,
            _transcript: &mut VerifierTranscript<'_, H>,
        ) -> Result<Self::HidingCommitment, OpeningsError> {
            Ok(Bn254G1::default())
        }
    }

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

        assert_proof_traits::<JoltProof>();
    }

    /// `verify` rejects a proof produced under other protocol axes as a
    /// configuration mismatch before reading the argument string or
    /// validating the preprocessing: here the argument string is empty and
    /// the bytecode is invalid, each of which would fail later.
    #[cfg(not(feature = "akita"))]
    #[test]
    fn protocol_mismatch_rejects_before_transcript_and_preprocessing() {
        #[cfg(feature = "field-inline")]
        use jolt_riscv::JoltInstructionKind;
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
        let mut protocol = JOLT_VERIFIER_CONFIG;
        protocol.zk = match protocol.zk {
            ZkConfig::BlindFold => ZkConfig::Transparent,
            ZkConfig::Transparent => ZkConfig::BlindFold,
        };
        let proof = JoltProof {
            protocol,
            narg: Vec::new(),
        };

        assert!(matches!(
            verify::<Fr, TestPcs, Pedersen<Bn254G1>, JoltSponge>(
                &preprocessing,
                &JoltDevice::default(),
                &proof,
                None,
            ),
            Err(VerifierError::ProtocolConfigMismatch { .. })
        ));
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
        let header = test_header();

        let checked = validate_inputs(&preprocessing, &public_io, &header, false).unwrap();

        assert_eq!(checked.public_io.inputs, vec![1, 2]);
        assert_eq!(checked.public_io.outputs, vec![3]);
        assert_eq!(checked.trace_length, header.trace_length);
        assert_eq!(checked.ram_K, header.ram_K);

        public_io.outputs = vec![0, 0];
        let checked = validate_inputs(&preprocessing, &public_io, &header, false).unwrap();
        assert!(checked.public_io.outputs.is_empty());
    }

    #[test]
    fn validate_inputs_rejects_public_io_layout_mismatch() {
        let preprocessing = test_preprocessing();
        let public_io = JoltDevice::default();

        assert!(matches!(
            validate_inputs(&preprocessing, &public_io, &test_header(), false),
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

        assert!(matches!(
            validate_inputs(&preprocessing, &public_io, &test_header(), false),
            Err(VerifierError::InvalidMemoryLayout { reason })
                if reason.contains("lowest remapped RAM address")
        ));
    }

    #[test]
    fn validate_inputs_rejects_ram_domain_below_layout_minimum() {
        let preprocessing = test_preprocessing();
        let public_io = JoltDevice {
            memory_layout: preprocessing.program.memory_layout().clone(),
            ..JoltDevice::default()
        };
        let header = ProofHeader {
            ram_K: 2,
            ..test_header()
        };

        assert!(matches!(
            validate_inputs(&preprocessing, &public_io, &header, false),
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
        let header = ProofHeader {
            ram_K: 1 << 20,
            ..test_header()
        };

        assert!(matches!(
            validate_inputs(&preprocessing, &public_io, &header, false),
            Err(VerifierError::InvalidRamK {
                got,
                min: 4,
                max,
            }) if got == 1 << 20 && max < got
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

        assert!(matches!(
            validate_inputs(&preprocessing, &public_io, &test_header(), false),
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

        assert!(matches!(
            validate_inputs(&preprocessing, &public_io, &test_header(), false),
            Err(VerifierError::InvalidVectorCommitmentCapacity { got: 1, .. })
        ));
    }

    fn test_header() -> ProofHeader {
        ProofHeader {
            trace_length: 1,
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
            trace_polynomial_order: TracePolynomialOrder::CycleMajor,
            untrusted_advice: false,
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
            16,
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
