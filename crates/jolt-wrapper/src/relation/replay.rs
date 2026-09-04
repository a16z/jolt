//! Recording replay of the native verifier: the transcript event log the
//! assign-mode walk reads challenge values and prover scalars from and asserts
//! the circuit's schedule against.

use common::jolt_device::JoltDevice;
use jolt_claims::protocols::jolt::JoltRelationId;
use jolt_field::Fr;
use jolt_transcript::{Blake3Transcript, Transcript};
use jolt_verifier::stages::{
    build_formula_dimensions, stage1, stage2, stage3, stage4, stage5, stage6a, stage6b, stage7,
    stage8,
};
use serde::{Deserialize, Serialize};

use crate::hash_table::{Decoder, Event as RecordedEvent, Recorded, RecordingTranscript};

use super::{Pcs, Preprocessing, Proof, RelationError, Vc};

/// Which decoder the verifier applies to the 16-byte squeeze: `challenge()`
/// (125-bit, Montgomery high-limb placement) or `challenge_scalar()` (128-bit
/// big-endian).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SqueezeKind {
    Challenge,
    Scalar,
}

#[derive(Clone, Debug)]
pub(crate) enum Event {
    Append(Vec<u8>),
    Squeeze { kind: SqueezeKind, value: Fr },
}

pub(crate) struct Replay {
    pub events: Vec<Event>,
    pub state_in: [u8; 32],
    pub records: Vec<Recorded>,
}

/// The clear-mode stage spine of `jolt_verifier::verify` on a recording
/// transcript. The seed (preamble + commitments) is hashed natively and only
/// its resulting state is kept; the log starts at stage 1.
pub(crate) fn replay(
    preprocessing: &Preprocessing,
    public_io: &JoltDevice,
    proof: &Proof,
) -> Result<Replay, RelationError> {
    let _ = RecordingTranscript::<Blake3Transcript>::take_log();
    let (checked, mut transcript) = jolt_verifier::validate_and_seed_transcript::<
        Pcs,
        Vc,
        RecordingTranscript<Blake3Transcript>,
        _,
    >(preprocessing, public_io, proof, None)?;
    let state_in = transcript.state();
    let mut records = RecordingTranscript::<Blake3Transcript>::take_log();
    let formula_dimensions = build_formula_dimensions(
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
    drop(stage8::verify(
        &checked,
        preprocessing,
        proof,
        &formula_dimensions,
        None,
        &mut transcript,
        &stage6b,
        &stage7,
    )?);
    let stage_records = RecordingTranscript::<Blake3Transcript>::take_log();
    let events = stage_records
        .iter()
        .map(|recorded| match &recorded.event {
            RecordedEvent::Append { bytes, .. } => Event::Append(bytes.clone()),
            RecordedEvent::Squeeze { decoder, value } => Event::Squeeze {
                kind: match decoder {
                    Decoder::Challenge125 => SqueezeKind::Challenge,
                    Decoder::Scalar128 => SqueezeKind::Scalar,
                },
                value: *value,
            },
            RecordedEvent::Start { .. } => unreachable!("stage replay reuses the transcript"),
        })
        .collect();
    records.extend(stage_records);
    Ok(Replay {
        events,
        state_in,
        records,
    })
}
