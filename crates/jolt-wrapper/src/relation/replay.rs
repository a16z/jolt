//! Recording replay of the native verifier: the transcript event log the
//! assign-mode walk reads challenge values and prover scalars from and asserts
//! the circuit's schedule against.

use std::cell::RefCell;

use common::jolt_device::JoltDevice;
use jolt_claims::protocols::jolt::JoltRelationId;
use jolt_field::Fr;
use jolt_transcript::{Blake3Transcript, Transcript};
use jolt_verifier::stages::{
    build_formula_dimensions, stage1, stage2, stage3, stage4, stage5, stage6a, stage6b, stage7,
    stage8,
};
use serde::{Deserialize, Serialize};

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
}

thread_local! {
    static LOG: RefCell<Vec<Event>> = const { RefCell::new(Vec::new()) };
}

fn take_log() -> Vec<Event> {
    LOG.with(|log| std::mem::take(&mut *log.borrow_mut()))
}

fn push(event: Event) {
    LOG.with(|log| log.borrow_mut().push(event));
}

/// Forwards every call to `T`, logging each `append_bytes` payload and each
/// squeeze with its decoder kind. `Transcript::new` takes no state, so the log
/// is thread-local (the replay is single-threaded on the transcript).
#[derive(Default)]
struct Recording<T>(T);

impl<T: Transcript<Challenge = Fr>> Transcript for Recording<T> {
    type Challenge = Fr;

    fn new(label: &'static [u8]) -> Self {
        Self(T::new(label))
    }

    fn append_bytes(&mut self, bytes: &[u8]) {
        push(Event::Append(bytes.to_vec()));
        self.0.append_bytes(bytes);
    }

    fn challenge(&mut self) -> Fr {
        let value = self.0.challenge();
        push(Event::Squeeze {
            kind: SqueezeKind::Challenge,
            value,
        });
        value
    }

    fn challenge_scalar(&mut self) -> Fr {
        let value = self.0.challenge_scalar();
        push(Event::Squeeze {
            kind: SqueezeKind::Scalar,
            value,
        });
        value
    }

    fn state(&self) -> [u8; 32] {
        self.0.state()
    }
}

/// The clear-mode stage spine of `jolt_verifier::verify` on a recording
/// transcript. The seed (preamble + commitments) is hashed natively and only
/// its resulting state is kept; the log starts at stage 1.
pub(crate) fn replay(
    preprocessing: &Preprocessing,
    public_io: &JoltDevice,
    proof: &Proof,
) -> Result<Replay, RelationError> {
    let _ = take_log();
    let (checked, mut transcript) = jolt_verifier::validate_and_seed_transcript::<
        Pcs,
        Vc,
        Recording<Blake3Transcript>,
        _,
    >(preprocessing, public_io, proof, None)?;
    let _ = take_log();
    let state_in = transcript.state();
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
    let events = take_log();
    Ok(Replay { events, state_in })
}
