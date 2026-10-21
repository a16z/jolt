//! A transcript wrapper for tests: scripted challenges, an event log, and no
//! proof of work.

use std::collections::VecDeque;

#[cfg(feature = "prover-grinding")]
use crate::pow::MessageSearch;
use crate::transcript::sealed::Sealed;
use crate::{ChallengeSetSize, FieldTranscript, ProverTranscript};
use spongefish::{Encoding, VerificationError};

/// One sponge operation a [`ScriptedTranscript`] saw, in order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TranscriptEvent<F> {
    /// A public or prover message, as its encoded bytes.
    Absorb(Vec<u8>),
    /// A squeeze of this many bytes outside a challenge draw, such as a
    /// proof-of-work check.
    Squeeze(usize),
    /// One challenge draw and the values it returned.
    Challenges(Vec<F>),
}

/// `T` with its challenge set declared as `2^256` and its challenges
/// optionally replaced by a script.
///
/// Every operation runs on `T` and is logged. A draw of `len` challenges draws
/// them from `T`, keeping its sponge in step, then returns the next scripted
/// values in their place while the script lasts. The declared challenge set
/// makes every difficulty from
/// [`grind_bits_for_loss`](crate::grind_bits_for_loss) 0, so proofs carry no
/// nonces.
///
/// Tests use it to choose challenges, to compare absorb orders, and to reach
/// checks that a hand-built proof would otherwise fail at its first
/// proof-of-work step. It overstates `T`'s challenge set and lets the caller
/// pick challenges, so it gives no soundness and must never back a real proof;
/// the `test-utils` feature that exports it is for dev-dependencies.
pub struct ScriptedTranscript<T, F> {
    inner: T,
    script: VecDeque<F>,
    events: Vec<TranscriptEvent<F>>,
}

impl<T, F> ScriptedTranscript<T, F> {
    /// `inner` with no script: its own challenges, unpriced.
    pub fn new(inner: T) -> Self {
        Self::with_script(inner, Vec::new())
    }

    /// `inner` whose next challenges are `script`, in order.
    pub fn with_script(inner: T, script: Vec<F>) -> Self {
        Self {
            inner,
            script: script.into(),
            events: Vec::new(),
        }
    }

    /// The operations so far, in order.
    pub fn events(&self) -> &[TranscriptEvent<F>] {
        &self.events
    }

    /// The bytes of each absorbed message so far, in order.
    pub fn absorbed(&self) -> Vec<&[u8]> {
        self.events
            .iter()
            .filter_map(|event| match event {
                TranscriptEvent::Absorb(bytes) => Some(bytes.as_slice()),
                TranscriptEvent::Squeeze(_) | TranscriptEvent::Challenges(_) => None,
            })
            .collect()
    }

    /// The wrapped transcript.
    pub fn into_inner(self) -> T {
        self.inner
    }
}

impl<T: Sealed, F> Sealed for ScriptedTranscript<T, F> {
    fn absorb<M: Encoding + ?Sized>(&mut self, message: &M) {
        self.events
            .push(TranscriptEvent::Absorb(message.encode().as_ref().to_vec()));
        self.inner.absorb(message);
    }

    fn squeeze<U>(&mut self, len: usize, decode: impl FnOnce(&[u8]) -> U) -> U {
        self.events.push(TranscriptEvent::Squeeze(len));
        self.inner.squeeze(len, decode)
    }

    fn require(&mut self, ok: bool) -> Result<(), VerificationError> {
        self.inner.require(ok)
    }
}

impl<T: FieldTranscript<F>, F: Clone> FieldTranscript<F> for ScriptedTranscript<T, F> {
    const CHALLENGE_SET: ChallengeSetSize = ChallengeSetSize::power_of_two(256);

    fn challenge_vector(&mut self, len: usize) -> Result<Vec<F>, VerificationError> {
        let mut challenges = self.inner.challenge_vector(len)?;
        for challenge in &mut challenges {
            match self.script.pop_front() {
                Some(scripted) => *challenge = scripted,
                None => break,
            }
        }
        self.events
            .push(TranscriptEvent::Challenges(challenges.clone()));
        Ok(challenges)
    }
}

impl<T: ProverTranscript, F> ProverTranscript for ScriptedTranscript<T, F> {
    type Sponge = T::Sponge;

    fn prover_message<M: Encoding + ?Sized>(&mut self, msg: &M) {
        self.events
            .push(TranscriptEvent::Absorb(msg.encode().as_ref().to_vec()));
        self.inner.prover_message(msg);
    }

    fn narg_string(&self) -> &[u8] {
        self.inner.narg_string()
    }

    #[cfg(feature = "prover-grinding")]
    fn message_search(&self) -> MessageSearch<T::Sponge> {
        self.inner.message_search()
    }

    #[cfg(feature = "prover-grinding")]
    fn grind(&mut self, bits: u32) -> Result<(), VerificationError> {
        self.inner.grind(bits)
    }

    #[cfg(feature = "prover-grinding")]
    fn public_grind(&mut self, bits: u32) -> Result<Option<u32>, VerificationError> {
        self.inner.public_grind(bits)
    }
}
