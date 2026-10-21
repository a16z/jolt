//! A transcript wrapper for tests: scripted challenges, an event log, and no
//! proof of work.

use std::borrow::{Borrow, BorrowMut};
use std::collections::VecDeque;

#[cfg(feature = "prover-grinding")]
use crate::pow::MessageSearch;
use crate::transcript::sealed::Sealed;
use crate::{
    ChallengeSetSize, FieldTranscript, ProverInit, ProverTranscript, VerifierInit,
    VerifierTranscript,
};
use spongefish::{
    DuplexSpongeInit, DuplexSpongeInterface, Encoding, FromNarg, VerificationError, VerifierState,
};

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

/// What a [`ScriptedTranscript`] does with the challenges of each draw.
pub trait Script<F> {
    /// Whether the transcript declares its challenge set as `2^256`, which
    /// makes every difficulty from
    /// [`grind_bits_for_loss`](crate::grind_bits_for_loss) 0 so proofs carry
    /// no nonces. Otherwise it declares the wrapped transcript's set.
    const UNPRICED: bool;

    /// Sees the challenges the wrapped transcript drew for one draw, and may
    /// replace them.
    fn draw(&mut self, challenges: &mut [F]);

    /// Sees the bytes of a squeeze outside a challenge draw, such as a
    /// proof-of-work check, and may replace them. By default it keeps them.
    fn check_bytes(&mut self, _bytes: &mut [u8]) {}

    /// Sees the encoding of a message about to be absorbed, a public message
    /// or a prover message the verifier reads, and may replace the bytes the
    /// sponge absorbs. A prover's own messages are not offered. By default it
    /// keeps them.
    fn absorb_bytes(&mut self, _bytes: &mut Vec<u8>) {}
}

/// Bytes absorbed as they are: the encoding of a message a [`Script`] saw.
struct Absorbed<'a>(&'a [u8]);

impl Encoding for Absorbed<'_> {
    fn encode(&self) -> impl AsRef<[u8]> {
        self.0
    }
}

/// A queue of challenges that replace the drawn ones, in order, while it
/// lasts. Unpriced.
impl<F> Script<F> for VecDeque<F> {
    const UNPRICED: bool = true;

    fn draw(&mut self, challenges: &mut [F]) {
        for challenge in challenges {
            match self.pop_front() {
                Some(scripted) => *challenge = scripted,
                None => break,
            }
        }
    }
}

/// `T` whose challenges pass through a [`Script`]: by default a queue of
/// replacement values, with the challenge set declared as `2^256`.
///
/// Every operation runs on `T` and is logged. A draw of `len` challenges draws
/// them from `T`, keeping its sponge in step, then hands them to the script,
/// which may replace them. With the default script the declared challenge set
/// makes every difficulty from
/// [`grind_bits_for_loss`](crate::grind_bits_for_loss) 0, so proofs carry no
/// nonces.
///
/// The script also sees the bytes of every other squeeze, such as a
/// proof-of-work check, and the bytes of every absorbed message except the
/// prover's own, and may replace them.
///
/// Tests use it to choose challenges, to compare absorb orders, to record and
/// replay challenge tapes, and to reach checks that a hand-built proof would
/// otherwise fail at its first proof-of-work step. It can overstate `T`'s
/// challenge set and lets the caller pick challenges, so it gives no soundness
/// and must never back a real proof; the `test-utils` feature that exports it
/// is for dev-dependencies.
pub struct ScriptedTranscript<T, F, S = VecDeque<F>> {
    inner: T,
    script: S,
    events: Vec<TranscriptEvent<F>>,
}

impl<T, F> ScriptedTranscript<T, F> {
    /// `inner` with no script: its own challenges, unpriced.
    pub fn new(inner: T) -> Self {
        Self::with_script(inner, Vec::new())
    }

    /// `inner` whose next challenges are `script`, in order.
    pub fn with_script(inner: T, script: Vec<F>) -> Self {
        Self::scripted_by(inner, script.into())
    }
}

impl<T, F, S> ScriptedTranscript<T, F, S> {
    /// `inner` whose draws pass through `script`.
    pub fn scripted_by(inner: T, script: S) -> Self {
        Self {
            inner,
            script,
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

/// The wrapped state, for a protocol that runs on it directly, such as an
/// embedded opening argument. Its operations bypass the script and the log.
impl<T, F, S> Borrow<T> for ScriptedTranscript<T, F, S> {
    fn borrow(&self) -> &T {
        &self.inner
    }
}

impl<T, F, S> BorrowMut<T> for ScriptedTranscript<T, F, S> {
    fn borrow_mut(&mut self) -> &mut T {
        &mut self.inner
    }
}

impl<'a, H: DuplexSpongeInterface, F, S> From<ScriptedTranscript<VerifierState<'a, H>, F, S>>
    for VerifierState<'a, H>
{
    fn from(transcript: ScriptedTranscript<VerifierState<'a, H>, F, S>) -> Self {
        transcript.inner
    }
}

impl<T: Sealed, F, S: Script<F>> Sealed for ScriptedTranscript<T, F, S> {
    fn absorb<M: Encoding + ?Sized>(&mut self, message: &M) {
        let mut bytes = message.encode().as_ref().to_vec();
        self.script.absorb_bytes(&mut bytes);
        self.inner.absorb(&Absorbed(&bytes));
        self.events.push(TranscriptEvent::Absorb(bytes));
    }

    fn squeeze<U>(&mut self, len: usize, decode: impl FnOnce(&[u8]) -> U) -> U {
        self.events.push(TranscriptEvent::Squeeze(len));
        let script = &mut self.script;
        self.inner.squeeze(len, |bytes| {
            let mut bytes = bytes.to_vec();
            script.check_bytes(&mut bytes);
            decode(&bytes)
        })
    }

    fn require(&mut self, ok: bool) -> Result<(), VerificationError> {
        self.inner.require(ok)
    }
}

impl<T, F, S> FieldTranscript<F> for ScriptedTranscript<T, F, S>
where
    T: FieldTranscript<F>,
    F: Clone,
    S: Script<F>,
{
    const CHALLENGE_SET: ChallengeSetSize = if S::UNPRICED {
        ChallengeSetSize::power_of_two(256)
    } else {
        T::CHALLENGE_SET
    };

    fn challenge_vector(&mut self, len: usize) -> Result<Vec<F>, VerificationError> {
        let mut challenges = self.inner.challenge_vector(len)?;
        self.script.draw(&mut challenges);
        self.events
            .push(TranscriptEvent::Challenges(challenges.clone()));
        Ok(challenges)
    }
}

impl<T: ProverTranscript, F, S: Script<F>> ProverTranscript for ScriptedTranscript<T, F, S> {
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

impl<H, F, S> VerifierTranscript for ScriptedTranscript<VerifierState<'_, H>, F, S>
where
    H: DuplexSpongeInterface<U = u8>,
    S: Script<F>,
{
    fn prover_message<M: Encoding + FromNarg>(&mut self) -> Result<M, VerificationError> {
        let script = &mut self.script;
        let mut absorbed = Vec::new();
        let message = self.inner.prover_message_with(M::from_narg, |message| {
            absorbed = message.encode().as_ref().to_vec();
            script.absorb_bytes(&mut absorbed);
            absorbed.clone()
        })?;
        self.events.push(TranscriptEvent::Absorb(absorbed));
        Ok(message)
    }

    fn check_grind(&mut self, bits: u32) -> Result<(), VerificationError> {
        self.inner.check_grind(bits)
    }

    fn check_eof(self) -> Result<(), VerificationError> {
        self.inner.check_eof()
    }
}

impl<'a, H, F, S> VerifierInit<'a> for ScriptedTranscript<VerifierState<'a, H>, F, S>
where
    H: DuplexSpongeInit<U = u8>,
    S: Script<F> + Default,
{
    fn init(session: &[u8], statement: &(impl Encoding + ?Sized), narg: &'a [u8]) -> Self {
        Self::scripted_by(
            <VerifierState<'a, H> as VerifierInit<'a>>::init(session, statement, narg),
            S::default(),
        )
    }
}

impl<T: ProverInit, F, S: Script<F> + Default> ProverInit for ScriptedTranscript<T, F, S> {
    fn init(session: &[u8], statement: &(impl Encoding + ?Sized)) -> Self {
        Self::scripted_by(T::init(session, statement), S::default())
    }
}
