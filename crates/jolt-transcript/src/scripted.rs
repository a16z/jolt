//! A transcript wrapper for tests: scripted challenges, an event log, and no
//! proof of work.

use std::collections::VecDeque;

use crate::transcript::sealed::Sealed;
use crate::{
    ChallengeSetSize, FieldTranscript, NargMessage, ProverInit, ProverTranscript, VerifierInit,
    VerifierTranscript, TAG_BYTES,
};
use spongefish::{Encoding, FromNarg, VerificationError};

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
/// proof-of-work check, and may replace them.
///
/// Tests use it to choose challenges, to compare absorb orders, to record and
/// replay challenge tapes, and to reach checks that a hand-built proof would
/// otherwise fail at its first proof-of-work step. It can overstate `T`'s
/// challenge set and lets the caller pick challenges, so it gives no soundness
/// and must never back a real proof; the `test-utils` feature that exports it
/// is for dev-dependencies. It wraps byte-sponge transcripts only, since its
/// script sees check bytes.
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

impl<T: Sealed<Unit = u8>, F, S: Script<F>> Sealed for ScriptedTranscript<T, F, S> {
    type Unit = u8;

    fn absorb<M: Encoding + ?Sized>(&mut self, message: &M) {
        self.events
            .push(TranscriptEvent::Absorb(message.encode().as_ref().to_vec()));
        self.inner.absorb(message);
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
    T: FieldTranscript<F, Unit = u8>,
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

impl<T: ProverTranscript<Unit = u8>, F, S: Script<F>> ProverTranscript
    for ScriptedTranscript<T, F, S>
{
    fn send<M: NargMessage + ?Sized>(&mut self, msg: &M) {
        self.events
            .push(TranscriptEvent::Absorb(msg.encode().as_ref().to_vec()));
        self.inner.send(msg);
    }

    fn narg_string(&self) -> &[u8] {
        self.inner.narg_string()
    }

    #[cfg(feature = "prover-grinding")]
    fn verifier_message_oracle(&self) -> impl Fn(&[u8], &mut [u8]) + Clone + use<T, F, S> {
        self.inner.verifier_message_oracle()
    }

    #[cfg(feature = "prover-grinding")]
    fn grind(&mut self, bits: u32) -> Result<(), VerificationError> {
        self.inner.grind(bits)
    }

    #[cfg(feature = "prover-grinding")]
    fn public_grind(&mut self, bits: u32) -> Result<Option<u64>, VerificationError> {
        self.inner.public_grind(bits)
    }
}

impl<T: VerifierTranscript<Unit = u8>, F, S: Script<F>> VerifierTranscript
    for ScriptedTranscript<T, F, S>
{
    fn prover_message<M: Encoding + FromNarg>(&mut self) -> Result<M, VerificationError> {
        let message = self.inner.prover_message::<M>()?;
        self.events
            .push(TranscriptEvent::Absorb(message.encode().as_ref().to_vec()));
        Ok(message)
    }

    fn check_grind(&mut self, bits: u32) -> Result<(), VerificationError> {
        self.inner.check_grind(bits)
    }

    fn check_eof(self) -> Result<(), VerificationError> {
        self.inner.check_eof()
    }
}

impl<'a, T: VerifierInit<'a, Unit = u8>, F, S: Script<F> + Default> VerifierInit<'a>
    for ScriptedTranscript<T, F, S>
{
    fn init(
        session: &[u8; TAG_BYTES],
        statement: &(impl Encoding + ?Sized),
        narg: &'a [u8],
    ) -> Self {
        Self::scripted_by(T::init(session, statement, narg), S::default())
    }
}

impl<T: ProverInit<Unit = u8>, F, S: Script<F> + Default> ProverInit
    for ScriptedTranscript<T, F, S>
{
    fn init(session: &[u8; TAG_BYTES], statement: &(impl Encoding + ?Sized)) -> Self {
        Self::scripted_by(T::init(session, statement), S::default())
    }
}
