//! The verifier's end: spongefish's verifier state, read through typed atoms.

use std::num::NonZeroU8;

use jolt_field::{CanonicalBytes, CanonicalDecode, CanonicalEncoding, Field};
use spongefish::VerifierState;

use crate::grinding::{grinding_accepts, nonce_bits, GRINDING_SEED_LEN};
use crate::site::{Log, TranscriptOp};
use crate::state::{domain, Framed, Squeeze, BYTE_BLOCK};
#[cfg(feature = "logging")]
use crate::TranscriptEvent;
use crate::{Channel, Nonce, ProtocolId, SiteId, Sponge, TranscriptError};

/// Verifier transcript: spongefish's [`VerifierState`] over the proof bytes.
///
/// The only way to obtain proof data is a `receive*` call, which reads and
/// absorbs through spongefish. This wrapper tracks the read offset itself so
/// it can bound a read before allocating and report typed errors. After any
/// failed receive the transcript is poisoned: later receives fail and
/// [`finish`](Self::finish) fails.
pub struct VerifierTranscript<'a, H: Sponge> {
    state: VerifierState<'a, H>,
    narg: &'a [u8],
    consumed: usize,
    poisoned: bool,
    log: Log,
}

impl<H: Sponge> core::fmt::Debug for VerifierTranscript<'_, H> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("VerifierTranscript")
            .field("consumed", &self.consumed)
            .field("remaining", &self.remaining())
            .field("poisoned", &self.poisoned)
            .finish_non_exhaustive()
    }
}

impl<'a, H: Sponge> VerifierTranscript<'a, H> {
    /// Starts a transcript for `protocol`, bound to `session`, over the proof `narg`.
    #[must_use]
    pub fn new(protocol: &ProtocolId, session: &[u8], narg: &'a [u8]) -> Self {
        Self {
            state: domain(protocol, session).to_verifier(H::default(), narg),
            narg,
            consumed: 0,
            poisoned: false,
            log: Log::default(),
        }
    }

    /// Receives one atom.
    ///
    /// # Errors
    ///
    /// [`TranscriptError::Truncated`] or [`TranscriptError::NonCanonical`].
    pub fn receive<A: CanonicalDecode>(&mut self) -> Result<A, TranscriptError> {
        self.bound(A::NUM_BYTES)?;
        let value = self.state.prover_message::<A>();
        let value = self.check(value.map_err(|_| TranscriptError::NonCanonical))?;
        self.advance(A::NUM_BYTES);
        Ok(value)
    }

    /// Receives `count` atoms. Checks that the proof holds them before allocating.
    ///
    /// # Errors
    ///
    /// [`TranscriptError::Truncated`] or [`TranscriptError::NonCanonical`].
    pub fn receive_n<A: CanonicalDecode>(
        &mut self,
        count: usize,
    ) -> Result<Vec<A>, TranscriptError> {
        let len = A::NUM_BYTES
            .checked_mul(count)
            .ok_or(TranscriptError::Truncated);
        let len = self.check(len)?;
        self.bound(len)?;
        let values = self.state.prover_messages_vec::<A>(count);
        let values = self.check(values.map_err(|_| TranscriptError::NonCanonical))?;
        self.advance(len);
        Ok(values)
    }

    /// Receives `len` bytes whose length the verifier already knows.
    ///
    /// # Errors
    ///
    /// [`TranscriptError::Truncated`].
    pub fn receive_bytes(&mut self, len: usize) -> Result<&'a [u8], TranscriptError> {
        self.bound(len)?;
        let narg = self.narg;
        let bytes = narg
            .get(self.consumed..)
            .and_then(|rest| rest.get(..len))
            .ok_or(TranscriptError::Truncated);
        let bytes = self.check(bytes)?;
        // The same block split the prover sends in, so both sides make the
        // same absorb calls whatever the sponge's block handling.
        let read = (0..len / BYTE_BLOCK)
            .try_for_each(|_| self.state.prover_message::<[u8; BYTE_BLOCK]>().map(|_| ()))
            .and_then(|()| {
                (0..len % BYTE_BLOCK)
                    .try_for_each(|_| self.state.prover_message::<[u8; 1]>().map(|_| ()))
            });
        self.check(read.map_err(|_| TranscriptError::Truncated))?;
        self.advance(len);
        Ok(bytes)
    }

    /// Receives a length-prefixed byte string of at most `max_len` bytes,
    /// checking the length before reading the body.
    ///
    /// # Errors
    ///
    /// [`TranscriptError::OutOfBounds`] if the length exceeds `max_len`, else as
    /// [`receive_bytes`](Self::receive_bytes).
    pub fn receive_bounded_bytes(&mut self, max_len: usize) -> Result<&'a [u8], TranscriptError> {
        let len = self.receive::<u32>()?;
        let len = usize::try_from(len)
            .ok()
            .filter(|&len| len <= max_len)
            .ok_or(TranscriptError::OutOfBounds);
        let len = self.check(len)?;
        self.receive_bytes(len)
    }

    /// Receives a [`Nonce`] below `2^nonce_bits`.
    ///
    /// # Errors
    ///
    /// [`TranscriptError::NonCanonical`] for a malformed or non-minimal
    /// encoding, [`TranscriptError::OutOfBounds`] for a value at or above
    /// `2^nonce_bits`.
    pub fn receive_nonce(&mut self, nonce_bits: u8) -> Result<u32, TranscriptError> {
        self.check(Ok(()))?;
        let nonce = self.state.prover_message::<Nonce>();
        let Nonce(nonce) = self.check(nonce.map_err(|_| TranscriptError::NonCanonical))?;
        self.advance(Nonce(nonce).encoded_len());
        let in_range = if u64::from(nonce) >> nonce_bits == 0 {
            Ok(nonce)
        } else {
            Err(TranscriptError::OutOfBounds)
        };
        self.check(in_range)
    }

    /// Checks `bits` bits of proof of work and returns the received nonce. A
    /// zero difficulty draws and reads nothing.
    ///
    /// # Errors
    ///
    /// [`TranscriptError::UnsupportedGrinding`] above
    /// [`MAX_GRINDING_BITS`](crate::MAX_GRINDING_BITS),
    /// [`TranscriptError::OutOfBounds`] for a nonce outside the search range,
    /// [`TranscriptError::GrindingRejected`] if the predicate fails, else as
    /// [`receive`](Self::receive).
    pub fn check_grind(&mut self, bits: u8) -> Result<u32, TranscriptError> {
        let Some(bits) = NonZeroU8::new(bits) else {
            return Ok(0);
        };
        let nonce_bits = nonce_bits(bits).ok_or(TranscriptError::UnsupportedGrinding);
        let nonce_bits = self.check(nonce_bits)?;
        let seed: [u8; GRINDING_SEED_LEN] = self.challenge_bytes();
        let nonce = self.receive_nonce(nonce_bits)?;
        let accepted = if grinding_accepts::<H>(&seed, nonce, bits) {
            Ok(nonce)
        } else {
            Err(TranscriptError::GrindingRejected)
        };
        self.check(accepted)
    }

    /// The proof bytes not yet received, without receiving or absorbing them.
    ///
    /// Only for parse-ahead by a component that needs a whole proof struct
    /// before it runs (dory-pcs). Every byte it parses must still be received
    /// through `receive*`, which is what binds it; parsed-ahead values carry
    /// no weight until then.
    #[must_use]
    pub fn unread(&self) -> &'a [u8] {
        self.narg.split_at(self.consumed).1
    }

    /// Bytes of the proof not yet received.
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.narg.len() - self.consumed
    }

    /// Recorded operations, in order.
    #[cfg(feature = "logging")]
    #[must_use]
    pub fn events(&self) -> &[TranscriptEvent] {
        self.log.events()
    }

    /// Ends the transcript. Call once, at the outermost verifier boundary,
    /// after the protocol's last message.
    ///
    /// # Errors
    ///
    /// [`TranscriptError::Poisoned`] after an earlier failure, and
    /// [`TranscriptError::TrailingBytes`] if proof bytes remain: without this
    /// check `proof || garbage` would verify.
    pub fn finish(self) -> Result<(), TranscriptError> {
        if self.poisoned {
            Err(TranscriptError::Poisoned)
        } else {
            self.state
                .check_eof()
                .map_err(|_| TranscriptError::TrailingBytes)
        }
    }

    /// Fails unless `len` more bytes remain.
    fn bound(&mut self, len: usize) -> Result<(), TranscriptError> {
        let fits = if len <= self.remaining() {
            Ok(())
        } else {
            Err(TranscriptError::Truncated)
        };
        self.check(fits)
    }

    /// Records a read of `len` bytes that spongefish already absorbed.
    fn advance(&mut self, len: usize) {
        let start = self.consumed;
        self.consumed += len;
        self.log
            .record(TranscriptOp::Message, len, Some(start..self.consumed));
    }

    /// Poisons the transcript on failure; fails immediately once poisoned.
    fn check<T>(&mut self, result: Result<T, TranscriptError>) -> Result<T, TranscriptError> {
        if self.poisoned {
            return Err(TranscriptError::Poisoned);
        }
        result.inspect_err(|_| self.poisoned = true)
    }
}

impl<H: Sponge> Channel for VerifierTranscript<'_, H> {
    type Sponge = H;

    fn site(&mut self, site: SiteId) {
        self.log.set_site(site);
    }

    fn public<A: CanonicalBytes>(&mut self, value: &A) {
        self.state.public_message(value);
        self.log.record(TranscriptOp::Public, A::NUM_BYTES, None);
    }

    fn public_all<A: CanonicalBytes>(&mut self, values: &[A]) {
        self.state.public_messages(values);
        self.log
            .record(TranscriptOp::Public, A::NUM_BYTES * values.len(), None);
    }

    fn public_bytes(&mut self, bytes: &[u8]) {
        self.state.public_message(&Framed(bytes));
        self.log.record(TranscriptOp::Public, 8 + bytes.len(), None);
    }

    fn exchange<A: CanonicalDecode>(&mut self, value: &mut A) -> Result<(), TranscriptError> {
        *value = self.receive()?;
        Ok(())
    }

    fn exchange_all<A: CanonicalDecode>(
        &mut self,
        values: &mut [A],
    ) -> Result<(), TranscriptError> {
        let received = self.receive_n(values.len())?;
        for (slot, value) in values.iter_mut().zip(received) {
            *slot = value;
        }
        Ok(())
    }

    fn challenge<F: Field>(&mut self) -> F {
        let (value, squeezed) = self.state.exact_challenge();
        self.log.record(TranscriptOp::Challenge, squeezed, None);
        value
    }

    fn challenge_small<F: CanonicalEncoding>(&mut self) -> F {
        let value = self.state.small_challenge();
        self.log
            .record(TranscriptOp::Challenge, crate::SMALL_CHALLENGE_BYTES, None);
        value
    }

    fn challenge_bytes<const N: usize>(&mut self) -> [u8; N] {
        let value = self.state.squeeze_array();
        self.log.record(TranscriptOp::Challenge, N, None);
        value
    }
}
