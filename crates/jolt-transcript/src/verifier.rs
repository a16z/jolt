//! The verifier's end: read every message from the argument string, absorb
//! exactly the bytes read.

use std::num::NonZeroU8;

use jolt_field::{CanonicalBytes, CanonicalDecode, CanonicalEncoding, Field};

use crate::duplex::Duplex;
use crate::grinding::{
    decode_nonce, grinding_predicate_accepts, nonce_bits, GRINDING_PREDICATE_LEN,
};
#[cfg(feature = "logging")]
use crate::TranscriptEvent;
use crate::{Channel, Preview, ProtocolId, SiteId, Sponge, TranscriptError};

/// Verifier transcript: a sponge plus a cursor over the proof bytes.
///
/// The only way to obtain proof data is a `receive*` call, which absorbs the
/// bytes it consumed. After any failed receive the transcript is poisoned:
/// later receives fail and [`finish`](Self::finish) fails.
#[derive(Clone, Debug)]
pub struct VerifierTranscript<'a, H> {
    duplex: Duplex<H>,
    narg: &'a [u8],
    consumed: usize,
    poisoned: bool,
}

impl<'a, H: Sponge> VerifierTranscript<'a, H> {
    /// Starts a transcript for `protocol`, bound to `session`, over the proof `narg`.
    #[must_use]
    pub fn new(protocol: &ProtocolId, session: &[u8], narg: &'a [u8]) -> Self {
        Self {
            duplex: Duplex::new(protocol, session),
            narg,
            consumed: 0,
            poisoned: false,
        }
    }

    /// Receives one atom.
    ///
    /// # Errors
    ///
    /// [`TranscriptError::Truncated`] or [`TranscriptError::NonCanonical`].
    pub fn receive<A: CanonicalDecode>(&mut self) -> Result<A, TranscriptError> {
        let bytes = self.peek(A::NUM_BYTES)?;
        let value = A::from_bytes_le_checked(bytes).ok_or(TranscriptError::NonCanonical);
        let value = self.check(value)?;
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
        let bytes = self.peek(len)?;
        let values = bytes
            .chunks_exact(A::NUM_BYTES)
            .map(|chunk| A::from_bytes_le_checked(chunk).ok_or(TranscriptError::NonCanonical))
            .collect::<Result<Vec<_>, _>>();
        let values = self.check(values)?;
        self.advance(len);
        Ok(values)
    }

    /// Receives `len` bytes whose length the verifier already knows.
    ///
    /// # Errors
    ///
    /// [`TranscriptError::Truncated`].
    pub fn receive_bytes(&mut self, len: usize) -> Result<&'a [u8], TranscriptError> {
        let bytes = self.peek(len)?;
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

    /// Receives a canonically encoded nonce below `2^nonce_bits`.
    ///
    /// # Errors
    ///
    /// [`TranscriptError::NonCanonical`] for a malformed, non-minimal, or
    /// out-of-range encoding.
    pub fn receive_nonce(&mut self, nonce_bits: u8) -> Result<u32, TranscriptError> {
        self.check(Ok(()))?;
        let rest = self.narg.split_at(self.consumed).1;
        let decoded = decode_nonce(rest, nonce_bits).ok_or(TranscriptError::NonCanonical);
        let (nonce, len) = self.check(decoded)?;
        self.advance(len);
        Ok(nonce)
    }

    /// Checks `bits` bits of proof of work and returns the received nonce. A
    /// zero difficulty reads nothing.
    ///
    /// # Errors
    ///
    /// [`TranscriptError::UnsupportedGrinding`] above
    /// [`MAX_GRINDING_BITS`](crate::MAX_GRINDING_BITS),
    /// [`TranscriptError::GrindingRejected`] if the predicate fails, else as
    /// [`receive_nonce`](Self::receive_nonce).
    pub fn check_grind(&mut self, bits: u8) -> Result<u32, TranscriptError> {
        let Some(bits) = NonZeroU8::new(bits) else {
            return Ok(0);
        };
        let nonce_bits = nonce_bits(bits).ok_or(TranscriptError::UnsupportedGrinding);
        let nonce_bits = self.check(nonce_bits)?;
        let nonce = self.receive_nonce(nonce_bits)?;
        let predicate: [u8; GRINDING_PREDICATE_LEN] = self.challenge_bytes();
        let accepted = if grinding_predicate_accepts(&predicate, bits) {
            Ok(nonce)
        } else {
            Err(TranscriptError::GrindingRejected)
        };
        self.check(accepted)
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
        self.duplex.events()
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
        } else if self.remaining() != 0 {
            Err(TranscriptError::TrailingBytes)
        } else {
            Ok(())
        }
    }

    /// The next `len` unread bytes, without consuming them.
    fn peek(&mut self, len: usize) -> Result<&'a [u8], TranscriptError> {
        self.check(Ok(()))?;
        let narg = self.narg;
        let bytes = narg
            .get(self.consumed..)
            .and_then(|rest| rest.get(..len))
            .ok_or(TranscriptError::Truncated);
        self.check(bytes)
    }

    /// Consumes and absorbs the next `len` bytes, which `peek` already bounded.
    fn advance(&mut self, len: usize) {
        let narg = self.narg;
        let (_, rest) = narg.split_at(self.consumed);
        let (bytes, _) = rest.split_at(len);
        self.duplex.absorb_message(bytes, self.consumed);
        self.consumed += len;
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
        self.duplex.set_site(site);
    }

    fn public<A: CanonicalBytes>(&mut self, value: &A) {
        self.duplex.absorb_public_atoms(std::slice::from_ref(value));
    }

    fn public_all<A: CanonicalBytes>(&mut self, values: &[A]) {
        self.duplex.absorb_public_atoms(values);
    }

    fn public_bytes(&mut self, bytes: &[u8]) {
        self.duplex.absorb_public_framed(bytes);
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
        self.duplex.challenge()
    }

    fn challenge_small<F: CanonicalEncoding>(&mut self) -> F {
        self.duplex.challenge_small()
    }

    fn challenge_bytes<const N: usize>(&mut self) -> [u8; N] {
        self.duplex.squeeze_array()
    }

    fn preview(&self) -> Preview<H> {
        self.duplex.preview()
    }
}
