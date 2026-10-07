//! The prover's end: spongefish's prover state, written through typed atoms.

use core::fmt::Result as FmtResult;
use std::num::NonZeroU8;

use jolt_field::{CanonicalBytes, CanonicalDecode, CanonicalEncoding, Field};
use rand::rngs::StdRng;
use spongefish::ProverState;

use crate::grinding::{grind_nonce, nonce_bits, GRINDING_SEED_LEN};
use crate::site::{Log, TranscriptOp};
use crate::state::SMALL_CHALLENGE_BYTES;
use crate::state::{domain, Framed, Squeeze, BYTE_BLOCK};
#[cfg(feature = "logging")]
use crate::TranscriptEvent;
use crate::{Channel, Nonce, ProtocolId, SiteId, Sponge, TranscriptError};
use core::fmt::Debug;
use core::fmt::Formatter;

/// Prover transcript: spongefish's [`ProverState`], which appends every prover
/// message to the argument string and absorbs exactly its encoding.
///
/// Not `Clone`: copying a prover state lets a caller rewind the sponge and
/// re-draw challenges.
pub struct ProverTranscript<H: Sponge> {
    state: ProverState<H, StdRng>,
    log: Log,
}

impl<H: Sponge> Debug for ProverTranscript<H> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.debug_struct("ProverTranscript")
            .field("narg_len", &self.state.narg_string().len())
            .finish_non_exhaustive()
    }
}

impl<H: Sponge> ProverTranscript<H> {
    /// Starts a transcript for `protocol`, bound to `session`.
    #[must_use]
    pub fn new(protocol: &ProtocolId, session: &[u8]) -> Self {
        Self {
            state: domain(protocol, session).to_prover(H::default()),
            log: Log::default(),
        }
    }

    /// Sends one atom.
    pub fn send<A: CanonicalBytes>(&mut self, value: &A) {
        let start = self.narg_len();
        self.state.prover_message(value);
        self.record_message(start);
    }

    /// Sends atoms in order.
    pub fn send_all<A: CanonicalBytes>(&mut self, values: &[A]) {
        let start = self.narg_len();
        self.state.prover_messages(values);
        self.record_message(start);
    }

    /// Sends bytes whose length the verifier already knows.
    pub fn send_bytes(&mut self, bytes: &[u8]) {
        let start = self.narg_len();
        let (blocks, rest) = bytes.as_chunks::<BYTE_BLOCK>();
        for block in blocks {
            self.state.prover_message(block);
        }
        for byte in rest {
            self.state.prover_message(&[*byte]);
        }
        self.record_message(start);
    }

    /// Sends a byte string of at most `max_len` bytes, prefixed by its `u32`
    /// little-endian length.
    ///
    /// # Errors
    ///
    /// [`TranscriptError::OutOfBounds`] if `bytes` exceeds `max_len` or `u32`.
    pub fn send_bounded_bytes(
        &mut self,
        bytes: &[u8],
        max_len: usize,
    ) -> Result<(), TranscriptError> {
        if bytes.len() > max_len {
            return Err(TranscriptError::OutOfBounds);
        }
        let len = u32::try_from(bytes.len()).map_err(|_| TranscriptError::OutOfBounds)?;
        self.send(&len);
        self.send_bytes(bytes);
        Ok(())
    }

    /// Sends a search counter as a [`Nonce`] message.
    pub fn send_nonce(&mut self, nonce: u32) {
        let start = self.narg_len();
        self.state.prover_message(&Nonce(nonce));
        self.record_message(start);
    }

    /// Grinds `bits` bits of proof of work and returns the sent nonce. A zero
    /// difficulty draws and sends nothing.
    ///
    /// Squeezes a seed, searches nonces on [`Fork`](crate::Fork)s, then sends
    /// the first accepted nonce as a [`Nonce`]; the protected challenge is
    /// drawn after.
    ///
    /// # Errors
    ///
    /// [`TranscriptError::UnsupportedGrinding`] above
    /// [`MAX_GRINDING_BITS`](crate::MAX_GRINDING_BITS), and
    /// [`TranscriptError::GrindingExhausted`] if no nonce in range succeeds.
    pub fn grind(&mut self, bits: u8) -> Result<u32, TranscriptError> {
        let Some(bits) = NonZeroU8::new(bits) else {
            return Ok(0);
        };
        let nonce_bits = nonce_bits(bits).ok_or(TranscriptError::UnsupportedGrinding)?;
        let seed: [u8; GRINDING_SEED_LEN] = self.challenge_bytes();
        let nonce =
            grind_nonce::<H>(&seed, bits, nonce_bits).ok_or(TranscriptError::GrindingExhausted)?;
        self.send_nonce(nonce);
        Ok(nonce)
    }

    /// The argument string written so far.
    #[must_use]
    pub fn narg(&self) -> &[u8] {
        self.state.narg_string()
    }

    /// Recorded operations, in order.
    #[cfg(feature = "logging")]
    #[must_use]
    pub fn events(&self) -> &[TranscriptEvent] {
        self.log.events()
    }

    /// Ends the transcript, returning the proof.
    #[must_use]
    pub fn finish(self) -> Vec<u8> {
        self.state.narg_string().to_vec()
    }

    fn narg_len(&self) -> usize {
        self.state.narg_string().len()
    }

    fn record_message(&mut self, start: usize) {
        let end = self.narg_len();
        self.log
            .record(TranscriptOp::Message, end - start, Some(start..end));
    }
}

impl<H: Sponge> Channel for ProverTranscript<H> {
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
        self.send(value);
        Ok(())
    }

    fn exchange_all<A: CanonicalDecode>(
        &mut self,
        values: &mut [A],
    ) -> Result<(), TranscriptError> {
        self.send_all(values);
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
            .record(TranscriptOp::Challenge, SMALL_CHALLENGE_BYTES, None);
        value
    }

    fn challenge_bytes<const N: usize>(&mut self) -> [u8; N] {
        let value = self.state.squeeze_array();
        self.log.record(TranscriptOp::Challenge, N, None);
        value
    }
}
