//! The prover's end: absorb every message exactly as it is written.

use std::num::NonZeroU8;

use jolt_field::{CanonicalBytes, CanonicalDecode, CanonicalEncoding, Field};

use crate::duplex::Duplex;
use crate::grinding::{encode_nonce, nonce_bits, search_nonce, GRINDING_PREDICATE_LEN};
#[cfg(feature = "logging")]
use crate::TranscriptEvent;
use crate::{Channel, Preview, ProtocolId, SiteId, Sponge, TranscriptError};

/// Prover transcript: a sponge plus the argument string it writes.
///
/// Every prover message is appended to the argument string and the appended
/// bytes are absorbed, so the bytes the verifier reads are exactly the bytes
/// that bound the following challenges.
#[derive(Clone, Debug)]
pub struct ProverTranscript<H> {
    duplex: Duplex<H>,
    narg: Vec<u8>,
}

impl<H: Sponge> ProverTranscript<H> {
    /// Starts a transcript for `protocol`, bound to `session`.
    #[must_use]
    pub fn new(protocol: &ProtocolId, session: &[u8]) -> Self {
        Self {
            duplex: Duplex::new(protocol, session),
            narg: Vec::new(),
        }
    }

    /// Sends one atom.
    pub fn send<A: CanonicalBytes>(&mut self, value: &A) {
        let start = self.narg.len();
        self.narg.resize(start + A::NUM_BYTES, 0);
        value.to_bytes_le(self.narg.split_at_mut(start).1);
        self.absorb_written(start);
    }

    /// Sends atoms in order.
    pub fn send_all<A: CanonicalBytes>(&mut self, values: &[A]) {
        let start = self.narg.len();
        self.narg.resize(start + A::NUM_BYTES * values.len(), 0);
        for (value, out) in values.iter().zip(
            self.narg
                .split_at_mut(start)
                .1
                .chunks_exact_mut(A::NUM_BYTES),
        ) {
            value.to_bytes_le(out);
        }
        self.absorb_written(start);
    }

    /// Sends bytes whose length the verifier already knows.
    pub fn send_bytes(&mut self, bytes: &[u8]) {
        let start = self.narg.len();
        self.narg.extend_from_slice(bytes);
        self.absorb_written(start);
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

    /// Sends a nonce in its canonical variable-length encoding.
    pub fn send_nonce(&mut self, nonce: u32) {
        self.send_bytes(encode_nonce(nonce).as_slice());
    }

    /// Grinds `bits` bits of proof of work and returns the committed nonce.
    /// A zero difficulty sends nothing.
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
        let nonce = search_nonce(bits, nonce_bits, |nonce| {
            let mut preview = self.preview();
            preview.absorb_nonce(nonce);
            preview.squeeze::<GRINDING_PREDICATE_LEN>()
        })
        .ok_or(TranscriptError::GrindingExhausted)?;
        self.send_nonce(nonce);
        let _predicate: [u8; GRINDING_PREDICATE_LEN] = self.challenge_bytes();
        Ok(nonce)
    }

    /// The argument string written so far.
    #[must_use]
    pub fn narg(&self) -> &[u8] {
        &self.narg
    }

    /// Recorded operations, in order.
    #[cfg(feature = "logging")]
    #[must_use]
    pub fn events(&self) -> &[TranscriptEvent] {
        self.duplex.events()
    }

    /// Ends the transcript, returning the proof.
    #[must_use]
    pub fn finish(self) -> Vec<u8> {
        self.narg
    }

    fn absorb_written(&mut self, start: usize) {
        let written = self.narg.split_at(start).1;
        self.duplex.absorb_message(written, start);
    }
}

impl<H: Sponge> Channel for ProverTranscript<H> {
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
