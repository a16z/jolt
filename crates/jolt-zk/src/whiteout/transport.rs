//! Whiteout's tail messages in the proof's NARG string, and the attempt
//! fork its Σ-protocol keys each attempt's challenge with.
//!
//! The tail (the pad commitment, the Σ-protocol's `t_D` and nonce, the
//! response proof) runs in the same spongefish string as Akita's opening.
//! A run of field elements is one length-prefixed [`BytesMsg`] of their
//! canonical bytes; the verifier checks the length against the run its
//! schedule expects and rejects a non-canonical element.

use std::fmt::{Debug, Formatter, Result as FmtResult};

use jolt_algebra::CanonicalEncoding;
#[cfg(feature = "whiteout-prover")]
use jolt_transcript::ProverTranscript;
use jolt_transcript::{BytesMsg, VerifierTranscript};
use spongefish::DuplexSpongeInit;

use crate::masking::ZkError;

/// Byte length of the seed an attempt fork is keyed by.
pub(crate) const FORK_SEED_LEN: usize = 32;

/// Byte length of the secret nonce an attempt fork is keyed by.
pub(crate) const ATTEMPT_NONCE_LEN: usize = 32;

/// Session identifier of attempt forks: a sponge state no transcript starts
/// from, so the random-oracle analysis of the Fiat–Shamir-with-aborts step
/// treats the fork as its own oracle.
const ATTEMPT_SESSION: [u8; 32] = *b"jolt-zk/whiteout/attempt-fork/v1";

/// Sends `values` as one message of their canonical bytes.
#[cfg(feature = "whiteout-prover")]
pub(crate) fn send_values<F: CanonicalEncoding, T: ProverTranscript + ?Sized>(
    transcript: &mut T,
    values: &[F],
) {
    let mut bytes = vec![0u8; values.len() * F::NUM_BYTES];
    for (chunk, value) in bytes.chunks_exact_mut(F::NUM_BYTES).zip(values) {
        value.to_bytes_le(chunk);
    }
    transcript.send(&BytesMsg(bytes));
}

/// Reads one message of exactly `len` canonical field elements.
///
/// # Errors
///
/// [`ZkError::TranscriptRejected`] on a truncated string, a message of
/// another length, or a non-canonical element.
pub(crate) fn receive_values<F: CanonicalEncoding, T: VerifierTranscript + ?Sized>(
    transcript: &mut T,
    len: usize,
) -> Result<Vec<F>, ZkError> {
    let BytesMsg(bytes) = transcript.prover_message::<BytesMsg>()?;
    let expected = len
        .checked_mul(F::NUM_BYTES)
        .ok_or(ZkError::TranscriptRejected)?;
    if bytes.len() != expected {
        return Err(ZkError::TranscriptRejected);
    }
    bytes
        .chunks_exact(F::NUM_BYTES)
        .map(|chunk| F::from_bytes_le_checked(chunk).ok_or(ZkError::TranscriptRejected))
        .collect()
}

/// Sends `bytes` as one message.
#[cfg(feature = "whiteout-prover")]
pub(crate) fn send_bytes<T: ProverTranscript + ?Sized>(transcript: &mut T, bytes: &[u8]) {
    transcript.send(&BytesMsg(bytes));
}

/// Reads one message of exactly `len` bytes.
///
/// # Errors
///
/// [`ZkError::TranscriptRejected`] on a truncated string or another length.
pub(crate) fn receive_bytes<T: VerifierTranscript + ?Sized>(
    transcript: &mut T,
    len: usize,
) -> Result<Vec<u8>, ZkError> {
    let BytesMsg(bytes) = transcript.prover_message::<BytesMsg>()?;
    if bytes.len() == len {
        Ok(bytes)
    } else {
        Err(ZkError::TranscriptRejected)
    }
}

/// A fresh sponge of the transcript's kind keyed by a seed squeezed from the
/// live transcript and a secret attempt nonce.
///
/// Zero knowledge needs every rejected attempt's oracle input to be
/// unpredictable, so the nonce is drawn uniformly from the prover's RNG for
/// every attempt and stays secret until the prover sends it with the
/// accepted attempt (Whiteout, Thm `wo:thm:zk`, hybrid H2). The verifier
/// squeezes the same seed, reads the nonce, and rebuilds the same fork. No
/// transcript state is ever copied.
#[derive(Clone)]
pub(crate) struct AttemptFork<H>(H);

/// Redacted: an attempt fork is keyed by a secret nonce, so its sponge
/// state is never printed.
impl<H> Debug for AttemptFork<H> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.write_str("AttemptFork(..)")
    }
}

impl<H: DuplexSpongeInit<U = u8>> AttemptFork<H> {
    /// The fork of one attempt under `seed` and `nonce`. The caller absorbs
    /// the attempt's first message before squeezing its challenge.
    #[must_use]
    pub(crate) fn new(seed: &[u8; FORK_SEED_LEN], nonce: &[u8; ATTEMPT_NONCE_LEN]) -> Self {
        let mut sponge = H::init(&ATTEMPT_SESSION);
        let _ = sponge.absorb(seed);
        let _ = sponge.absorb(nonce);
        Self(sponge)
    }

    /// Absorbs `bytes`.
    pub(crate) fn absorb(&mut self, bytes: &[u8]) {
        let _ = self.0.absorb(bytes);
    }

    /// Squeezes `N` bytes.
    #[must_use]
    pub(crate) fn squeeze<const N: usize>(&mut self) -> [u8; N] {
        let mut out = [0u8; N];
        let _ = self.0.squeeze(&mut out);
        out
    }
}

#[cfg(test)]
mod tests {
    use jolt_transcript::Blake2b512;

    use super::*;

    fn attempt_challenge(seed: [u8; FORK_SEED_LEN], nonce: [u8; ATTEMPT_NONCE_LEN]) -> [u8; 32] {
        let mut fork = AttemptFork::<Blake2b512>::new(&seed, &nonce);
        fork.absorb(b"first message");
        fork.squeeze()
    }

    #[test]
    fn attempt_forks_are_keyed_by_seed_and_nonce() {
        let base = attempt_challenge([1; 32], [2; 32]);
        assert_eq!(base, attempt_challenge([1; 32], [2; 32]));
        assert_ne!(base, attempt_challenge([3; 32], [2; 32]));
        let mut nonce = [2; 32];
        nonce[31] ^= 1;
        assert_ne!(base, attempt_challenge([1; 32], nonce));
    }

    /// Known answer: the verifier rebuilds this fork from the transcript
    /// seed and the sent nonce, so its bytes are protocol.
    #[test]
    fn attempt_fork_known_answer() {
        let hex = attempt_challenge([1; 32], [2; 32])
            .iter()
            .flat_map(|byte| [byte >> 4, byte & 15])
            .filter_map(|nibble| char::from_digit(u32::from(nibble), 16))
            .collect::<String>();
        assert_eq!(
            hex,
            "3016da6edbc61e5b11637790c7d8adc9c220ad039b00c07fa2da4e655973cd0d"
        );
    }
}
