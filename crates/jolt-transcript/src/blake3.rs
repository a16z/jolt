//! Streaming BLAKE3 Fiat-Shamir transcript.
//!
//! Appends stream into one running keyed BLAKE3 chunk; the running chaining
//! value is the transcript state. A squeeze finalizes the pending block with
//! the ROOT flag, reads 64 bytes of root output, and splits them: bytes
//! `0..32` (the standard 32-byte digest) become the next state, which keys
//! the next segment; bytes `32..48` are the challenge. Appends never exceed
//! one chunk (1024 bytes) per segment: a full chunk is closed the same way,
//! minus the challenge, so every compression in the chain is a plain block
//! compression — no tree parents.
//!
//! Compression count of a segment of `n` pending bytes ending in a squeeze or
//! a chunk close: `max(1, ceil(n / 64))` — the finalize is the last block's
//! compression, not an extra one. An empty append absorbs nothing; payload
//! delimiting is the protocol layer's job (`LabelWithCount`).

use std::fmt::{self, Debug, Formatter};
use std::marker::PhantomData;

use blake3::Hasher;
use jolt_field::{CanonicalEncoding, Fr};

use crate::legacy::{Transcript, MAX_LABEL_LEN};

/// One BLAKE3 chunk: the most a keyed segment absorbs before it is closed.
const SEGMENT_BYTES: usize = 1024;

/// Fiat-Shamir transcript over streaming keyed BLAKE3 segments.
pub struct Blake3Transcript<F = Fr> {
    hasher: Hasher,
    pending: usize,
    _field: PhantomData<F>,
}

impl<F: CanonicalEncoding> Clone for Blake3Transcript<F> {
    fn clone(&self) -> Self {
        Self {
            hasher: self.hasher.clone(),
            pending: self.pending,
            _field: PhantomData,
        }
    }
}

impl<F: CanonicalEncoding> Default for Blake3Transcript<F> {
    fn default() -> Self {
        Self::new(b"")
    }
}

impl<F: CanonicalEncoding> Debug for Blake3Transcript<F> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("Blake3Transcript")
            .field("state", &format_args!("{:02x?}", self.state()))
            .field("pending", &self.pending)
            .finish_non_exhaustive()
    }
}

impl<F: CanonicalEncoding> Blake3Transcript<F> {
    fn rekey(&mut self, state: &[u8; 32]) {
        self.hasher = Hasher::new_keyed(state);
        self.pending = 0;
    }

    /// Closes the full chunk into the next state (no challenge drawn).
    fn close_segment(&mut self) {
        let state: [u8; 32] = self.hasher.finalize().into();
        self.rekey(&state);
    }

    fn squeeze16(&mut self) -> [u8; 16] {
        let mut out = [0u8; 64];
        self.hasher.finalize_xof().fill(&mut out);
        let mut state = [0u8; 32];
        let mut challenge = [0u8; 16];
        let (head, tail) = out.split_at(32);
        state.copy_from_slice(head);
        challenge.copy_from_slice(tail.split_at(16).0);
        self.rekey(&state);
        challenge
    }
}

impl<F: CanonicalEncoding> Transcript for Blake3Transcript<F> {
    type Challenge = F;

    fn new(label: &'static [u8]) -> Self {
        assert!(
            label.len() <= MAX_LABEL_LEN,
            "label must be at most {MAX_LABEL_LEN} bytes",
        );
        let mut padded = [0u8; MAX_LABEL_LEN];
        let (head, _) = padded.split_at_mut(label.len());
        head.copy_from_slice(label);
        let state: [u8; 32] = blake3::hash(&padded).into();
        Self {
            hasher: Hasher::new_keyed(&state),
            pending: 0,
            _field: PhantomData,
        }
    }

    fn append_bytes(&mut self, mut bytes: &[u8]) {
        while !bytes.is_empty() {
            if self.pending == SEGMENT_BYTES {
                self.close_segment();
            }
            let take = bytes.len().min(SEGMENT_BYTES - self.pending);
            let (head, tail) = bytes.split_at(take);
            let _ = self.hasher.update(head);
            self.pending += take;
            bytes = tail;
        }
    }

    fn challenge(&mut self) -> F {
        F::from_challenge_bytes(&self.squeeze16())
    }

    fn challenge_scalar(&mut self) -> F {
        F::from_scalar_challenge_bytes(&self.squeeze16())
    }

    /// The state a squeeze would chain from now: the keyed digest of the
    /// pending bytes.
    fn state(&self) -> [u8; 32] {
        self.hasher.finalize().into()
    }
}
