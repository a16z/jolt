//! Core traits for Fiat-Shamir transcript transformation.
//!
//! This module provides the [`Transcript`] trait for building Fiat-Shamir transcripts
//! and the [`AppendToTranscript`] trait for types that can be absorbed into a transcript.

use crate::domain::Label;
use jolt_field::{Field, FromPrimitiveInt};

/// Fiat-Shamir transcript for non-interactive proofs.
///
/// A transcript absorbs data and produces deterministic challenges. Both prover
/// and verifier maintain identical transcripts to derive the same challenges,
/// transforming an interactive proof into a non-interactive one.
///
/// Hash-based transcripts (`Blake2bTranscript`, `KeccakTranscript`) are generic
/// over `F: Field` and produce field-element challenges directly.
///
/// # Security
///
/// Domain separation is provided via the label in [`new`](Transcript::new).
/// Use unique labels per protocol to prevent cross-protocol attacks.
pub trait Transcript: Default + Clone + Sync + Send + 'static {
    /// The challenge type produced by this transcript.
    ///
    /// For hash-based transcripts this is `F` (the field type), so challenges
    /// can be used directly in polynomial operations without conversion.
    type Challenge: Copy + Default + PartialEq + Eq + std::fmt::Debug + std::hash::Hash;

    /// Creates a new transcript with the given domain separation label.
    ///
    /// # Panics
    ///
    /// Panics if `label.len() > 32`.
    fn new(label: &'static [u8]) -> Self;

    /// Absorbs raw bytes into the transcript.
    ///
    /// Prefer [`append`](Transcript::append)
    /// for a type-safe/ergonomic absorption of data.
    fn append_bytes(&mut self, bytes: &[u8]);

    /// Absorbs a value into the transcript.
    ///
    /// This is the primary method for adding data to the transcript. Any type
    /// implementing [`AppendToTranscript`] can be absorbed.
    fn append<A: AppendToTranscript>(&mut self, value: &A) {
        value.append_to_transcript(self);
    }

    /// Absorbs a domain label followed by a value.
    ///
    /// Jolt's core proof transcript commonly absorbs scalar payloads as
    /// `label || payload`; this method makes that pattern explicit at the
    /// transcript API boundary.
    fn append_labeled<A: AppendToTranscript>(&mut self, label: &'static [u8], value: &A) {
        self.append(&Label(label));
        self.append(value);
    }

    /// Squeezes a challenge from the transcript.
    ///
    /// Each call produces a new challenge and advances the transcript state.
    #[must_use]
    fn challenge(&mut self) -> Self::Challenge;

    /// Squeezes a non-optimized scalar challenge from the transcript.
    #[must_use]
    fn challenge_scalar(&mut self) -> Self::Challenge {
        self.challenge()
    }

    /// Squeezes multiple challenges from the transcript.
    #[must_use]
    fn challenge_vector(&mut self, len: usize) -> Vec<Self::Challenge> {
        (0..len).map(|_| self.challenge()).collect()
    }

    /// Squeezes one scalar challenge and returns its powers `[1, gamma, gamma^2, ...]`.
    #[must_use]
    fn challenge_scalar_powers(&mut self, len: usize) -> Vec<Self::Challenge>
    where
        Self::Challenge: Field,
    {
        let gamma = self.challenge_scalar();
        let mut powers = vec![Self::Challenge::from_u64(1); len];
        for index in 1..len {
            powers[index] = powers[index - 1] * gamma;
        }
        powers
    }

    /// Returns the current 256-bit transcript state.
    ///
    /// Useful for debugging and testing transcript synchronization.
    #[must_use]
    fn state(&self) -> &[u8; 32];

    /// Enables transcript comparison for testing.
    ///
    /// After calling this, the transcript will panic if its state ever diverges
    /// from the expected state history recorded in `other`.
    #[cfg(test)]
    fn compare_to(&mut self, other: &Self);
}

/// Maximum label length in bytes. Labels are padded to this size before hashing.
pub const MAX_LABEL_LEN: usize = 32;

/// Implement this trait to define how your type serializes into transcript bytes.
/// This keeps the [`Transcript`] trait decoupled from specific serialization formats.
pub trait AppendToTranscript {
    /// Absorbs this value into the transcript.
    fn append_to_transcript<T: Transcript>(&self, transcript: &mut T);

    /// Byte length of the payload absorbed by [`append_to_transcript`].
    ///
    /// Types that need to match `jolt-core`'s variable-length labeled
    /// transcript methods should override this so callers can prepend the same
    /// packed label/length word before absorbing the payload.
    fn transcript_payload_len(&self) -> Option<u64> {
        None
    }
}
