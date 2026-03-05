//! Core traits for Fiat-Shamir transcript transformation.
//!
//! This module provides the [`Transcript`] trait for building Fiat-Shamir transcripts
//! and the [`AppendToTranscript`] trait for types that can be absorbed into a transcript.

/// Fiat-Shamir transcript for non-interactive proofs.
///
/// A transcript absorbs data and produces deterministic challenges. Both prover
/// and verifier maintain identical transcripts to derive the same challenges,
/// transforming an interactive proof into a non-interactive one.
///
/// # Security
///
/// Domain separation is provided via the label in [`new`](Transcript::new).
/// Use unique labels per protocol to prevent cross-protocol attacks.
pub trait Transcript: Default + Clone + Sync + Send + 'static {
    /// The challenge type produced by this transcript.
    ///
    /// Typically a Field element. Caller is responsible for serialization
    /// of the Challenge type
    type Challenge: Copy + Default;

    /// Creates a new transcript with the given domain separation label.
    ///
    /// # Panics
    ///
    /// Panics if `label.len() >= 33`.
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

    /// Squeezes a challenge from the transcript.
    ///
    /// Each call produces a new challenge and advances the transcript state.
    #[must_use]
    fn challenge(&mut self) -> Self::Challenge;

    /// Squeezes multiple challenges from the transcript.
    #[must_use]
    fn challenge_vector(&mut self, len: usize) -> Vec<Self::Challenge> {
        (0..len).map(|_| self.challenge()).collect()
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

/// Implement this trait to define how your type serializes into transcript bytes.
/// This keeps the [`Transcript`] trait decoupled from specific serialization formats.
pub trait AppendToTranscript {
    /// Absorbs this value into the transcript.
    fn append_to_transcript<T: Transcript>(&self, transcript: &mut T);
}
