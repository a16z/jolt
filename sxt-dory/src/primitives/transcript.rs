//! Transcript trait for fiat shamir
use crate::arithmetic::Field;
use ark_serialize::CanonicalSerialize;

/// Transcript to standardize fiat shamir for generic concrete transcripts
pub trait Transcript {
    /// The scalar field associated with the transcript, matching the groups used.
    type Scalar: Field;

    /// Appends arbitrary bytes to the transcript.
    ///
    /// # Arguments
    ///
    /// * `label` - A domain-separation label for the bytes.
    /// * `bytes` - The byte slice to append.
    fn append_bytes(&mut self, label: &[u8], bytes: &[u8]);

    /// Appends a single field element to the transcript.
    /// The field element is typically serialized to bytes before appending.
    ///
    /// # Arguments
    ///
    /// * `label` - A domain-separation label for the field element.
    /// * `x` - The field element to append.
    fn append_field(&mut self, label: &[u8], x: &Self::Scalar);

    /// Appends a group element to the transcript.
    /// The group element is typically serialized in its compressed form.
    ///
    /// # Arguments
    ///
    /// * `label` - A domain-separation label for the group element.
    /// * `g` - The group element to append. It must implement `CanonicalSerialize`.
    fn append_group<G: CanonicalSerialize>(&mut self, label: &[u8], g: &G);

    /// Appends any `serde`-serializable element to the transcript.
    ///
    /// # Arguments
    ///
    /// * `label` - A domain-separation label for the element.
    /// * `s` - The element to append. It must implement `serde::Serialize`.
    fn append_serde<S: serde::Serialize>(&mut self, label: &[u8], s: &S);

    /// Produces a challenge derived from the transcript
    /// The scalar should be non-zero
    ///
    /// # Arguments
    ///
    /// * `label` - A domain-separation label for this challenge.
    ///
    /// # Returns
    ///
    /// A scalar derived from the transcript's current state.
    fn challenge_scalar(&mut self, label: &[u8]) -> Self::Scalar;

    /// Resets the transcript to its initial state with the given domain.
    ///
    /// # Arguments
    ///
    /// * `domain_label` - A domain-separation label for the transcript.
    fn reset(&mut self, domain_label: &[u8]);
}
