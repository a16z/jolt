//! Canonical byte representation: the Fiat-Shamir transcript surface.

use std::fmt::Debug;
use std::hash::Hash;

/// Canonical little-endian representation of a field element.
///
/// This trait is the transcript surface: Fiat-Shamir absorption and challenge
/// derivation use these explicit canonical encodings so the hashed byte
/// stream is specified independently of any serialization library. Proof and
/// wire serialization go through serde + bincode instead; the two must not be
/// conflated.
///
/// # Invariants
///
/// - The encoding is injective on canonical representatives: equal elements
///   produce equal bytes, distinct elements produce distinct bytes.
/// - [`to_bytes_le`](Self::to_bytes_le) always writes exactly
///   [`NUM_BYTES`](Self::NUM_BYTES) bytes of the unique representative.
pub trait CanonicalRepr:
    Sized + Copy + Default + PartialEq + Eq + Debug + Hash + Sync + Send + 'static
{
    /// Byte length of the fixed-size canonical encoding.
    const NUM_BYTES: usize;

    /// Writes the canonical little-endian encoding into `out`.
    fn to_bytes_le(&self, out: &mut [u8]);

    /// Returns the canonical little-endian encoding as a vector.
    #[inline]
    fn to_bytes_le_vec(&self) -> Vec<u8> {
        let mut out = vec![0u8; Self::NUM_BYTES];
        self.to_bytes_le(&mut out);
        out
    }

    /// Deserializes little-endian bytes by reducing into this type.
    fn from_le_bytes_mod_order(bytes: &[u8]) -> Self;

    /// Returns the canonical representative as `u64` if it fits.
    fn to_canonical_u64_checked(&self) -> Option<u64>;

    /// Number of significant bits in this element's canonical representative.
    ///
    /// Zero is considered to have zero significant bits.
    fn num_bits(&self) -> u32;

    /// Constructs a Fiat-Shamir challenge from squeezed transcript bytes.
    #[inline]
    fn from_challenge_bytes(bytes: &[u8]) -> Self {
        Self::from_le_bytes_mod_order(bytes)
    }

    /// Constructs a non-optimized scalar challenge from transcript bytes.
    #[inline]
    fn from_scalar_challenge_bytes(bytes: &[u8]) -> Self {
        Self::from_challenge_bytes(bytes)
    }
}
