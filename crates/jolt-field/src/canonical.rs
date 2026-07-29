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

#[cfg(test)]
mod tests {
    #[cfg(any(feature = "bn254", feature = "akita"))]
    use super::CanonicalRepr;

    /// Every scalar-challenge field must use the legacy transcript
    /// convention the Blake2b transcripts squeeze against: interpret the
    /// digest as a big-endian integer (reverse the bytes, then reduce
    /// little-endian). A prover field and verifier field diverging here
    /// surfaces as an opaque stage-claim mismatch deep in an e2e test — this
    /// pins every implementation to one formula.
    #[cfg(any(feature = "bn254", feature = "akita"))]
    fn assert_legacy_scalar_convention<F: CanonicalRepr>() {
        let mut low_byte_set = [0u8; 16];
        low_byte_set[0] = 1;
        let probes: [[u8; 16]; 4] = [[0u8; 16], low_byte_set, *b"jolt-fiat-shamir", [0xff; 16]];
        for probe in probes {
            let mut reversed = probe;
            reversed.reverse();
            assert_eq!(
                F::from_scalar_challenge_bytes(&probe),
                F::from_le_bytes_mod_order(&reversed),
                "scalar challenge must reduce the byte-reversed digest"
            );
        }
        // Direction sensitivity: an asymmetric digest must not decode the
        // same unreversed, or the reversal has been silently dropped.
        assert_ne!(
            F::from_scalar_challenge_bytes(&low_byte_set),
            F::from_le_bytes_mod_order(&low_byte_set),
            "scalar challenge convention must be direction-sensitive"
        );
    }

    #[cfg(feature = "bn254")]
    #[test]
    fn fr_uses_the_legacy_scalar_challenge_convention() {
        assert_legacy_scalar_convention::<crate::Fr>();
    }

    #[cfg(feature = "bn254")]
    #[test]
    fn fq_uses_the_legacy_scalar_challenge_convention() {
        assert_legacy_scalar_convention::<crate::Fq>();
    }

    #[cfg(feature = "akita")]
    #[test]
    fn akita_field_matches_the_legacy_scalar_challenge_convention() {
        assert_legacy_scalar_convention::<akita_config::proof_optimized::fp128::Field>();
    }
}
