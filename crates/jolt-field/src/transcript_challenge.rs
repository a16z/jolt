/// Fiat-Shamir challenge decoding from squeezed transcript bytes.
pub trait TranscriptChallenge:
    Sized + Copy + Default + PartialEq + Eq + std::fmt::Debug + std::hash::Hash + Sync + Send + 'static
{
    /// Constructs a challenge from transcript bytes.
    fn from_challenge_bytes(bytes: &[u8]) -> Self;

    /// Constructs a non-optimized scalar challenge from transcript bytes.
    fn from_scalar_challenge_bytes(bytes: &[u8]) -> Self {
        Self::from_challenge_bytes(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::TranscriptChallenge;
    use crate::ReducingBytes;

    /// Every scalar-challenge field must use the legacy transcript
    /// convention the Blake2b transcripts squeeze against: interpret the
    /// digest as a big-endian integer (reverse the bytes, then reduce
    /// little-endian). A prover field and verifier field diverging here
    /// surfaces as an opaque stage-claim mismatch deep in an e2e test — this
    /// pins every implementation to one formula.
    fn assert_legacy_scalar_convention<F>()
    where
        F: TranscriptChallenge + ReducingBytes,
    {
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

    #[test]
    fn fr_uses_the_legacy_scalar_challenge_convention() {
        assert_legacy_scalar_convention::<crate::Fr>();
    }

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
