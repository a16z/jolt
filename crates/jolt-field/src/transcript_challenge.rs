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
