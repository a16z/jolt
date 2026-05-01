/// Fiat-Shamir challenge decoding from squeezed transcript bytes.
pub trait TranscriptChallenge: Sized {
    /// Constructs a challenge from transcript bytes.
    fn from_challenge_bytes(bytes: &[u8]) -> Self;
}
