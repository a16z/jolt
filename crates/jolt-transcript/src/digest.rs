//! Generic digest-based Fiat-Shamir transcript, byte-compatible with
//! `jolt-prover-legacy`'s hash transcripts.
//!
//! Provides [`DigestTranscript`], used to verify proofs produced by
//! `jolt-prover-legacy` provers: appends hash `state || round || payload`, squeezes
//! hash `state || round`, and challenges use the same decoding paths as
//! `jolt-prover-legacy`. The spongefish-backed [`crate::SpongeTranscript`] is the
//! native transcript; this one exists for the core-compat boundary.

use digest::{consts::U32, Digest};

use crate::legacy::{Transcript, MAX_LABEL_LEN};

#[cfg(test)]
#[derive(Clone, Default)]
struct TestState {
    state_history: Vec<[u8; 32]>,
    expected_state_history: Option<Vec<[u8; 32]>>,
}

/// Fiat-Shamir transcript backed by a 256-bit digest.
///
/// Generic over the hash function `D` and field type `F`. Challenges are
/// produced as field elements through `F::from_challenge_bytes`.
pub struct DigestTranscript<D: Digest<OutputSize = U32> + 'static, F> {
    state: [u8; 32],
    n_rounds: u32,
    #[cfg(test)]
    test_state: TestState,
    _marker: std::marker::PhantomData<(fn() -> D, F)>,
}

impl<D, F> Clone for DigestTranscript<D, F>
where
    D: Digest<OutputSize = U32>,
    F: jolt_field::TranscriptChallenge,
{
    fn clone(&self) -> Self {
        Self {
            state: self.state,
            n_rounds: self.n_rounds,
            #[cfg(test)]
            test_state: self.test_state.clone(),
            _marker: std::marker::PhantomData,
        }
    }
}

/// Exists only because [`Transcript`] requires `Default`.
///
/// WARNING: not byte-compatible with `jolt-prover-legacy`'s derived `Default` (zero
/// state, no initial hash); use [`Transcript::new`] for core-compatible
/// transcripts.
impl<D, F> Default for DigestTranscript<D, F>
where
    D: Digest<OutputSize = U32>,
    F: jolt_field::TranscriptChallenge,
{
    fn default() -> Self {
        Self::new(b"")
    }
}

impl<D, F> std::fmt::Debug for DigestTranscript<D, F>
where
    D: Digest<OutputSize = U32>,
    F: jolt_field::TranscriptChallenge,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DigestTranscript")
            .field("state", &format_args!("{:02x?}", self.state))
            .field("n_rounds", &self.n_rounds)
            .finish_non_exhaustive()
    }
}

impl<D, F> DigestTranscript<D, F>
where
    D: Digest<OutputSize = U32>,
    F: jolt_field::TranscriptChallenge,
{
    /// Raw multi-byte squeeze backing jolt-prover-legacy's challenge
    /// decoding, so its legacy `Transcript` vocabulary can drive this engine
    /// directly (no state handoffs). Hidden: protocol code squeezes through
    /// the [`Transcript`] challenge methods.
    #[doc(hidden)]
    pub fn raw_challenge_bytes(&mut self, out: &mut [u8]) {
        self.challenge_bytes(out);
    }

    #[inline]
    fn hasher(&self) -> D {
        let mut round_bytes = [0u8; 32];
        round_bytes[28..].copy_from_slice(&self.n_rounds.to_be_bytes());
        D::new().chain_update(self.state).chain_update(round_bytes)
    }

    fn challenge_bytes(&mut self, out: &mut [u8]) {
        let mut remaining = out.len();
        let mut offset = 0;

        while remaining > 32 {
            let mut chunk = [0u8; 32];
            self.challenge_bytes32(&mut chunk);
            out[offset..offset + 32].copy_from_slice(&chunk);
            offset += 32;
            remaining -= 32;
        }

        let mut final_chunk = [0u8; 32];
        self.challenge_bytes32(&mut final_chunk);
        out[offset..offset + remaining].copy_from_slice(&final_chunk[..remaining]);
    }

    #[inline]
    fn challenge_bytes32(&mut self, out: &mut [u8; 32]) {
        let hash: [u8; 32] = self.hasher().finalize().into();
        out.copy_from_slice(&hash);
        self.update_state(hash);
    }

    fn update_state(&mut self, new_state: [u8; 32]) {
        self.state = new_state;
        self.n_rounds += 1;

        #[cfg(test)]
        {
            if let Some(ref expected) = self.test_state.expected_state_history {
                assert_eq!(
                    new_state, expected[self.n_rounds as usize],
                    "Fiat-Shamir transcript mismatch at round {}",
                    self.n_rounds
                );
            }
            self.test_state.state_history.push(new_state);
        }
    }
}

impl<D, F> Transcript for DigestTranscript<D, F>
where
    D: Digest<OutputSize = U32>,
    F: jolt_field::TranscriptChallenge,
{
    type Challenge = F;

    fn new(label: &'static [u8]) -> Self {
        assert!(
            label.len() <= MAX_LABEL_LEN,
            "label must be at most {MAX_LABEL_LEN} bytes",
        );

        let mut padded = [0u8; MAX_LABEL_LEN];
        padded[..label.len()].copy_from_slice(label);

        let hash: [u8; 32] = D::new().chain_update(padded).finalize().into();

        Self {
            state: hash,
            n_rounds: 0,
            #[cfg(test)]
            test_state: TestState {
                state_history: vec![hash],
                expected_state_history: None,
            },
            _marker: std::marker::PhantomData,
        }
    }

    fn append_bytes(&mut self, bytes: &[u8]) {
        let hash: [u8; 32] = self.hasher().chain_update(bytes).finalize().into();
        self.update_state(hash);
    }

    fn challenge(&mut self) -> F {
        let mut buf = [0u8; 16];
        self.challenge_bytes(&mut buf);
        F::from_challenge_bytes(&buf)
    }

    fn challenge_scalar(&mut self) -> F {
        let mut buf = [0u8; 16];
        self.challenge_bytes(&mut buf);
        F::from_scalar_challenge_bytes(&buf)
    }

    #[inline]
    fn state(&self) -> [u8; 32] {
        self.state
    }

    #[cfg(test)]
    fn compare_to(&mut self, other: &Self) {
        self.test_state.expected_state_history = Some(other.test_state.state_history.clone());
    }
}

#[cfg(all(test, feature = "transcript-blake2b"))]
mod tests {
    use super::*;
    use jolt_field::Fr;

    type TestTranscript = DigestTranscript<blake2::Blake2b<digest::consts::U32>, Fr>;

    /// `compare_to` arms the replaying transcript with the reference state
    /// history; an identical operation sequence must replay clean.
    #[test]
    fn compare_to_accepts_an_identical_replay() {
        let mut reference = TestTranscript::new(b"compare");
        reference.append_bytes(b"round-1");
        let _ = reference.challenge();
        reference.append_bytes(b"round-2");

        let mut replay = TestTranscript::new(b"compare");
        replay.compare_to(&reference);
        replay.append_bytes(b"round-1");
        let _ = replay.challenge();
        replay.append_bytes(b"round-2");
        assert_eq!(replay.state(), reference.state());
    }

    #[test]
    #[should_panic(expected = "Fiat-Shamir transcript mismatch at round 1")]
    fn compare_to_panics_at_the_first_divergent_round() {
        let mut reference = TestTranscript::new(b"compare");
        reference.append_bytes(b"expected payload");

        let mut replay = TestTranscript::new(b"compare");
        replay.compare_to(&reference);
        replay.append_bytes(b"divergent payload");
    }
}
