//! Generic digest-based Fiat-Shamir transcript.
//!
//! Provides [`DigestTranscript`], a Fiat-Shamir transcript backed by any
//! 256-bit hash function implementing [`Digest`]. Concrete instantiations
//! (Blake2b, Keccak) are type aliases in their respective modules.

use digest::{consts::U32, Digest};

use crate::transcript::{Transcript, MAX_LABEL_LEN};

#[cfg(test)]
#[derive(Clone, Default)]
struct TestState {
    state_history: Vec<[u8; 32]>,
    expected_state_history: Option<Vec<[u8; 32]>>,
}

/// Fiat-Shamir transcript backed by a 256-bit digest.
///
/// Generic over the hash function `D` and field type `F`. Challenges are
/// produced as field elements directly via `F::from_u128()`.
pub struct DigestTranscript<
    D: Digest<OutputSize = U32> + 'static,
    F: jolt_field::Field = jolt_field::Fr,
> {
    state: [u8; 32],
    n_rounds: u32,
    #[cfg(test)]
    test_state: TestState,
    _marker: std::marker::PhantomData<(fn() -> D, F)>,
}

impl<D: Digest<OutputSize = U32>, F: jolt_field::Field> Clone for DigestTranscript<D, F> {
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

impl<D: Digest<OutputSize = U32>, F: jolt_field::Field> Default for DigestTranscript<D, F> {
    fn default() -> Self {
        Self::new(b"")
    }
}

impl<D: Digest<OutputSize = U32>, F: jolt_field::Field> std::fmt::Debug for DigestTranscript<D, F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DigestTranscript")
            .field("state", &format_args!("{:02x?}", self.state))
            .field("n_rounds", &self.n_rounds)
            .finish_non_exhaustive()
    }
}

impl<D: Digest<OutputSize = U32>, F: jolt_field::Field> DigestTranscript<D, F> {
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
        let hash: [u8; 32] = self
            .hasher()
            .chain_update([0x01]) // squeeze domain tag
            .finalize()
            .into();
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

impl<D: Digest<OutputSize = U32>, F: jolt_field::Field> Transcript for DigestTranscript<D, F> {
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
        let hash: [u8; 32] = self
            .hasher()
            .chain_update([0x00]) // absorb domain tag
            .chain_update(bytes)
            .finalize()
            .into();
        self.update_state(hash);
    }

    fn challenge(&mut self) -> F {
        let mut buf = [0u8; 16];
        self.challenge_bytes(&mut buf);
        F::from_u128(u128::from_le_bytes(buf))
    }

    #[inline]
    fn state(&self) -> &[u8; 32] {
        &self.state
    }

    #[cfg(test)]
    fn compare_to(&mut self, other: &Self) {
        self.test_state.expected_state_history = Some(other.test_state.state_history.clone());
    }
}
