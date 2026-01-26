//! Macro for implementing the Transcript trait with different hash functions.

/// Implements the `Transcript` trait for a hash-based transcript.
macro_rules! impl_transcript {
    ($name:ident, $hasher:ty, $new_hasher:expr) => {
        use $crate::transcript::Transcript;

        /// Internal state for test-time transcript comparison.
        #[cfg(test)]
        #[derive(Clone, Default)]
        struct TestState {
            /// Complete history of transcript states.
            state_history: Vec<[u8; 32]>,
            /// Expected state history for verification.
            expected_state_history: Option<Vec<[u8; 32]>>,
        }

        #[doc = concat!("Fiat-Shamir transcript implementation using ", stringify!($name), ".")]
        #[derive(Clone)]
        pub struct $name {
            /// 256-bit running state.
            state: [u8; 32],
            /// Round counter for domain separation.
            n_rounds: u32,
            /// Test-only state for transcript comparison.
            #[cfg(test)]
            test_state: TestState,
        }

        impl Default for $name {
            fn default() -> Self {
                Self {
                    state: [0u8; 32],
                    n_rounds: 0,
                    #[cfg(test)]
                    test_state: TestState::default(),
                }
            }
        }

        impl std::fmt::Debug for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.debug_struct(stringify!($name))
                    .field("state", &hex::encode(self.state))
                    .field("n_rounds", &self.n_rounds)
                    .finish()
            }
        }

        impl $name {
            /// Creates a hasher initialized with current state and round counter.
            #[inline]
            fn hasher(&self) -> $hasher {
                let mut round_bytes = [0u8; 32];
                round_bytes[28..].copy_from_slice(&self.n_rounds.to_be_bytes());
                <$hasher as Digest>::new()
                    .chain_update(self.state)
                    .chain_update(round_bytes)
            }

            /// Loads arbitrary byte lengths using ceil(out/32) hash invocations.
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

            /// Loads exactly 32 bytes from the transcript.
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

        impl Transcript for $name {
            type Challenge = u128;

            fn new(label: &'static [u8]) -> Self {
                assert!(label.len() < 33, "label must be less than 33 bytes");

                let mut padded = [0u8; 32];
                padded[..label.len()].copy_from_slice(label);

                let hash: [u8; 32] = <$hasher as Digest>::new()
                    .chain_update(padded)
                    .finalize()
                    .into();

                Self {
                    state: hash,
                    n_rounds: 0,
                    #[cfg(test)]
                    test_state: TestState {
                        state_history: vec![hash],
                        expected_state_history: None,
                    },
                }
            }

            fn append_bytes(&mut self, bytes: &[u8]) {
                let hash: [u8; 32] = self.hasher().chain_update(bytes).finalize().into();
                self.update_state(hash);
            }

            fn challenge(&mut self) -> Self::Challenge {
                let mut buf = [0u8; 16];
                self.challenge_bytes(&mut buf);
                buf.reverse();
                u128::from_be_bytes(buf)
            }

            #[inline]
            fn state(&self) -> &[u8; 32] {
                &self.state
            }

            #[cfg(test)]
            fn compare_to(&mut self, other: &Self) {
                self.test_state.expected_state_history =
                    Some(other.test_state.state_history.clone());
            }
        }
    };
}

pub(crate) use impl_transcript;
