//! Poseidon-based Fiat-Shamir transcript for SNARK-friendly verification.
//!
//! Uses width-3 Poseidon (3 field element inputs) over BN254 Fr with
//! circom-compatible parameters via [`light_poseidon`]. Each hash operation:
//! `state = poseidon(state, n_rounds, data)`.
//!
//! # Why Poseidon?
//!
//! Poseidon is ~600× cheaper in-circuit than Keccak (~250 constraints vs
//! ~150,000). When the Jolt verifier runs inside a Groth16/gnark circuit,
//! all Fiat-Shamir challenges must be recomputed — using a SNARK-friendly
//! hash makes this feasible.
//!
//! # Parameters
//!
//! - **Width**: 3 field elements (state, round counter, data)
//! - **Curve**: BN254 scalar field (Fr)
//! - **Constants**: circom-compatible (`light_poseidon::new_circom`)
//! - **Rounds**: 8 full + 56 partial, x^5 S-box
//!
//! # Domain separation
//!
//! Each `append_bytes` call includes an `n_rounds` counter in the hash input
//! for domain separation. Multi-chunk appends chain: first chunk includes
//! `n_rounds`, remaining chunks chain as `poseidon(prev, 0, chunk_i)`.

use ark_bn254::Fr;
use ark_ff::{PrimeField, Zero};
use ark_serialize::CanonicalSerialize;
use light_poseidon::{Poseidon, PoseidonHasher};

use crate::transcript::Transcript;

/// Poseidon hash width: 3 field elements.
const WIDTH: usize = 3;

/// Bytes per BN254 Fr field element.
const BYTES_PER_CHUNK: usize = 32;

/// Fiat-Shamir transcript using Poseidon hash over BN254.
///
/// Generic over the challenge type `C`. Use `From<u128>` challenge types
/// like `MontU128Challenge<Fr>` for optimized field operations.
#[derive(Clone)]
pub struct PoseidonTranscript<C: Copy + Default + From<u128> = u128> {
    /// 256-bit running state (canonical LE serialization of Fr).
    state: [u8; 32],
    /// Round counter for domain separation.
    n_rounds: u32,
    /// Test-only state history for transcript comparison.
    #[cfg(test)]
    state_history: Vec<[u8; 32]>,
    #[cfg(test)]
    expected_state_history: Option<Vec<[u8; 32]>>,
    _challenge: std::marker::PhantomData<C>,
}

impl<C: Copy + Default + From<u128>> Default for PoseidonTranscript<C> {
    fn default() -> Self {
        Self {
            state: [0u8; 32],
            n_rounds: 0,
            #[cfg(test)]
            state_history: Vec::new(),
            #[cfg(test)]
            expected_state_history: None,
            _challenge: std::marker::PhantomData,
        }
    }
}

impl<C: Copy + Default + From<u128>> std::fmt::Debug for PoseidonTranscript<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PoseidonTranscript")
            .field("state", &hex::encode(self.state))
            .field("n_rounds", &self.n_rounds)
            .finish_non_exhaustive()
    }
}

impl<C: Copy + Default + From<u128>> PoseidonTranscript<C> {
    fn hasher() -> Poseidon<Fr> {
        Poseidon::<Fr>::new_circom(WIDTH).expect("failed to initialize Poseidon")
    }

    /// Squeeze exactly 32 challenge bytes: `poseidon(state, n_rounds, 0)`.
    fn challenge_bytes32(&mut self, out: &mut [u8; 32]) {
        let mut poseidon = Self::hasher();
        let state_f = Fr::from_le_bytes_mod_order(&self.state);
        let round_f = Fr::from(u64::from(self.n_rounds));
        let zero = Fr::zero();

        let output = poseidon
            .hash(&[state_f, round_f, zero])
            .expect("Poseidon hash failed");

        output
            .serialize_uncompressed(&mut out[..])
            .expect("Fr serialization failed");

        self.update_state(*out);
    }

    /// Fill `out` with challenge bytes using ceil(len / 32) hash invocations.
    fn challenge_bytes(&mut self, out: &mut [u8]) {
        let mut remaining = out.len();
        let mut offset = 0;

        while remaining > BYTES_PER_CHUNK {
            let mut chunk = [0u8; 32];
            self.challenge_bytes32(&mut chunk);
            out[offset..offset + BYTES_PER_CHUNK].copy_from_slice(&chunk);
            offset += BYTES_PER_CHUNK;
            remaining -= BYTES_PER_CHUNK;
        }

        let mut final_chunk = [0u8; 32];
        self.challenge_bytes32(&mut final_chunk);
        out[offset..offset + remaining].copy_from_slice(&final_chunk[..remaining]);
    }

    fn update_state(&mut self, new_state: [u8; 32]) {
        self.state = new_state;
        self.n_rounds += 1;

        #[cfg(test)]
        {
            if let Some(ref expected) = self.expected_state_history {
                assert!(
                    (self.n_rounds as usize) < expected.len(),
                    "Fiat-Shamir transcript: n_rounds {} exceeds expected history length {}",
                    self.n_rounds,
                    expected.len()
                );
                assert_eq!(
                    new_state, expected[self.n_rounds as usize],
                    "Fiat-Shamir transcript mismatch at round {}",
                    self.n_rounds
                );
            }
            self.state_history.push(new_state);
        }
    }
}

impl<C: Copy + Default + From<u128> + Send + Sync + 'static> Transcript for PoseidonTranscript<C> {
    type Challenge = C;

    fn new(label: &'static [u8]) -> Self {
        assert!(label.len() < 33, "label must be less than 33 bytes");

        let mut poseidon = Self::hasher();
        let label_f = Fr::from_le_bytes_mod_order(label);
        let zero = Fr::zero();

        let initial = poseidon
            .hash(&[label_f, zero, zero])
            .expect("Poseidon hash failed");

        let mut state = [0u8; 32];
        initial
            .serialize_uncompressed(&mut state[..])
            .expect("Fr serialization failed");

        Self {
            state,
            n_rounds: 0,
            #[cfg(test)]
            state_history: vec![state],
            #[cfg(test)]
            expected_state_history: None,
            _challenge: std::marker::PhantomData,
        }
    }

    fn append_bytes(&mut self, bytes: &[u8]) {
        let mut poseidon = Self::hasher();
        let state_f = Fr::from_le_bytes_mod_order(&self.state);
        let round_f = Fr::from(u64::from(self.n_rounds));
        let zero = Fr::zero();

        let mut chunks = bytes.chunks(BYTES_PER_CHUNK);

        let first_f = chunks
            .next()
            .map_or(zero, Fr::from_le_bytes_mod_order);

        let mut current = poseidon
            .hash(&[state_f, round_f, first_f])
            .expect("Poseidon hash failed");

        for chunk in chunks {
            let chunk_f = Fr::from_le_bytes_mod_order(chunk);
            current = poseidon
                .hash(&[current, zero, chunk_f])
                .expect("Poseidon hash failed");
        }

        let mut new_state = [0u8; 32];
        current
            .serialize_uncompressed(&mut new_state[..])
            .expect("Fr serialization failed");

        self.update_state(new_state);
    }

    fn challenge(&mut self) -> C {
        let mut buf = [0u8; 16];
        self.challenge_bytes(&mut buf);
        C::from(u128::from_le_bytes(buf))
    }

    #[inline]
    fn state(&self) -> &[u8; 32] {
        &self.state
    }

    #[cfg(test)]
    fn compare_to(&mut self, other: &Self) {
        self.expected_state_history = Some(other.state_history.clone());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_initializes_from_label() {
        let t1 = PoseidonTranscript::new(b"protocol_a");
        let t2 = PoseidonTranscript::new(b"protocol_b");

        // Different labels → different initial states
        assert_ne!(t1.state, t2.state);
        assert_eq!(t1.n_rounds, 0);
    }

    #[test]
    fn same_label_same_state() {
        let t1 = PoseidonTranscript::new(b"same");
        let t2 = PoseidonTranscript::new(b"same");

        assert_eq!(t1.state, t2.state);
    }

    #[test]
    fn append_changes_state() {
        let mut t = PoseidonTranscript::new(b"test");
        let before = t.state;

        t.append_bytes(b"hello");
        assert_ne!(t.state, before);
        assert_eq!(t.n_rounds, 1);
    }

    #[test]
    fn append_order_matters() {
        let mut t1 = PoseidonTranscript::new(b"test");
        let mut t2 = PoseidonTranscript::new(b"test");

        t1.append_bytes(b"a");
        t1.append_bytes(b"b");

        t2.append_bytes(b"b");
        t2.append_bytes(b"a");

        assert_ne!(t1.state, t2.state);
    }

    #[test]
    fn challenge_advances_state() {
        let mut t = PoseidonTranscript::new(b"test");
        t.append_bytes(b"data");
        let before = t.state;

        let _ = t.challenge();
        assert_ne!(t.state, before);
    }

    #[test]
    fn deterministic_challenges() {
        let mut t1 = PoseidonTranscript::new(b"test");
        let mut t2 = PoseidonTranscript::new(b"test");

        t1.append_bytes(b"same_data");
        t2.append_bytes(b"same_data");

        assert_eq!(t1.challenge(), t2.challenge());
    }

    #[test]
    fn multi_chunk_append() {
        let mut t = PoseidonTranscript::new(b"test");

        // 64 bytes = 2 chunks of 32 bytes
        let data = [0xABu8; 64];
        t.append_bytes(&data);

        // Should still be a single n_rounds increment (chained internally)
        assert_eq!(t.n_rounds, 1);
    }

    #[test]
    fn challenge_vector_produces_distinct() {
        let mut t = PoseidonTranscript::new(b"test");
        t.append_bytes(b"seed");

        let challenges = t.challenge_vector(5);
        // All challenges should be distinct
        for i in 0..5 {
            for j in (i + 1)..5 {
                assert_ne!(challenges[i], challenges[j]);
            }
        }
    }

    #[test]
    fn clone_independence() {
        let mut t = PoseidonTranscript::new(b"test");
        t.append_bytes(b"shared");

        let mut fork = t.clone();
        t.append_bytes(b"branch_a");
        fork.append_bytes(b"branch_b");

        assert_ne!(t.state, fork.state);
    }

    #[test]
    fn hash_zeros_produces_known_output() {
        // Validate that our Poseidon parameters produce a non-trivial hash
        // for the simplest input. This catches parameter initialization bugs.
        let mut poseidon = PoseidonTranscript::hasher();
        let result = poseidon
            .hash(&[Fr::zero(), Fr::zero(), Fr::zero()])
            .expect("hash failed");
        assert_ne!(result, Fr::zero(), "hash(0,0,0) should not be zero");
    }

    #[test]
    fn transcript_comparison() {
        let mut prover = PoseidonTranscript::new(b"test");
        prover.append_bytes(b"data");
        let _ = prover.challenge();

        let mut verifier = PoseidonTranscript::new(b"test");
        verifier.compare_to(&prover);
        verifier.append_bytes(b"data");
        let _ = verifier.challenge();
        // No panic means states matched
    }
}
