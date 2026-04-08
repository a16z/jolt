use super::transcript::Transcript;
use crate::field::JoltField;
use ark_bn254::Fr;
use ark_ff::PrimeField;
use ark_serialize::CanonicalSerialize;
use ark_std::Zero;
use light_poseidon::{Poseidon, PoseidonHasher};

/// Poseidon hash width: 3 field elements (state, n_rounds, data).
const POSEIDON_WIDTH: usize = 3;

/// Bytes per field element chunk (BN254 Fr = 32 bytes).
const BYTES_PER_CHUNK: usize = 32;

/// Fiat-Shamir transcript using Poseidon hash for BN254.
///
/// Uses width-3 Poseidon (3 field element inputs) with explicit domain separation
/// via `n_rounds` counter. Each hash: `hash(state, n_rounds, data)`.
///
/// Parameters: BN254 Fr, circom-compatible (light-poseidon).
#[derive(Default, Clone)]
pub struct PoseidonTranscript {
    /// 256-bit running state
    pub state: [u8; 32],
    /// We append an ordinal to each invocation of the hash
    pub n_rounds: u32,
    #[cfg(test)]
    /// A complete history of the transcript's `state`; used for testing.
    state_history: Vec<[u8; 32]>,
    #[cfg(test)]
    /// For a proof to be valid, the verifier's `state_history` should always match
    /// the prover's. In testing, the Jolt verifier may be provided the prover's
    /// `state_history` so that we can detect any deviations and the backtrace can
    /// tell us where it happened.
    expected_state_history: Option<Vec<[u8; 32]>>,
}

impl PoseidonTranscript {
    /// Create a new Poseidon hasher instance with BN254 circom-compatible parameters.
    fn hasher() -> Poseidon<Fr> {
        Poseidon::<Fr>::new_circom(POSEIDON_WIDTH).expect("Failed to initialize Poseidon for Fr")
    }

    /// Hash exactly 32 bytes using Poseidon and update state.
    /// Includes n_rounds for domain separation, matching Blake2b/Keccak semantics.
    ///
    /// # Panics
    /// Panics if `bytes.len() != 32`.
    fn hash_bytes32_and_update(&mut self, bytes: &[u8]) {
        assert_eq!(
            bytes.len(),
            BYTES_PER_CHUNK,
            "hash_bytes32_and_update requires exactly 32 bytes"
        );

        let mut poseidon = Self::hasher();

        // Convert state bytes to Fr (LE format, matching arkworks).
        let state_f = Fr::from_le_bytes_mod_order(&self.state);
        let round_f = Fr::from(self.n_rounds as u64);
        let input_f = Fr::from_le_bytes_mod_order(bytes);

        let output = poseidon
            .hash(&[state_f, round_f, input_f])
            .expect("Poseidon hash failed");

        let mut new_state = [0u8; 32];
        output.serialize_uncompressed(&mut new_state[..]).unwrap();

        // serialize_uncompressed gives LE bytes, keep as is (no reverse)
        self.update_state(new_state);
    }

    /// Loads arbitrary byte lengths using ceil(out/32) invocations of 32 byte randoms.
    /// Discards top bits when the size is less than 32 bytes.
    fn challenge_bytes(&mut self, out: &mut [u8]) {
        let mut remaining_len = out.len();
        let mut start = 0;
        while remaining_len > BYTES_PER_CHUNK {
            self.challenge_bytes32(&mut out[start..start + BYTES_PER_CHUNK]);
            start += BYTES_PER_CHUNK;
            remaining_len -= BYTES_PER_CHUNK;
        }
        // We load a full 32 byte random region
        let mut full_rand = [0u8; 32];
        self.challenge_bytes32(&mut full_rand);
        // Then only copy the first bytes of this random region to perfectly fill out
        out[start..start + remaining_len].copy_from_slice(&full_rand[..remaining_len]);
    }

    /// Loads exactly 32 bytes from the transcript by hashing the seed with the round constant
    fn challenge_bytes32(&mut self, out: &mut [u8]) {
        assert_eq!(BYTES_PER_CHUNK, out.len());
        let mut poseidon = Self::hasher();
        let state_f = Fr::from_le_bytes_mod_order(&self.state);
        let round_f = Fr::from(self.n_rounds as u64);
        let zero = Fr::zero();
        let output = poseidon
            .hash(&[state_f, round_f, zero])
            .expect("Poseidon hash failed");

        // serialize_uncompressed gives LE bytes, keep as is (no reverse)
        output.serialize_uncompressed(&mut out[..]).unwrap();
        self.update_state(out.try_into().unwrap());
    }

    fn update_state(&mut self, new_state: [u8; 32]) {
        self.state = new_state;
        self.n_rounds += 1;
        #[cfg(test)]
        {
            if let Some(expected_state_history) = &self.expected_state_history {
                assert!(
                    (self.n_rounds as usize) < expected_state_history.len(),
                    "Fiat-Shamir transcript mismatch: n_rounds {} exceeds expected history length {}",
                    self.n_rounds,
                    expected_state_history.len()
                );
                assert!(
                    new_state == expected_state_history[self.n_rounds as usize],
                    "Fiat-Shamir transcript mismatch at round {}",
                    self.n_rounds
                );
            }
            self.state_history.push(new_state);
        }
    }
}

impl Transcript for PoseidonTranscript {
    fn new(label: &'static [u8]) -> Self {
        // Hash in the label
        assert!(label.len() <= BYTES_PER_CHUNK);
        let mut poseidon = Self::hasher();

        // from_le_bytes_mod_order works with any length; trailing zeros in LE don't change value
        let label_f = Fr::from_le_bytes_mod_order(label);

        let zero = Fr::zero();
        let initial_state = poseidon
            .hash(&[label_f, zero, zero])
            .expect("Poseidon hash failed");

        let mut state = [0u8; 32];
        initial_state
            .serialize_uncompressed(&mut state[..])
            .unwrap();

        Self {
            state,
            n_rounds: 0,
            #[cfg(test)]
            state_history: vec![state],
            #[cfg(test)]
            expected_state_history: None,
        }
    }

    #[cfg(test)]
    /// Compare this transcript to `other` and panic if/when they deviate.
    /// Typically used to compare the verifier's transcript to the prover's.
    fn compare_to(&mut self, other: Self) {
        self.expected_state_history = Some(other.state_history);
    }

    // === Internal raw_append_* methods ===

    fn raw_append_label(&mut self, label: &'static [u8]) {
        // Labels must fit into one hash chunk
        assert!(label.len() <= BYTES_PER_CHUNK);
        let mut packed = [0u8; BYTES_PER_CHUNK];
        packed[..label.len()].copy_from_slice(label);
        self.hash_bytes32_and_update(&packed);
    }

    fn raw_append_bytes(&mut self, bytes: &[u8]) {
        // Poseidon has fixed arity (3 field elements), so we chunk the input.
        // First chunk: hash(state, n_rounds, chunk) - includes domain separation.
        // Subsequent chunks: hash(prev, 0, chunk) - chained without redundant n_rounds.
        let mut poseidon = Self::hasher();
        let state_f = Fr::from_le_bytes_mod_order(&self.state);
        let round_f = Fr::from(self.n_rounds as u64);
        let zero = Fr::zero();

        let mut chunks = bytes.chunks(BYTES_PER_CHUNK);

        // First hash: includes n_rounds for domain separation
        // from_le_bytes_mod_order handles any length; no padding needed
        let first_chunk_f = chunks
            .next()
            .map(Fr::from_le_bytes_mod_order)
            .unwrap_or(zero);
        let mut current = poseidon
            .hash(&[state_f, round_f, first_chunk_f])
            .expect("Poseidon hash failed");

        // Remaining chunks: chain without n_rounds
        for chunk in chunks {
            let chunk_f = Fr::from_le_bytes_mod_order(chunk);
            current = poseidon
                .hash(&[current, zero, chunk_f])
                .expect("Poseidon hash failed");
        }

        let mut new_state = [0u8; 32];
        current.serialize_uncompressed(&mut new_state[..]).unwrap();

        self.update_state(new_state);
    }

    fn raw_append_u64(&mut self, x: u64) {
        // Pack as native LE: from_le_bytes_mod_order(packed) = x directly
        let mut packed = [0u8; BYTES_PER_CHUNK];
        packed[..8].copy_from_slice(&x.to_le_bytes());
        self.hash_bytes32_and_update(&packed);
    }

    fn raw_append_scalar<JF: JoltField>(&mut self, scalar: &JF) {
        let mut buf = vec![];
        scalar.serialize_uncompressed(&mut buf).unwrap();
        // LE bytes of scalar → from_le_bytes_mod_order = scalar itself.
        // No byte reversal needed (Groth16 circuit, not EVM).
        self.raw_append_bytes(&buf);
    }

    // === Public API (overrides) ===

    /// Override: skip buf.reverse() from the trait default (EVM compat not needed for Groth16)
    fn append_serializable<T: CanonicalSerialize>(&mut self, label: &'static [u8], data: &T) {
        let mut buf = vec![];
        data.serialize_uncompressed(&mut buf).unwrap();
        self.raw_append_label_with_len(label, buf.len() as u64);
        // LE bytes directly, no byte reversal
        self.raw_append_bytes(&buf);
    }

    // === Challenge generation methods ===

    fn challenge_u128(&mut self) -> u128 {
        let mut buf = [0u8; 16];
        self.challenge_bytes(&mut buf);
        // LE bytes directly, no reversal
        u128::from_le_bytes(buf)
    }

    fn challenge_scalar<JF: JoltField>(&mut self) -> JF {
        // Full 32-byte hash output = full Fr challenge (no truncation).
        // challenge_bytes(32) → challenge_bytes32 → one hash invocation.
        // from_le_bytes_mod_order(serialize_le(Fr)) = Fr (identity).
        let mut buf = vec![0u8; 32];
        self.challenge_bytes(&mut buf);
        JF::from_bytes(&buf)
    }

    fn challenge_scalar_128_bits<JF: JoltField>(&mut self) -> JF {
        unimplemented!("128-bit challenges are unsupported for PoseidonTranscript");
    }

    fn challenge_vector<JF: JoltField>(&mut self, len: usize) -> Vec<JF> {
        (0..len)
            .map(|_i| self.challenge_scalar())
            .collect::<Vec<JF>>()
    }

    // Compute powers of scalar q : (1, q, q^2, ..., q^(len-1))
    fn challenge_scalar_powers<JF: JoltField>(&mut self, len: usize) -> Vec<JF> {
        let q: JF = self.challenge_scalar();
        let mut q_powers = vec![JF::one(); len];
        for i in 1..len {
            q_powers[i] = q_powers[i - 1] * q;
        }
        q_powers
    }

    fn challenge_scalar_optimized<JF: JoltField>(&mut self) -> JF::Challenge {
        // Full Fr challenge via challenge_scalar, then wrap in Challenge type.
        // Mont254BitChallenge<F> is a newtype of F → same memory layout → transmute is safe.
        let scalar: JF = self.challenge_scalar();
        unsafe { std::mem::transmute_copy::<JF, JF::Challenge>(&scalar) }
    }

    fn challenge_vector_optimized<JF: JoltField>(&mut self, len: usize) -> Vec<JF::Challenge> {
        (0..len)
            .map(|_| self.challenge_scalar_optimized::<JF>())
            .collect()
    }

    fn challenge_scalar_powers_optimized<JF: JoltField>(&mut self, len: usize) -> Vec<JF> {
        // This is still different from challenge_scalar_powers as inside the for loop
        // we use an optimised multiplication every time we compute the powers.
        let q: JF::Challenge = self.challenge_scalar_optimized::<JF>();
        let mut q_powers = vec![<JF as ark_std::One>::one(); len];
        for i in 1..len {
            q_powers[i] = q * q_powers[i - 1]; // this is optimised
        }
        q_powers
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;
    use std::collections::HashSet;

    type TestTranscript = PoseidonTranscript;

    // ============================================================================
    // Append Methods Tests
    // ============================================================================

    #[test]
    fn test_append_bytes_chunking() {
        // Test that chunking (>32 bytes) is handled correctly

        // Test 1: Single chunk (exactly 32 bytes)
        let mut t1 = TestTranscript::new(b"chunk_test");
        let data_32 = [0xABu8; 32];
        t1.append_bytes(b"data", &data_32);
        let state_32 = t1.state;

        // Test 2: Two chunks (33 bytes = 32 + 1)
        let mut t2 = TestTranscript::new(b"chunk_test");
        let data_33 = [0xABu8; 33];
        t2.append_bytes(b"data", &data_33);
        let state_33 = t2.state;

        // Adding one byte should produce different state (different chunk boundary)
        assert_ne!(
            state_32, state_33,
            "32-byte and 33-byte inputs should differ"
        );

        // Test 3: Determinism for multi-chunk input
        let mut t3 = TestTranscript::new(b"chunk_test");
        let data_100 = [0xCDu8; 100]; // 100 bytes = 4 chunks (32+32+32+4)
        t3.append_bytes(b"data", &data_100);

        let mut t4 = TestTranscript::new(b"chunk_test");
        t4.append_bytes(b"data", &data_100);
        assert_eq!(t3.state, t4.state, "Multi-chunk should be deterministic");

        // Test 4: Append order matters (chunks processed sequentially)
        let mut t5 = TestTranscript::new(b"chunk_test");
        t5.append_bytes(b"data", &data_100[..50]);
        t5.append_bytes(b"data", &data_100[50..]);

        // Splitting across append_bytes calls should produce different result
        assert_ne!(
            t3.state, t5.state,
            "Single append vs split append should differ"
        );

        // Test 5: Different data in last chunk
        let mut t6 = TestTranscript::new(b"chunk_test");
        let mut data_100_diff = data_100;
        data_100_diff[99] = 0xFF; // Change last byte
        t6.append_bytes(b"data", &data_100_diff);
        assert_ne!(t3.state, t6.state, "Different data should differ");
    }

    #[test]
    fn test_append_scalar_batch_vs_single() {
        // Test that batch and single appends produce different states
        // (batch uses label_with_len, single uses label repeatedly)
        use ark_std::UniformRand;
        let mut rng = ark_std::test_rng();

        let scalars: Vec<Fr> = (0..3).map(|_| Fr::rand(&mut rng)).collect();

        // Batch: append_scalars calls label_with_len(label, count) once
        let mut t_batch = TestTranscript::new(b"batch_test");
        t_batch.append_scalars(b"s", &scalars);

        // Sequential: append_scalar calls label(label) for each scalar
        let mut t_sequential = TestTranscript::new(b"batch_test");
        for scalar in &scalars {
            t_sequential.append_scalar(b"s", scalar);
        }

        assert_ne!(
            t_batch.state, t_sequential.state,
            "Batch and sequential appends should differ (different label handling)"
        );

        // Test edge cases: zero and max field element are handled correctly
        let mut t_zero = TestTranscript::new(b"edge_test");
        t_zero.append_scalar(b"s", &Fr::from(0u64));

        let mut t_max = TestTranscript::new(b"edge_test");
        let max_scalar = -Fr::from(1u64); // p-1 in Fr
        t_max.append_scalar(b"s", &max_scalar);

        assert_ne!(
            t_zero.state, t_max.state,
            "Zero and max field element should produce different states"
        );

        // Batch determinism
        let mut t_batch2 = TestTranscript::new(b"batch_test");
        t_batch2.append_scalars(b"s", &scalars);
        assert_eq!(
            t_batch.state, t_batch2.state,
            "Batch should be deterministic"
        );
    }

    #[test]
    fn test_append_point_batch_vs_single() {
        // Test infinity handling and batch vs single behavior
        use crate::curve::Bn254G1;
        use ark_bn254::G1Projective;
        use ark_std::UniformRand;

        let mut rng = ark_std::test_rng();
        let infinity = Bn254G1::default();

        // Test 1: Infinity should not panic and should update state
        let mut t1 = TestTranscript::new(b"infinity_test");
        t1.append_commitment(b"pt", &infinity);
        assert_eq!(t1.n_rounds, 2, "Label + point = 2 rounds");

        // Test 2: Infinity should be deterministic
        let mut t2 = TestTranscript::new(b"infinity_test");
        t2.append_commitment(b"pt", &infinity);
        assert_eq!(t1.state, t2.state, "Infinity should be deterministic");

        // Test 3: Infinity differs from non-infinity
        let mut t3 = TestTranscript::new(b"infinity_test");
        let non_infinity = Bn254G1(G1Projective::rand(&mut rng));
        t3.append_commitment(b"pt", &non_infinity);
        assert_ne!(t1.state, t3.state, "Infinity vs non-infinity should differ");

        // Test 4: Batch and sequential appends differ (different label handling)
        let points: Vec<Bn254G1> = (0..3)
            .map(|_| Bn254G1(G1Projective::rand(&mut rng)))
            .collect();

        let mut t_batch = TestTranscript::new(b"batch_test");
        t_batch.append_commitments(b"pts", &points);

        let mut t_sequential = TestTranscript::new(b"batch_test");
        for point in &points {
            t_sequential.append_commitment(b"pts", point);
        }

        assert_ne!(
            t_batch.state, t_sequential.state,
            "Batch and sequential appends should differ (different label handling)"
        );

        // Batch determinism
        let mut t_batch2 = TestTranscript::new(b"batch_test");
        t_batch2.append_commitments(b"pts", &points);
        assert_eq!(
            t_batch.state, t_batch2.state,
            "Batch should be deterministic"
        );
    }

    // ============================================================================
    // Challenge Methods Tests
    // ============================================================================

    #[test]
    fn test_challenge_u128() {
        let mut transcript1 = TestTranscript::new(b"u128_test");
        let mut transcript2 = TestTranscript::new(b"u128_test");

        let c1 = transcript1.challenge_u128();
        let c2 = transcript2.challenge_u128();

        assert_eq!(c1, c2, "Deterministic challenge_u128");
        assert_ne!(c1, 0, "Challenge should not be zero");
    }

    #[test]
    fn test_challenge_scalar_full_field() {
        // Poseidon returns full Fr challenges (no 128-bit truncation)
        let mut transcript = TestTranscript::new(b"test_full_field_scalar");
        let mut scalars = HashSet::new();

        for i in 0..1000 {
            let scalar: Fr = transcript.challenge_scalar();
            assert!(
                scalars.insert(scalar),
                "Duplicate scalar found at iteration {i}",
            );
        }
        // Verify we got distinct scalars (basic sanity check)
        assert_eq!(scalars.len(), 1000);
    }

    #[test]
    fn test_challenge_vector_operations() {
        // Test regular and optimized challenge vector generation

        // Test 1: Determinism
        let mut transcript1 = TestTranscript::new(b"vector_test");
        let mut transcript2 = TestTranscript::new(b"vector_test");

        let challenges1: Vec<Fr> = transcript1.challenge_vector(5);
        let challenges2: Vec<Fr> = transcript2.challenge_vector(5);

        assert_eq!(challenges1.len(), 5);
        assert_eq!(
            challenges1, challenges2,
            "challenge_vector should be deterministic"
        );

        // Test 2: Collision resistance (challenges are distinct)
        let unique_challenges: HashSet<Fr> = challenges1.iter().copied().collect();
        assert_eq!(
            unique_challenges.len(),
            challenges1.len(),
            "All challenges should be distinct"
        );

        // Test 3: Optimized version determinism
        let mut t3 = TestTranscript::new(b"opt_vector_test");
        let mut t4 = TestTranscript::new(b"opt_vector_test");

        let challenges3: Vec<<Fr as JoltField>::Challenge> = t3.challenge_vector_optimized::<Fr>(5);
        let challenges4: Vec<<Fr as JoltField>::Challenge> = t4.challenge_vector_optimized::<Fr>(5);

        assert_eq!(challenges3.len(), 5);
        assert_eq!(
            challenges3, challenges4,
            "challenge_vector_optimized should be deterministic"
        );
    }

    #[test]
    fn test_challenge_scalar_powers_operations() {
        // Test regular and optimized challenge_scalar_powers

        // Test 1: Regular powers verification
        let mut transcript = TestTranscript::new(b"powers_test");
        let powers: Vec<Fr> = transcript.challenge_scalar_powers(5);

        assert_eq!(powers.len(), 5);
        assert_eq!(powers[0], Fr::from(1u64), "First power should be 1");

        // Verify powers[i] = powers[i-1] * q (powers[1] is the base)
        for i in 2..powers.len() {
            assert_eq!(
                powers[i],
                powers[i - 1] * powers[1],
                "Power at index {i} should equal powers[{i}-1] * base"
            );
        }

        // Test 2: Optimized powers should also satisfy power property
        let mut t2 = TestTranscript::new(b"opt_powers_test");
        let opt_powers: Vec<Fr> = t2.challenge_scalar_powers_optimized(5);

        assert_eq!(opt_powers.len(), 5);
        assert_eq!(opt_powers[0], Fr::from(1u64), "First power should be 1");

        // Verify power property holds for optimized version
        for i in 2..opt_powers.len() {
            assert_eq!(
                opt_powers[i],
                opt_powers[i - 1] * opt_powers[1],
                "Optimized power at index {i} should equal opt_powers[{i}-1] * base"
            );
        }

        // Test 3: Equivalence between regular and optimized (same transcript state)
        let mut t3 = TestTranscript::new(b"equiv_test");
        let mut t4 = TestTranscript::new(b"equiv_test");

        let regular_powers: Vec<Fr> = t3.challenge_scalar_powers(5);
        let optimized_powers: Vec<Fr> = t4.challenge_scalar_powers_optimized(5);

        assert_eq!(
            regular_powers, optimized_powers,
            "Regular and optimized powers should be equivalent"
        );
    }

    #[test]
    fn test_challenge_optimized_arithmetic_equivalence() {
        // Verify that challenge_scalar_optimized produces a Truncate128 type
        // that is arithmetically equivalent to a full Fr element via transmute
        use ark_std::UniformRand;
        let mut rng = ark_std::test_rng();
        let mut transcript1 = TestTranscript::new(b"test_trivial_challenge");

        let challenge = transcript1.challenge_scalar_optimized::<Fr>();
        // The same challenge as a full fat Fr element
        let challenge_regular: Fr = challenge.into();

        let field_elements: Vec<Fr> = (0..10).map(|_| Fr::rand(&mut rng)).collect();

        for (i, &field_elem) in field_elements.iter().enumerate() {
            let result_challenge = field_elem * challenge;
            let result_regular = field_elem * challenge_regular;

            assert_eq!(
                result_challenge, result_regular,
                "Multiplication mismatch at index {i}"
            );
        }

        let field_elem = Fr::rand(&mut rng);
        #[allow(clippy::op_ref)]
        let result_ref = field_elem * &challenge;
        let result_regular = field_elem * challenge;
        assert_eq!(
            result_ref, result_regular,
            "Reference multiplication mismatch"
        );
    }

    // ============================================================================
    // Property Tests
    // ============================================================================

    #[test]
    fn test_deterministic_challenges() {
        // Same inputs should produce same challenges across multiple test cases
        let test_cases = vec![
            (b"test1" as &[u8], 123u64, b"data1" as &[u8]),
            (b"test2" as &[u8], 456u64, b"data2" as &[u8]),
            (b"test3" as &[u8], 789u64, b"longer test data here" as &[u8]),
        ];

        for (label, num, data) in test_cases {
            let mut transcript1 = TestTranscript::new(label);
            transcript1.append_u64(b"num", num);
            transcript1.append_bytes(b"data", data);
            let challenge1: Fr = transcript1.challenge_scalar();

            let mut transcript2 = TestTranscript::new(label);
            transcript2.append_u64(b"num", num);
            transcript2.append_bytes(b"data", data);
            let challenge2: Fr = transcript2.challenge_scalar();

            assert_eq!(
                challenge1,
                challenge2,
                "Determinism failed for label={:?}",
                std::str::from_utf8(label).unwrap()
            );
        }
    }

    #[test]
    fn test_append_order_sensitivity() {
        // Verify that append order matters (critical for Fiat-Shamir security)
        let scalar_a = Fr::from(111u64);
        let scalar_b = Fr::from(222u64);

        // Append A then B
        let mut t1 = TestTranscript::new(b"order_test");
        t1.append_scalar(b"s", &scalar_a);
        t1.append_scalar(b"s", &scalar_b);

        // Append B then A
        let mut t2 = TestTranscript::new(b"order_test");
        t2.append_scalar(b"s", &scalar_b);
        t2.append_scalar(b"s", &scalar_a);

        assert_ne!(
            t1.state, t2.state,
            "Append order should affect transcript state"
        );

        // Also test with different append types
        let mut t3 = TestTranscript::new(b"mixed_order_test");
        t3.append_u64(b"num", 123);
        t3.append_bytes(b"data", b"test");

        let mut t4 = TestTranscript::new(b"mixed_order_test");
        t4.append_bytes(b"data", b"test");
        t4.append_u64(b"num", 123);

        assert_ne!(
            t3.state, t4.state,
            "Append order should matter across different types"
        );
    }

    #[test]
    fn test_label_sensitivity() {
        // Verify that different labels produce different states
        let scalar = Fr::from(12345u64);

        let mut t1 = TestTranscript::new(b"label_test");
        t1.append_scalar(b"label1", &scalar);

        let mut t2 = TestTranscript::new(b"label_test");
        t2.append_scalar(b"label2", &scalar);

        assert_ne!(
            t1.state, t2.state,
            "Different labels should produce different states"
        );

        // Test initialization label sensitivity
        let t3 = TestTranscript::new(b"init_label_1");
        let t4 = TestTranscript::new(b"init_label_2");

        assert_ne!(
            t3.state, t4.state,
            "Different initialization labels should produce different states"
        );
    }
}
