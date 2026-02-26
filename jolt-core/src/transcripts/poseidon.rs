use super::transcript::Transcript;
use crate::field::JoltField;
use ark_bn254::Fr;
use ark_ec::{AffineRepr, CurveGroup};
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

    fn raw_append_point<G: CurveGroup>(&mut self, point: &G) {
        // Point at infinity: hash 64 zero bytes
        if point.is_zero() {
            self.raw_append_bytes(&[0_u8; 2 * BYTES_PER_CHUNK]);
            return;
        }

        // Extract affine coordinates and serialize as LE bytes (no reversal for Groth16)
        let aff = point.into_affine();
        let mut combined = [0u8; 2 * BYTES_PER_CHUNK];
        aff.x()
            .unwrap()
            .serialize_compressed(&mut combined[..BYTES_PER_CHUNK])
            .unwrap();
        aff.y()
            .unwrap()
            .serialize_compressed(&mut combined[BYTES_PER_CHUNK..])
            .unwrap();
        self.raw_append_bytes(&combined);
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
        // Debug print enabled by `debug-expected-output` feature for comparing
        // Rust verifier challenges against transpiled circuit implementations.
        #[cfg(feature = "debug-expected-output")]
        eprintln!(">>> challenge_scalar called <<<");
        self.challenge_scalar_128_bits()
    }

    fn challenge_scalar_128_bits<JF: JoltField>(&mut self) -> JF {
        // Full 32-byte hash output = full Fr challenge (no truncation).
        // challenge_bytes(32) → challenge_bytes32 → one hash invocation.
        // from_le_bytes_mod_order(serialize_le(Fr)) = Fr (identity).
        let mut buf = vec![0u8; 32];
        self.challenge_bytes(&mut buf);
        JF::from_bytes(&buf)
    }

    fn challenge_vector<JF: JoltField>(&mut self, len: usize) -> Vec<JF> {
        // Debug print enabled by `debug-expected-output` feature for comparing
        // Rust verifier challenges against transpiled circuit implementations.
        #[cfg(feature = "debug-expected-output")]
        eprintln!(">>> challenge_vector called with len={} <<<", len);
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
        // Full Fr challenge via challenge_scalar_128_bits, then wrap in Challenge type.
        // Mont254BitChallenge<F> is a newtype of F → same memory layout → transmute is safe.
        let scalar: JF = self.challenge_scalar_128_bits();
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

    fn debug_state(&self, label: &str) {
        #[cfg(feature = "debug-expected-output")]
        {
            let state_f = Fr::from_le_bytes_mod_order(&self.state);
            eprintln!("TRANSCRIPT DEBUG [{}]: n_rounds={}", label, self.n_rounds);
            eprintln!("  state (LE): {:02x?}", &self.state);

            // Compute what the next challenge would be (hash output)
            let mut poseidon = Self::hasher();
            let round_f = Fr::from(self.n_rounds as u64);
            let zero = Fr::zero();
            let hash_output = poseidon
                .hash(&[state_f, round_f, zero])
                .expect("Poseidon hash failed");
            let mut hash_bytes = [0u8; 32];
            hash_output
                .serialize_uncompressed(&mut hash_bytes[..])
                .unwrap();
            eprintln!("  next_challenge (LE): {:02x?}", &hash_bytes);
        }
        #[cfg(not(feature = "debug-expected-output"))]
        let _ = label;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;
    use std::collections::HashSet;

    type TestTranscript = PoseidonTranscript;

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
    fn test_challenge_u128() {
        let mut transcript1 = TestTranscript::new(b"u128_test");
        let mut transcript2 = TestTranscript::new(b"u128_test");

        let c1 = transcript1.challenge_u128();
        let c2 = transcript2.challenge_u128();

        assert_eq!(c1, c2, "Deterministic challenge_u128");
        assert_ne!(c1, 0, "Challenge should not be zero");
    }

    #[test]
    fn test_challenge_special_trivial() {
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

    #[test]
    fn test_deterministic_challenges() {
        // Same inputs should produce same challenges
        let mut transcript1 = TestTranscript::new(b"deterministic");
        transcript1.append_u64(b"num", 123);
        transcript1.append_bytes(b"data", b"test data");
        let challenge1: Fr = transcript1.challenge_scalar();

        let mut transcript2 = TestTranscript::new(b"deterministic");
        transcript2.append_u64(b"num", 123);
        transcript2.append_bytes(b"data", b"test data");
        let challenge2: Fr = transcript2.challenge_scalar();

        assert_eq!(challenge1, challenge2);
    }

    #[test]
    fn test_append_bytes_multi_chunk() {
        // Test that >32 byte input is chunked correctly
        let mut transcript1 = TestTranscript::new(b"multi_chunk_test");
        let mut transcript2 = TestTranscript::new(b"multi_chunk_test");

        // 100 bytes = 4 chunks (32 + 32 + 32 + 4)
        let data = [0xABu8; 100];
        transcript1.append_bytes(b"data", &data);
        transcript2.append_bytes(b"data", &data);

        assert_eq!(transcript1.state, transcript2.state);

        // Different data should produce different state
        let mut transcript3 = TestTranscript::new(b"multi_chunk_test");
        let other_data = [0xCDu8; 100];
        transcript3.append_bytes(b"data", &other_data);
        assert_ne!(transcript1.state, transcript3.state);
    }

    #[test]
    fn test_append_scalar() {
        let mut transcript1 = TestTranscript::new(b"scalar_test");
        let mut transcript2 = TestTranscript::new(b"scalar_test");

        let scalar = Fr::from(12345u64);
        transcript1.append_scalar(b"s", &scalar);
        transcript2.append_scalar(b"s", &scalar);

        assert_eq!(transcript1.state, transcript2.state);

        // Different scalar should produce different state
        let mut transcript3 = TestTranscript::new(b"scalar_test");
        transcript3.append_scalar(b"s", &Fr::from(99999u64));
        assert_ne!(transcript1.state, transcript3.state);
    }

    #[test]
    fn test_append_point() {
        use ark_bn254::G1Projective;
        use ark_std::UniformRand;

        let mut rng = ark_std::test_rng();
        let point1 = G1Projective::rand(&mut rng);
        let point2 = G1Projective::rand(&mut rng);

        let mut transcript1 = TestTranscript::new(b"point_test");
        let mut transcript2 = TestTranscript::new(b"point_test");

        transcript1.append_point(b"pt", &point1);
        transcript2.append_point(b"pt", &point1);

        assert_eq!(transcript1.state, transcript2.state);

        // Different point should produce different state
        let mut transcript3 = TestTranscript::new(b"point_test");
        transcript3.append_point(b"pt", &point2);
        assert_ne!(transcript1.state, transcript3.state);
    }

    #[test]
    fn test_append_point_infinity() {
        use ark_bn254::G1Projective;
        use ark_std::UniformRand;

        let mut rng = ark_std::test_rng();
        let infinity = G1Projective::default(); // point at infinity
        let non_infinity = G1Projective::rand(&mut rng);

        let mut transcript1 = TestTranscript::new(b"infinity_test");
        transcript1.append_point(b"pt", &infinity);

        // Should not panic and should update state (label + point = 2 rounds)
        assert_eq!(transcript1.n_rounds, 2);

        // Infinity should produce different state than non-infinity point
        let mut transcript2 = TestTranscript::new(b"infinity_test");
        transcript2.append_point(b"pt", &non_infinity);
        assert_ne!(transcript1.state, transcript2.state);
    }

    #[test]
    fn test_append_scalars() {
        let mut transcript1 = TestTranscript::new(b"scalars_test");
        let mut transcript2 = TestTranscript::new(b"scalars_test");

        let scalars: Vec<Fr> = vec![Fr::from(1u64), Fr::from(2u64), Fr::from(3u64)];
        transcript1.append_scalars(b"scalars", &scalars);
        transcript2.append_scalars(b"scalars", &scalars);

        assert_eq!(transcript1.state, transcript2.state);

        // Different scalars should produce different state
        let mut transcript3 = TestTranscript::new(b"scalars_test");
        let other_scalars: Vec<Fr> = vec![Fr::from(4u64), Fr::from(5u64), Fr::from(6u64)];
        transcript3.append_scalars(b"scalars", &other_scalars);
        assert_ne!(transcript1.state, transcript3.state);
    }

    #[test]
    fn test_append_points() {
        use ark_bn254::G1Projective;
        use ark_std::UniformRand;

        let mut rng = ark_std::test_rng();
        let points: Vec<G1Projective> = (0..3).map(|_| G1Projective::rand(&mut rng)).collect();

        let mut transcript1 = TestTranscript::new(b"points_test");
        let mut transcript2 = TestTranscript::new(b"points_test");

        transcript1.append_points(b"points", &points);
        transcript2.append_points(b"points", &points);

        assert_eq!(transcript1.state, transcript2.state);
    }

    #[test]
    fn test_append_serializable() {
        let mut transcript1 = TestTranscript::new(b"serializable_test");
        let mut transcript2 = TestTranscript::new(b"serializable_test");

        let scalar = Fr::from(12345u64);
        transcript1.append_serializable(b"ser", &scalar);
        transcript2.append_serializable(b"ser", &scalar);

        assert_eq!(transcript1.state, transcript2.state);
    }

    #[test]
    fn test_challenge_vector() {
        let mut transcript1 = TestTranscript::new(b"vector_test");
        let mut transcript2 = TestTranscript::new(b"vector_test");

        let challenges1: Vec<Fr> = transcript1.challenge_vector(5);
        let challenges2: Vec<Fr> = transcript2.challenge_vector(5);

        assert_eq!(challenges1.len(), 5);
        assert_eq!(challenges1, challenges2);

        // All challenges should be distinct
        for i in 0..challenges1.len() {
            for j in (i + 1)..challenges1.len() {
                assert_ne!(challenges1[i], challenges1[j]);
            }
        }
    }

    #[test]
    fn test_challenge_scalar_powers() {
        let mut transcript = TestTranscript::new(b"powers_test");

        let powers: Vec<Fr> = transcript.challenge_scalar_powers(5);

        assert_eq!(powers.len(), 5);
        assert_eq!(powers[0], Fr::from(1u64)); // First power is always 1

        // Verify powers[i] = powers[i-1] * q
        for i in 2..powers.len() {
            assert_eq!(powers[i], powers[i - 1] * powers[1]);
        }
    }

    #[test]
    fn test_challenge_vector_optimized() {
        let mut transcript1 = TestTranscript::new(b"opt_vector_test");
        let mut transcript2 = TestTranscript::new(b"opt_vector_test");

        let challenges1: Vec<<Fr as JoltField>::Challenge> =
            transcript1.challenge_vector_optimized::<Fr>(5);
        let challenges2: Vec<<Fr as JoltField>::Challenge> =
            transcript2.challenge_vector_optimized::<Fr>(5);

        assert_eq!(challenges1.len(), 5);
        assert_eq!(challenges1, challenges2);
    }

    #[test]
    fn test_challenge_scalar_powers_optimized() {
        let mut transcript = TestTranscript::new(b"opt_powers_test");

        let powers: Vec<Fr> = transcript.challenge_scalar_powers_optimized(5);

        assert_eq!(powers.len(), 5);
        assert_eq!(powers[0], Fr::from(1u64));
    }
}
