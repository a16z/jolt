use super::poseidon_fq_params::{
    hex_to_bytes, FQ_ALPHA, FQ_FULL_ROUNDS, FQ_MDS, FQ_PARTIAL_ROUNDS, FQ_ROUND_CONSTANTS, FQ_WIDTH,
};
use super::transcript::Transcript;
use crate::field::JoltField;
use ark_bn254::{Fq, Fr};
use ark_ec::{AffineRepr, CurveGroup};
use ark_ff::PrimeField;
use ark_serialize::CanonicalSerialize;
use light_poseidon::{Poseidon, PoseidonHasher, PoseidonParameters};
use std::borrow::Borrow;
use std::marker::PhantomData;

/// Poseidon hash width: 3 field elements (state, n_rounds, data).
const POSEIDON_WIDTH: usize = 3;

/// Trait for providing Poseidon parameters for a specific field.
///
/// This enables type-level configuration: different fields get different
/// Poseidon parameters (MDS matrix, round constants) selected at compile time.
pub trait PoseidonParams<F: PrimeField>: Clone + Send + Sync + 'static {
    fn poseidon() -> Poseidon<F>;
}

/// Poseidon parameters for BN254 Fr (scalar field).
/// Uses the well-audited circom-compatible parameters from light-poseidon.
#[derive(Clone, Copy)]
pub struct FrParams;

impl PoseidonParams<Fr> for FrParams {
    fn poseidon() -> Poseidon<Fr> {
        Poseidon::<Fr>::new_circom(POSEIDON_WIDTH).expect("Failed to initialize Poseidon for Fr")
    }
}

/// Poseidon parameters for BN254 Fq (base field).
/// Uses parameters generated with poseidon-paramgen v0.4.0 (audited by NCC Group).
#[derive(Clone, Copy)]
pub struct FqParams;

impl PoseidonParams<Fq> for FqParams {
    fn poseidon() -> Poseidon<Fq> {
        // Parse MDS matrix from hex strings
        let mut mds: Vec<Vec<Fq>> = Vec::with_capacity(FQ_WIDTH);
        for row in FQ_MDS.iter() {
            let mut mds_row: Vec<Fq> = Vec::with_capacity(FQ_WIDTH);
            for elem_str in row.iter() {
                let elem = Fq::from_be_bytes_mod_order(&hex_to_bytes(elem_str));
                mds_row.push(elem);
            }
            mds.push(mds_row);
        }

        // Parse round constants from hex strings
        let mut ark: Vec<Fq> = Vec::with_capacity(FQ_ROUND_CONSTANTS.len());
        for c_str in FQ_ROUND_CONSTANTS.iter() {
            let c = Fq::from_be_bytes_mod_order(&hex_to_bytes(c_str));
            ark.push(c);
        }

        let params =
            PoseidonParameters::new(ark, mds, FQ_FULL_ROUNDS, FQ_PARTIAL_ROUNDS, FQ_WIDTH, FQ_ALPHA);
        Poseidon::new(params)
    }
}

/// Type alias for the original Fr-based transcript (backwards compatible)
pub type PoseidonTranscriptFr = PoseidonTranscript<Fr, FrParams>;

/// Type alias for Fq-based transcript (for SNARK composition / Grumpkin Fr)
pub type PoseidonTranscriptFq = PoseidonTranscript<Fq, FqParams>;

/// Represents the current state of the protocol's Fiat-Shamir transcript using Poseidon.
///
/// This implementation uses Poseidon with width 3 (accepting 3 field element inputs)
/// to enable explicit domain separation through the n_rounds counter. Each hash operation
/// includes the current round number as input: hash(state, n_rounds, data). This matches
/// the security semantics of Blake2b and Keccak transcripts, which also bind n_rounds
/// into every hash invocation.
///
/// The type parameter `F` specifies the field, and `P` provides the Poseidon parameters.
/// For BN254 Fr (scalar field), use `PoseidonTranscript<Fr, FrParams>`.
/// For BN254 Fq (base field), use `PoseidonTranscript<Fq, FqParams>`.
#[derive(Clone)]
pub struct PoseidonTranscript<F: PrimeField, P: PoseidonParams<F>> {
    /// 256-bit running state
    pub state: [u8; 32],
    /// We append an ordinal to each invocation of the hash
    n_rounds: u32,
    #[cfg(test)]
    /// A complete history of the transcript's `state`; used for testing.
    state_history: Vec<[u8; 32]>,
    #[cfg(test)]
    /// For a proof to be valid, the verifier's `state_history` should always match
    /// the prover's. In testing, the Jolt verifier may be provided the prover's
    /// `state_history` so that we can detect any deviations and the backtrace can
    /// tell us where it happened.
    expected_state_history: Option<Vec<[u8; 32]>>,
    _marker: PhantomData<(F, P)>,
}

impl<F: PrimeField, P: PoseidonParams<F>> Default for PoseidonTranscript<F, P> {
    fn default() -> Self {
        Self {
            state: [0u8; 32],
            n_rounds: 0,
            #[cfg(test)]
            state_history: Vec::new(),
            #[cfg(test)]
            expected_state_history: None,
            _marker: PhantomData,
        }
    }
}

impl<F: PrimeField, P: PoseidonParams<F>> PoseidonTranscript<F, P> {
    /// Create a new Poseidon hasher instance.
    fn hasher() -> Poseidon<F> {
        P::poseidon()
    }

    /// Hash exactly 32 bytes using Poseidon and update state.
    /// Includes n_rounds for domain separation, matching Blake2b/Keccak semantics.
    ///
    /// # Panics
    /// Panics if `bytes.len() != 32`.
    fn hash_bytes32_and_update(&mut self, bytes: &[u8]) {
        assert_eq!(
            bytes.len(),
            32,
            "hash_bytes32_and_update requires exactly 32 bytes"
        );

        let mut poseidon = Self::hasher();

        // Convert state bytes to F (LE format, matching arkworks).
        // Note: from_le_bytes_mod_order reduces inputs mod p, which is standard
        // behavior but means values >= p map to values < p.
        let state_f = F::from_le_bytes_mod_order(&self.state);
        let round_f = F::from(self.n_rounds as u64);

        let input_f = F::from_le_bytes_mod_order(bytes);

        let output = poseidon
            .hash(&[state_f, round_f, input_f])
            .expect("Poseidon hash failed");

        let mut new_state = [0u8; 32];
        output.serialize_uncompressed(&mut new_state[..]).unwrap();

        // serialize_uncompressed gives LE bytes, keep as is (no reverse)
        self.update_state(new_state);
    }

    /// Loads arbitrary byte lengths using ceil(out/32) invocations of 32 byte randoms
    fn challenge_bytes(&mut self, out: &mut [u8]) {
        let mut remaining_len = out.len();
        let mut start = 0;
        while remaining_len > 32 {
            self.challenge_bytes32(&mut out[start..start + 32]);
            start += 32;
            remaining_len -= 32;
        }
        let mut full_rand = [0u8; 32];
        self.challenge_bytes32(&mut full_rand);
        out[start..start + remaining_len].copy_from_slice(&full_rand[..remaining_len]);
    }

    /// Loads exactly 32 bytes from the transcript
    fn challenge_bytes32(&mut self, out: &mut [u8]) {
        assert_eq!(out.len(), 32);
        let mut poseidon = Self::hasher();
        let state_f = F::from_le_bytes_mod_order(&self.state);
        let round_f = F::from(self.n_rounds as u64);
        let zero = F::zero();
        let output = poseidon
            .hash(&[state_f, round_f, zero])
            .expect("Poseidon hash failed");

        let mut rand = [0u8; 32];
        output.serialize_uncompressed(&mut rand[..]).unwrap();

        // serialize_uncompressed gives LE bytes, keep as is (no reverse)
        out.copy_from_slice(&rand);
        self.update_state(rand);
    }

    fn update_state(&mut self, new_state: [u8; 32]) {
        self.state = new_state;
        self.n_rounds += 1;
        #[cfg(test)]
        {
            if let Some(expected_state_history) = &self.expected_state_history {
                assert!(
                    self.n_rounds as usize <= expected_state_history.len(),
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

impl<F: PrimeField, P: PoseidonParams<F>> Transcript for PoseidonTranscript<F, P> {
    fn new(label: &'static [u8]) -> Self {
        // Hash in the label
        assert!(label.len() <= 32);
        let mut poseidon = Self::hasher();

        let mut label_padded = [0u8; 32];
        label_padded[..label.len()].copy_from_slice(label);
        let label_f = F::from_le_bytes_mod_order(&label_padded);

        let zero = F::zero();
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
            _marker: PhantomData,
        }
    }

    #[cfg(test)]
    fn compare_to(&mut self, other: Self) {
        self.expected_state_history = Some(other.state_history);
    }

    fn append_message(&mut self, msg: &'static [u8]) {
        // We require all messages to fit into one evm word and then right pad them
        // right padding matches the format of the strings when cast to bytes 32 in solidity
        assert!(msg.len() <= 32);
        let mut packed = [0u8; 32];
        packed[..msg.len()].copy_from_slice(msg);
        self.hash_bytes32_and_update(&packed);
    }

    fn append_bytes(&mut self, bytes: &[u8]) {
        // Hash all bytes using Poseidon with domain separation via n_rounds.
        // First chunk: hash(state, n_rounds, chunk), includes domain separator.
        // Subsequent chunks: hash(prev, 0, chunk), chained but without redundant n_rounds.
        // This matches Blake2b/Keccak semantics (n_rounds included once per append operation).
        let mut poseidon = Self::hasher();
        let state_f = F::from_le_bytes_mod_order(&self.state);
        let round_f = F::from(self.n_rounds as u64);
        let zero = F::zero();

        let mut chunks = bytes.chunks(32);

        // First chunk: includes n_rounds for domain separation
        let mut current = if let Some(first_chunk) = chunks.next() {
            let mut padded = [0u8; 32];
            padded[..first_chunk.len()].copy_from_slice(first_chunk);
            let chunk_f = F::from_le_bytes_mod_order(&padded);
            poseidon
                .hash(&[state_f, round_f, chunk_f])
                .expect("Poseidon hash failed")
        } else {
            // Empty bytes: just hash state with n_rounds and zero
            poseidon
                .hash(&[state_f, round_f, zero])
                .expect("Poseidon hash failed")
        };

        // Remaining chunks: no n_rounds (already accounted for)
        for chunk in chunks {
            let mut padded = [0u8; 32];
            padded[..chunk.len()].copy_from_slice(chunk);
            let chunk_f = F::from_le_bytes_mod_order(&padded);
            current = poseidon
                .hash(&[current, zero, chunk_f])
                .expect("Poseidon hash failed");
        }

        let mut new_state = [0u8; 32];
        current.serialize_uncompressed(&mut new_state[..]).unwrap();

        self.update_state(new_state);
    }

    fn append_u64(&mut self, x: u64) {
        // Allocate into a 32 byte region (BE-padded to match EVM word format)
        let mut packed = [0u8; 32];
        packed[24..].copy_from_slice(&x.to_be_bytes());
        self.hash_bytes32_and_update(&packed);
    }

    fn append_scalar<JF: JoltField>(&mut self, scalar: &JF) {
        let mut buf = vec![];
        scalar.serialize_uncompressed(&mut buf).unwrap();
        // Serialize uncompressed gives the scalar in LE byte order which is not
        // a natural representation in the EVM for scalar math so we reverse
        // to get an EVM compatible version.
        buf = buf.into_iter().rev().collect();
        self.append_bytes(&buf);
    }

    fn append_serializable<S: CanonicalSerialize>(&mut self, scalar: &S) {
        let mut buf = vec![];
        scalar.serialize_uncompressed(&mut buf).unwrap();
        // Serialize uncompressed gives the scalar in LE byte order which is not
        // a natural representation in the EVM for scalar math so we reverse
        // to get an EVM compatible version.
        buf = buf.into_iter().rev().collect();
        self.append_bytes(&buf);
    }

    fn append_scalars<JF: JoltField>(&mut self, scalars: &[impl Borrow<JF>]) {
        self.append_message(b"begin_append_vector");
        for item in scalars.iter() {
            self.append_scalar(item.borrow());
        }
        self.append_message(b"end_append_vector");
    }

    fn append_point<G: CurveGroup>(&mut self, point: &G) {
        // If we add the point at infinity then we hash over a region of zeros
        if point.is_zero() {
            self.append_bytes(&[0_u8; 64]);
            return;
        }

        let aff = point.into_affine();
        let mut x_bytes = vec![];
        let mut y_bytes = vec![];
        // The native serialize for the points are le encoded in x,y format and simply reversing
        // can lead to errors so we extract the affine coordinates and the encode them be before writing
        let x = aff.x().unwrap();
        x.serialize_compressed(&mut x_bytes).unwrap();
        x_bytes = x_bytes.into_iter().rev().collect();
        let y = aff.y().unwrap();
        y.serialize_compressed(&mut y_bytes).unwrap();
        y_bytes = y_bytes.into_iter().rev().collect();

        // Concatenate x and y (64 bytes total), then hash using append_bytes.
        // Blake2b does: hasher().chain_update(x).chain_update(y).finalize()
        // which is equivalent to hasher().chain_update(x||y).finalize()
        // append_bytes treats the 64 bytes as a single logical unit with one n_rounds increment.
        let mut combined = x_bytes;
        combined.extend_from_slice(&y_bytes);
        self.append_bytes(&combined);
    }

    fn append_points<G: CurveGroup>(&mut self, points: &[G]) {
        self.append_message(b"begin_append_vector");
        for item in points.iter() {
            self.append_point(item);
        }
        self.append_message(b"end_append_vector");
    }

    fn challenge_u128(&mut self) -> u128 {
        let mut buf = vec![0u8; 16];
        self.challenge_bytes(&mut buf);
        buf = buf.into_iter().rev().collect();
        u128::from_be_bytes(buf.try_into().unwrap())
    }

    fn challenge_scalar<JF: JoltField>(&mut self) -> JF {
        // Under the hood all Fr are 128 bits for performance
        self.challenge_scalar_128_bits()
    }

    fn challenge_scalar_128_bits<JF: JoltField>(&mut self) -> JF {
        let mut buf = vec![0u8; 16];
        self.challenge_bytes(&mut buf);

        buf = buf.into_iter().rev().collect();
        JF::from_bytes(&buf)
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
        // The smaller challenge which is then converted into a
        // MontU128Challenge
        let challenge_scalar: u128 = self.challenge_u128();
        JF::Challenge::from(challenge_scalar)
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
    use ark_std::Zero;
    use std::collections::HashSet;

    // Use the Fr variant for backwards-compatible tests
    type TestTranscript = PoseidonTranscriptFr;

    #[test]
    fn test_challenge_scalar_128_bits() {
        let mut transcript = TestTranscript::new(b"test_128_bit_scalar");
        let mut scalars = HashSet::new();

        for i in 0..10000 {
            let scalar: Fr = transcript.challenge_scalar_128_bits();

            let num_bits = scalar.num_bits();
            assert!(
                num_bits <= 128,
                "Scalar at iteration {i} has {num_bits} bits, expected <= 128",
            );

            assert!(
                scalars.insert(scalar),
                "Duplicate scalar found at iteration {i}",
            );
        }
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
        transcript1.append_u64(123);
        transcript1.append_bytes(b"test data");
        let challenge1: Fr = transcript1.challenge_scalar();

        let mut transcript2 = TestTranscript::new(b"deterministic");
        transcript2.append_u64(123);
        transcript2.append_bytes(b"test data");
        let challenge2: Fr = transcript2.challenge_scalar();

        assert_eq!(challenge1, challenge2);
    }

    #[test]
    fn test_label_padding_consistency() {
        // Test that padding is consistent: shorter label gets zero-padded to 32 bytes
        let label_27_bytes = b"a_label_with_27_bytes_here_"; // 27 bytes
        let label_28_bytes = b"a_label_with_27_bytes_here_\x00"; // 28 bytes with explicit null

        let transcript1 = TestTranscript::new(label_27_bytes);
        let transcript2 = TestTranscript::new(label_28_bytes);

        // After padding, 27-byte label becomes identical to 28-byte label with trailing null
        // So they should produce the SAME initial state (this is correct behavior)
        assert_eq!(
            transcript1.state, transcript2.state,
            "27-byte label padded with null should equal explicit 28-byte label with trailing null"
        );
    }

    #[test]
    fn test_append_bytes_empty() {
        let mut transcript1 = TestTranscript::new(b"empty_test");
        let mut transcript2 = TestTranscript::new(b"empty_test");

        transcript1.append_bytes(&[]);
        transcript2.append_bytes(&[0u8; 0]);

        assert_eq!(transcript1.state, transcript2.state);
        assert_eq!(transcript1.n_rounds, 1);
    }

    #[test]
    fn test_append_scalar() {
        let mut transcript1 = TestTranscript::new(b"scalar_test");
        let mut transcript2 = TestTranscript::new(b"scalar_test");

        let scalar = Fr::from(12345u64);
        transcript1.append_scalar(&scalar);
        transcript2.append_scalar(&scalar);

        assert_eq!(transcript1.state, transcript2.state);

        // Different scalar should produce different state
        let mut transcript3 = TestTranscript::new(b"scalar_test");
        transcript3.append_scalar(&Fr::from(99999u64));
        assert_ne!(transcript1.state, transcript3.state);
    }

    #[test]
    fn test_append_point() {
        use ark_bn254::G1Projective;
        use ark_std::UniformRand;

        let mut rng = ark_std::test_rng();
        let point = G1Projective::rand(&mut rng);

        let mut transcript1 = TestTranscript::new(b"point_test");
        let mut transcript2 = TestTranscript::new(b"point_test");

        transcript1.append_point(&point);
        transcript2.append_point(&point);

        assert_eq!(transcript1.state, transcript2.state);
    }

    #[test]
    fn test_append_point_infinity() {
        use ark_bn254::G1Projective;

        let infinity = G1Projective::default(); // point at infinity

        let mut transcript = TestTranscript::new(b"infinity_test");
        transcript.append_point(&infinity);

        // Should not panic and should update state
        assert_eq!(transcript.n_rounds, 1);
    }

    #[test]
    fn test_append_scalars() {
        let mut transcript1 = TestTranscript::new(b"scalars_test");
        let mut transcript2 = TestTranscript::new(b"scalars_test");

        let scalars: Vec<Fr> = vec![Fr::from(1u64), Fr::from(2u64), Fr::from(3u64)];
        transcript1.append_scalars(&scalars);
        transcript2.append_scalars(&scalars);

        assert_eq!(transcript1.state, transcript2.state);

        // Different scalars should produce different state
        let mut transcript3 = TestTranscript::new(b"scalars_test");
        let other_scalars: Vec<Fr> = vec![Fr::from(4u64), Fr::from(5u64), Fr::from(6u64)];
        transcript3.append_scalars(&other_scalars);
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

        transcript1.append_points(&points);
        transcript2.append_points(&points);

        assert_eq!(transcript1.state, transcript2.state);
    }

    #[test]
    fn test_append_serializable() {
        let mut transcript1 = TestTranscript::new(b"serializable_test");
        let mut transcript2 = TestTranscript::new(b"serializable_test");

        let scalar = Fr::from(12345u64);
        transcript1.append_serializable(&scalar);
        transcript2.append_serializable(&scalar);

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

    #[test]
    fn test_fq_transcript_works() {
        // Test that Fq transcript can be created and used
        let mut transcript: PoseidonTranscriptFq = Transcript::new(b"fq_test");
        transcript.append_u64(12345);
        transcript.append_bytes(b"test data");

        // Should produce valid challenges
        let challenge: Fq = Fq::from_le_bytes_mod_order(&{
            let mut buf = [0u8; 32];
            transcript.challenge_bytes(&mut buf);
            buf
        });

        assert!(!challenge.is_zero());
    }

    #[test]
    fn test_fr_and_fq_produce_different_results() {
        // Fr and Fq transcripts should produce different results
        // (different MDS matrices and round constants)
        let fr_transcript: PoseidonTranscriptFr = Transcript::new(b"test");
        let fq_transcript: PoseidonTranscriptFq = Transcript::new(b"test");

        // Initial states should differ due to different Poseidon parameters
        assert_ne!(fr_transcript.state, fq_transcript.state);
    }
}
