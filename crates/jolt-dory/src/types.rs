//! Public Dory types for the `jolt-openings` commitment traits.

use std::io::{Cursor, Read, Result as IoResult, Write};

use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, SerializationError, Valid, Validate,
};
use dory::backends::arkworks::{
    ArkDoryProof, ArkG1, ArkGT, ArkworksProverSetup, ArkworksVerifierSetup,
};
use jolt_crypto::{Bn254G1, Bn254GT, HomomorphicCommitment};
use jolt_field::Fr;
use jolt_transcript::{AppendToTranscript, Transcript};
use serde::{de::Error as DeError, ser::Error as SerError};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Caps the upstream `Vec::with_capacity(num_rounds)` allocation against
/// attacker-supplied round counts during proof deserialization. Real Dory
/// proofs use `num_rounds = ceil(log2(N/2))` for an N-coefficient polynomial,
/// so 64 covers polynomials up to 2^65 evaluations.
pub const MAX_SERIALIZED_PROOF_ROUNDS: usize = 64;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DoryCommitment(pub Bn254GT);

impl Serialize for DoryCommitment {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.0.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for DoryCommitment {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        // Bn254GT::deserialize enforces the GT subgroup check (rejects zero
        // and non-r-torsion elements), which the previous round-trip through
        // ArkGT skipped.
        Bn254GT::deserialize(deserializer).map(Self)
    }
}

impl CanonicalSerialize for DoryCommitment {
    fn serialize_with_mode<W: Write>(
        &self,
        writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.0.serialize_with_mode(writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.0.serialized_size(compress)
    }
}

impl Valid for DoryCommitment {
    fn check(&self) -> Result<(), SerializationError> {
        self.0.check()
    }
}

impl CanonicalDeserialize for DoryCommitment {
    fn deserialize_with_mode<R: Read>(
        reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        Bn254GT::deserialize_with_mode(reader, compress, validate).map(Self)
    }
}

impl AppendToTranscript for DoryCommitment {
    fn append_to_transcript<T: Transcript>(&self, transcript: &mut T) {
        self.0.append_to_transcript(transcript);
    }
}

impl<F: jolt_field::Field> HomomorphicCommitment<F> for DoryCommitment {
    #[inline]
    fn linear_combine(c1: &Self, c2: &Self, scalar: &F) -> Self {
        Self(HomomorphicCommitment::linear_combine(&c1.0, &c2.0, scalar))
    }
}

#[derive(Clone, Debug)]
pub struct DoryProof(pub ArkDoryProof);

impl Serialize for DoryProof {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        canonical_serialize(&self.0, serializer)
    }
}

impl<'de> Deserialize<'de> for DoryProof {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let buf: Vec<u8> = Deserialize::deserialize(deserializer)?;
        validate_proof_round_count(&buf).map_err(DeError::custom)?;
        ArkDoryProof::deserialize_compressed(&buf[..])
            .map_err(DeError::custom)
            .map(Self)
    }
}

impl CanonicalSerialize for DoryProof {
    fn serialize_with_mode<W: Write>(
        &self,
        writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.0.serialize_with_mode(writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.0.serialized_size(compress)
    }
}

impl Valid for DoryProof {
    fn check(&self) -> Result<(), SerializationError> {
        self.0.check()
    }
}

impl CanonicalDeserialize for DoryProof {
    fn deserialize_with_mode<R: Read>(
        reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        deserialize_proof_with_round_limit(reader, compress, validate).map(Self)
    }
}

#[derive(Clone, Debug)]
pub struct DoryProverSetup(pub ArkworksProverSetup);

impl Serialize for DoryProverSetup {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        canonical_serialize(&self.0, serializer)
    }
}

impl<'de> Deserialize<'de> for DoryProverSetup {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        canonical_deserialize(deserializer).map(Self)
    }
}

impl CanonicalSerialize for DoryProverSetup {
    fn serialize_with_mode<W: Write>(
        &self,
        writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.0.serialize_with_mode(writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.0.serialized_size(compress)
    }
}

impl Valid for DoryProverSetup {
    fn check(&self) -> Result<(), SerializationError> {
        self.0.check()
    }
}

impl CanonicalDeserialize for DoryProverSetup {
    fn deserialize_with_mode<R: Read>(
        reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        ArkworksProverSetup::deserialize_with_mode(reader, compress, validate).map(Self)
    }
}

#[derive(Clone, Debug)]
pub struct DoryVerifierSetup(pub ArkworksVerifierSetup);

impl Serialize for DoryVerifierSetup {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        canonical_serialize(&self.0, serializer)
    }
}

impl<'de> Deserialize<'de> for DoryVerifierSetup {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        canonical_deserialize(deserializer).map(Self)
    }
}

impl CanonicalSerialize for DoryVerifierSetup {
    fn serialize_with_mode<W: Write>(
        &self,
        writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.0.serialize_with_mode(writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.0.serialized_size(compress)
    }
}

impl Valid for DoryVerifierSetup {
    fn check(&self) -> Result<(), SerializationError> {
        self.0.check()
    }
}

impl CanonicalDeserialize for DoryVerifierSetup {
    fn deserialize_with_mode<R: Read>(
        reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        ArkworksVerifierSetup::deserialize_with_mode(reader, compress, validate).map(Self)
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DoryHint {
    pub(crate) row_commitments: Vec<Bn254G1>,
    pub(crate) commit_blind: Fr,
    pub(crate) chunk_len: usize,
}

impl DoryHint {
    pub(crate) fn new(row_commitments: Vec<Bn254G1>, commit_blind: Fr, chunk_len: usize) -> Self {
        Self {
            row_commitments,
            commit_blind,
            chunk_len,
        }
    }
}

fn canonical_serialize<T: CanonicalSerialize, S: Serializer>(
    value: &T,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    let mut buf = Vec::new();
    value
        .serialize_compressed(&mut buf)
        .map_err(SerError::custom)?;
    serializer.serialize_bytes(&buf)
}

fn canonical_deserialize<'de, T: CanonicalDeserialize, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<T, D::Error> {
    let buf: Vec<u8> = Deserialize::deserialize(deserializer)?;
    T::deserialize_compressed(&buf[..]).map_err(DeError::custom)
}

/// Pre-validates the round count from the proof's wire bytes before invoking
/// the upstream `CanonicalDeserialize`, which calls `Vec::with_capacity(num_rounds)`
/// and would OOM on attacker-supplied lengths near `u32::MAX`.
fn validate_proof_round_count(buf: &[u8]) -> Result<(), String> {
    let mut cursor = Cursor::new(buf);
    read_and_validate_proof_round_count(&mut cursor, Compress::Yes, Validate::No)
        .map_err(|e| format!("invalid Dory proof prefix: {e}"))
}

fn deserialize_proof_with_round_limit<R: Read>(
    reader: R,
    compress: Compress,
    validate: Validate,
) -> Result<ArkDoryProof, SerializationError> {
    let mut recorder = RecordingReader::new(reader);
    read_and_validate_proof_round_count(&mut recorder, compress, validate)?;
    let replay = Cursor::new(recorder.recorded).chain(recorder.reader);
    ArkDoryProof::deserialize_with_mode(replay, compress, validate)
}

fn read_and_validate_proof_round_count<R: Read>(
    reader: &mut R,
    compress: Compress,
    validate: Validate,
) -> Result<(), SerializationError> {
    let _: ArkGT =
        CanonicalDeserialize::deserialize_with_mode(reader.by_ref(), compress, validate)?;
    let _: ArkGT =
        CanonicalDeserialize::deserialize_with_mode(reader.by_ref(), compress, validate)?;
    let _: ArkG1 =
        CanonicalDeserialize::deserialize_with_mode(reader.by_ref(), compress, validate)?;
    let num_rounds: u32 =
        CanonicalDeserialize::deserialize_with_mode(reader.by_ref(), compress, validate)?;
    if num_rounds as usize > MAX_SERIALIZED_PROOF_ROUNDS {
        return Err(SerializationError::InvalidData);
    }
    Ok(())
}

struct RecordingReader<R> {
    reader: R,
    recorded: Vec<u8>,
}

impl<R> RecordingReader<R> {
    fn new(reader: R) -> Self {
        Self {
            reader,
            recorded: Vec::new(),
        }
    }
}

impl<R: Read> Read for RecordingReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> IoResult<usize> {
        let count = self.reader.read(buf)?;
        self.recorded.extend_from_slice(&buf[..count]);
        Ok(count)
    }
}

#[cfg(test)]
#[expect(clippy::expect_used, reason = "tests may panic on assertion failures")]
mod tests {
    use super::*;
    use crate::DoryScheme;
    use jolt_field::{Fr, RandomSampling};
    use jolt_openings::{CommitmentScheme, CommitmentSchemeVerifier};
    use jolt_poly::Polynomial;
    use jolt_transcript::{Blake2bTranscript, Transcript};
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    #[test]
    fn dory_commitment_serde_round_trip() {
        let num_vars = 3;
        let mut rng = ChaCha20Rng::seed_from_u64(400);

        let prover_setup = DoryScheme::setup_prover(num_vars);
        let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
        let (commitment, _) = DoryScheme::commit(poly.evaluations(), &prover_setup);

        let serialized = serde_json::to_vec(&commitment).expect("serialize commitment");
        let deserialized: DoryCommitment =
            serde_json::from_slice(&serialized).expect("deserialize commitment");

        assert_eq!(commitment, deserialized);
    }

    #[test]
    fn dory_verifier_setup_serde_round_trip() {
        let num_vars = 2;
        let verifier_setup = DoryScheme::setup_verifier(num_vars);

        let serialized = serde_json::to_vec(&verifier_setup).expect("serialize verifier setup");
        let deserialized: DoryVerifierSetup =
            serde_json::from_slice(&serialized).expect("deserialize verifier setup");

        let mut rng = ChaCha20Rng::seed_from_u64(401);
        let prover_setup = DoryScheme::setup_prover(num_vars);

        let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
        let point: Vec<Fr> = (0..num_vars)
            .map(|_| <Fr as RandomSampling>::random(&mut rng))
            .collect();
        let eval = poly.evaluate(&point);
        let (commitment, hint) = DoryScheme::commit(poly.evaluations(), &prover_setup);

        let mut prove_transcript = Blake2bTranscript::new(b"serde-vs");
        let proof = DoryScheme::open(
            &poly,
            &point,
            eval,
            &prover_setup,
            Some(hint),
            &mut prove_transcript,
        );

        let mut verify_transcript = Blake2bTranscript::new(b"serde-vs");
        let result = DoryScheme::verify(
            &commitment,
            &point,
            eval,
            &proof,
            &deserialized,
            &mut verify_transcript,
        );
        assert!(
            result.is_ok(),
            "deserialized verifier setup must verify correctly"
        );
    }

    #[test]
    fn dory_proof_serde_round_trip() {
        let num_vars = 2;
        let mut rng = ChaCha20Rng::seed_from_u64(402);

        let prover_setup = DoryScheme::setup_prover(num_vars);

        let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
        let point: Vec<Fr> = (0..num_vars)
            .map(|_| <Fr as RandomSampling>::random(&mut rng))
            .collect();
        let eval = poly.evaluate(&point);

        let mut transcript = Blake2bTranscript::new(b"serde-bp");
        let proof = DoryScheme::open(&poly, &point, eval, &prover_setup, None, &mut transcript);

        let serialized = serde_json::to_vec(&proof).expect("serialize proof");
        let deserialized: DoryProof =
            serde_json::from_slice(&serialized).expect("deserialize proof");

        let verifier_setup = DoryVerifierSetup(prover_setup.0.to_verifier_setup());
        let (commitment, _) = DoryScheme::commit(poly.evaluations(), &prover_setup);

        let mut verify_transcript = Blake2bTranscript::new(b"serde-bp");
        let result = DoryScheme::verify(
            &commitment,
            &point,
            eval,
            &deserialized,
            &verifier_setup,
            &mut verify_transcript,
        );
        assert!(result.is_ok(), "deserialized proof must verify correctly");
    }

    #[test]
    fn dory_canonical_round_trips_support_core_proof_storage() {
        let num_vars = 2;
        let mut rng = ChaCha20Rng::seed_from_u64(405);

        let prover_setup = DoryScheme::setup_prover(num_vars);
        let mut prover_setup_bytes = Vec::new();
        prover_setup
            .serialize_compressed(&mut prover_setup_bytes)
            .expect("serialize prover setup");
        let prover_setup = DoryProverSetup::deserialize_compressed(&prover_setup_bytes[..])
            .expect("deserialize prover setup");

        let verifier_setup = DoryVerifierSetup(prover_setup.0.to_verifier_setup());
        let mut verifier_setup_bytes = Vec::new();
        verifier_setup
            .serialize_compressed(&mut verifier_setup_bytes)
            .expect("serialize verifier setup");
        let verifier_setup = DoryVerifierSetup::deserialize_compressed(&verifier_setup_bytes[..])
            .expect("deserialize verifier setup");

        let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
        let point: Vec<Fr> = (0..num_vars)
            .map(|_| <Fr as RandomSampling>::random(&mut rng))
            .collect();
        let eval = poly.evaluate(&point);
        let (commitment, hint) = DoryScheme::commit(poly.evaluations(), &prover_setup);

        let mut commitment_bytes = Vec::new();
        commitment
            .serialize_compressed(&mut commitment_bytes)
            .expect("serialize commitment");
        let commitment = DoryCommitment::deserialize_compressed(&commitment_bytes[..])
            .expect("deserialize commitment");

        let mut prove_transcript = Blake2bTranscript::new(b"canonical");
        let proof = DoryScheme::open(
            &poly,
            &point,
            eval,
            &prover_setup,
            Some(hint),
            &mut prove_transcript,
        );

        let mut proof_bytes = Vec::new();
        proof
            .serialize_compressed(&mut proof_bytes)
            .expect("serialize proof");
        let proof = DoryProof::deserialize_compressed(&proof_bytes[..]).expect("deserialize proof");

        let mut verify_transcript = Blake2bTranscript::new(b"canonical");
        let result = DoryScheme::verify(
            &commitment,
            &point,
            eval,
            &proof,
            &verifier_setup,
            &mut verify_transcript,
        );
        assert!(
            result.is_ok(),
            "canonical round-tripped Dory types must verify correctly"
        );
    }

    #[test]
    fn dory_proof_deserialization_preserves_following_bytes() {
        let num_vars = 2;
        let mut rng = ChaCha20Rng::seed_from_u64(406);

        let prover_setup = DoryScheme::setup_prover(num_vars);
        let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
        let point: Vec<Fr> = (0..num_vars)
            .map(|_| <Fr as RandomSampling>::random(&mut rng))
            .collect();
        let eval = poly.evaluate(&point);

        let mut transcript = Blake2bTranscript::new(b"canonical-stream");
        let proof = DoryScheme::open(&poly, &point, eval, &prover_setup, None, &mut transcript);

        let mut bytes = Vec::new();
        vec![proof]
            .serialize_compressed(&mut bytes)
            .expect("serialize proof vector");
        7_u64
            .serialize_compressed(&mut bytes)
            .expect("serialize trailing field");

        let mut cursor = Cursor::new(bytes);
        let proofs = Vec::<DoryProof>::deserialize_compressed(&mut cursor)
            .expect("deserialize proof vector");
        let trailing =
            u64::deserialize_compressed(&mut cursor).expect("deserialize trailing field");

        assert_eq!(proofs.len(), 1);
        assert_eq!(trailing, 7);
    }

    #[test]
    fn dory_proof_rejects_oversized_round_count() {
        let num_vars = 2;
        let mut rng = ChaCha20Rng::seed_from_u64(403);

        let prover_setup = DoryScheme::setup_prover(num_vars);
        let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
        let point: Vec<Fr> = (0..num_vars)
            .map(|_| <Fr as RandomSampling>::random(&mut rng))
            .collect();
        let eval = poly.evaluate(&point);

        let mut transcript = Blake2bTranscript::new(b"serde-oversized");
        let proof = DoryScheme::open(&poly, &point, eval, &prover_setup, None, &mut transcript);

        let mut bytes = Vec::new();
        proof
            .0
            .serialize_compressed(&mut bytes)
            .expect("serialize proof");

        let mut prefix = Vec::new();
        proof
            .0
            .vmv_message
            .c
            .serialize_compressed(&mut prefix)
            .expect("serialize VMV.c");
        proof
            .0
            .vmv_message
            .d2
            .serialize_compressed(&mut prefix)
            .expect("serialize VMV.d2");
        proof
            .0
            .vmv_message
            .e1
            .serialize_compressed(&mut prefix)
            .expect("serialize VMV.e1");

        let mut oversized_rounds = Vec::new();
        u32::MAX
            .serialize_compressed(&mut oversized_rounds)
            .expect("serialize round count");
        bytes[prefix.len()..prefix.len() + oversized_rounds.len()]
            .copy_from_slice(&oversized_rounds);

        let encoded = serde_json::to_vec(&bytes).expect("encode proof bytes");
        let result = serde_json::from_slice::<DoryProof>(&encoded);
        assert!(result.is_err(), "oversized round count must be rejected");
    }
}
