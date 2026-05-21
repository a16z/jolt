//! Wrapper types bridging dory-pcs to jolt-openings.

use std::io::Cursor;

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use dory::backends::arkworks::{
    ArkDoryProof, ArkG1, ArkGT, ArkworksProverSetup, ArkworksVerifierSetup,
};
use jolt_crypto::{Bn254G1, Bn254GT, HomomorphicCommitment};
use jolt_field::Fr;
use jolt_transcript::{AppendToTranscript, Transcript};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Caps the upstream `Vec::with_capacity(num_rounds)` allocation against
/// attacker-supplied round counts during proof deserialization. Real Dory
/// proofs use `num_rounds = ceil(log2(N/2))` for an N-coefficient polynomial,
/// so 64 covers polynomials up to 2^65 evaluations.
pub const MAX_SERIALIZED_PROOF_ROUNDS: usize = 64;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DoryCommitment(pub Bn254GT);

impl Default for DoryCommitment {
    #[inline]
    fn default() -> Self {
        Self(Bn254GT::default())
    }
}

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

impl AppendToTranscript for DoryCommitment {
    fn append_to_transcript<T: Transcript>(&self, transcript: &mut T) {
        self.0.append_to_transcript(transcript);
    }

    fn transcript_payload_len(&self) -> Option<u64> {
        self.0.transcript_payload_len()
    }
}

impl<F: jolt_field::Field> HomomorphicCommitment<F> for DoryCommitment {
    #[inline]
    fn add(c1: &Self, c2: &Self) -> Self {
        Self(<Bn254GT as HomomorphicCommitment<F>>::add(&c1.0, &c2.0))
    }

    #[inline]
    fn linear_combine(c1: &Self, c2: &Self, scalar: &F) -> Self {
        Self(HomomorphicCommitment::linear_combine(&c1.0, &c2.0, scalar))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct DoryProof(pub ArkDoryProof);

impl Eq for DoryProof {}

impl Serialize for DoryProof {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        canonical_serialize(&self.0, serializer)
    }
}

impl<'de> Deserialize<'de> for DoryProof {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let buf: Vec<u8> = Deserialize::deserialize(deserializer)?;
        validate_proof_round_count(&buf).map_err(serde::de::Error::custom)?;
        ArkDoryProof::deserialize_compressed(&buf[..])
            .map_err(serde::de::Error::custom)
            .map(Self)
    }
}

#[derive(Clone)]
pub struct DoryProverSetup(pub ArkworksProverSetup);

#[derive(Clone)]
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

#[derive(Clone, Debug, Default)]
pub struct DoryHint {
    pub(crate) row_commitments: Vec<Bn254G1>,
    pub(crate) commit_blind: Fr,
}

impl DoryHint {
    pub(crate) fn new(row_commitments: Vec<Bn254G1>, commit_blind: Fr) -> Self {
        Self {
            row_commitments,
            commit_blind,
        }
    }
}

#[derive(Clone)]
pub struct DoryPartialCommitment {
    pub row_commitments: Vec<Bn254G1>,
}

fn canonical_serialize<T: CanonicalSerialize, S: Serializer>(
    value: &T,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    let mut buf = Vec::new();
    value
        .serialize_compressed(&mut buf)
        .map_err(serde::ser::Error::custom)?;
    serializer.serialize_bytes(&buf)
}

fn canonical_deserialize<'de, T: CanonicalDeserialize, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<T, D::Error> {
    let buf: Vec<u8> = Deserialize::deserialize(deserializer)?;
    T::deserialize_compressed(&buf[..]).map_err(serde::de::Error::custom)
}

/// Pre-validates the round count from the proof's wire bytes before invoking
/// the upstream `CanonicalDeserialize`, which calls `Vec::with_capacity(num_rounds)`
/// and would OOM on attacker-supplied lengths near `u32::MAX`.
fn validate_proof_round_count(buf: &[u8]) -> Result<(), String> {
    let mut cursor = Cursor::new(buf);
    let _: ArkGT = CanonicalDeserialize::deserialize_compressed(&mut cursor)
        .map_err(|e| format!("invalid Dory proof VMV.c: {e}"))?;
    let _: ArkGT = CanonicalDeserialize::deserialize_compressed(&mut cursor)
        .map_err(|e| format!("invalid Dory proof VMV.d2: {e}"))?;
    let _: ArkG1 = CanonicalDeserialize::deserialize_compressed(&mut cursor)
        .map_err(|e| format!("invalid Dory proof VMV.e1: {e}"))?;
    let num_rounds: u32 = CanonicalDeserialize::deserialize_compressed(&mut cursor)
        .map_err(|e| format!("invalid Dory proof round count: {e}"))?;
    if num_rounds as usize > MAX_SERIALIZED_PROOF_ROUNDS {
        return Err(format!(
            "Dory proof round count ({num_rounds}) exceeds maximum ({MAX_SERIALIZED_PROOF_ROUNDS})"
        ));
    }
    Ok(())
}

#[cfg(test)]
#[expect(clippy::expect_used, reason = "tests may panic on assertion failures")]
mod tests {
    use super::*;
    use jolt_field::RandomSampling;
    use jolt_openings::CommitmentScheme;
    use jolt_poly::Polynomial;
    use jolt_transcript::Transcript;
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    use jolt_field::Fr;

    #[test]
    fn dory_commitment_serde_round_trip() {
        let num_vars = 3;
        let mut rng = ChaCha20Rng::seed_from_u64(400);

        let prover_setup = crate::DoryScheme::setup_prover(num_vars);
        let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
        let (commitment, _) = crate::DoryScheme::commit(poly.evaluations(), &prover_setup);

        let serialized = serde_json::to_vec(&commitment).expect("serialize commitment");
        let deserialized: DoryCommitment =
            serde_json::from_slice(&serialized).expect("deserialize commitment");

        assert_eq!(commitment, deserialized);
    }

    #[test]
    fn dory_verifier_setup_serde_round_trip() {
        let num_vars = 2;
        let verifier_setup = crate::DoryScheme::setup_verifier(num_vars);

        let serialized = serde_json::to_vec(&verifier_setup).expect("serialize verifier setup");
        let deserialized: DoryVerifierSetup =
            serde_json::from_slice(&serialized).expect("deserialize verifier setup");

        let mut rng = ChaCha20Rng::seed_from_u64(401);
        let prover_setup = crate::DoryScheme::setup_prover(num_vars);

        let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
        let point: Vec<Fr> = (0..num_vars)
            .map(|_| <Fr as RandomSampling>::random(&mut rng))
            .collect();
        let eval = poly.evaluate(&point);
        let (commitment, hint) = crate::DoryScheme::commit(poly.evaluations(), &prover_setup);

        let mut prove_transcript = jolt_transcript::Blake2bTranscript::new(b"serde-vs");
        let proof = crate::DoryScheme::open(
            &poly,
            &point,
            eval,
            &prover_setup,
            Some(hint),
            &mut prove_transcript,
        );

        let mut verify_transcript = jolt_transcript::Blake2bTranscript::new(b"serde-vs");
        let result = crate::DoryScheme::verify(
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

        let prover_setup = crate::DoryScheme::setup_prover(num_vars);

        let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
        let point: Vec<Fr> = (0..num_vars)
            .map(|_| <Fr as RandomSampling>::random(&mut rng))
            .collect();
        let eval = poly.evaluate(&point);

        let mut transcript = jolt_transcript::Blake2bTranscript::new(b"serde-bp");
        let proof =
            crate::DoryScheme::open(&poly, &point, eval, &prover_setup, None, &mut transcript);

        let serialized = serde_json::to_vec(&proof).expect("serialize proof");
        let deserialized: DoryProof =
            serde_json::from_slice(&serialized).expect("deserialize proof");

        let verifier_setup = DoryVerifierSetup(prover_setup.0.to_verifier_setup());
        let (commitment, _) = crate::DoryScheme::commit(poly.evaluations(), &prover_setup);

        let mut verify_transcript = jolt_transcript::Blake2bTranscript::new(b"serde-bp");
        let result = crate::DoryScheme::verify(
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
    fn dory_proof_rejects_oversized_round_count() {
        let num_vars = 2;
        let mut rng = ChaCha20Rng::seed_from_u64(403);

        let prover_setup = crate::DoryScheme::setup_prover(num_vars);
        let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
        let point: Vec<Fr> = (0..num_vars)
            .map(|_| <Fr as RandomSampling>::random(&mut rng))
            .collect();
        let eval = poly.evaluate(&point);

        let mut transcript = jolt_transcript::Blake2bTranscript::new(b"serde-oversized");
        let proof =
            crate::DoryScheme::open(&poly, &point, eval, &prover_setup, None, &mut transcript);

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
