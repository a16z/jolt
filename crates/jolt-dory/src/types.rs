//! Wrapper types for dory-pcs types used in the public API.
//!
//! These wrappers bridge between `dory-pcs`'s types and the trait bounds
//! required by `jolt-openings::CommitmentScheme`.

use dory::backends::arkworks::{ArkDoryProof, ArkworksProverSetup, ArkworksVerifierSetup};
use dory::primitives::serialization::{DoryDeserialize, DorySerialize};
use jolt_crypto::{Bn254G1, Bn254GT};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Commitment produced by the Dory scheme.
///
/// Wraps a BN254 pairing target element (`Bn254GT`). The commitment is
/// computed as a multi-pairing of row-level Pedersen commitments in G1
/// with the SRS generators in G2.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DoryCommitment(pub Bn254GT);

impl Serialize for DoryCommitment {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        dory_serialize(&self.0, serializer)
    }
}

impl<'de> Deserialize<'de> for DoryCommitment {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        dory_deserialize(deserializer).map(Self)
    }
}

/// Opening proof for a single polynomial.
///
/// Wraps the `dory-pcs` proof structure. Serialization uses the arkworks
/// backend's concrete `ArkDoryProof` type since the generic `DoryProof`
/// doesn't derive serialization traits.
#[derive(Clone, Debug)]
pub struct DoryProof(pub ArkDoryProof);

impl Serialize for DoryProof {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        canonical_serialize(&self.0, serializer)
    }
}

impl<'de> Deserialize<'de> for DoryProof {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        canonical_deserialize(deserializer).map(Self)
    }
}

/// Prover-side structured reference string (SRS) for Dory.
///
/// Uses the arkworks backend wrapper for disk-persistence and
/// deterministic setup support via `new_from_urs()`.
#[derive(Clone)]
pub struct DoryProverSetup(pub ArkworksProverSetup);

// SAFETY: ArkworksProverSetup contains only group element vectors which are plain data.
// The missing auto-traits come from arkworks type-level plumbing.
unsafe impl Send for DoryProverSetup {}
// SAFETY: Same rationale as Send impl above.
unsafe impl Sync for DoryProverSetup {}

/// Verifier-side structured reference string (SRS) for Dory.
///
/// A subset of the prover SRS sufficient for verifying opening proofs.
#[derive(Clone)]
pub struct DoryVerifierSetup(pub ArkworksVerifierSetup);

// SAFETY: ArkworksVerifierSetup contains only group elements and field elements,
// all plain data without interior mutability.
unsafe impl Send for DoryVerifierSetup {}
// SAFETY: Same rationale as Send impl above.
unsafe impl Sync for DoryVerifierSetup {}

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

/// Row-level commitments used as an opening proof hint.
#[derive(Clone, Debug)]
pub struct DoryHint(pub Vec<Bn254G1>);

/// Partial commitment state accumulated during streaming.
#[derive(Clone)]
pub struct DoryPartialCommitment {
    /// Row commitments accumulated from processed chunks.
    pub row_commitments: Vec<Bn254G1>,
}

fn dory_serialize<T: DorySerialize, S: Serializer>(
    value: &T,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    let mut buf = Vec::new();
    value
        .serialize_compressed(&mut buf)
        .map_err(serde::ser::Error::custom)?;
    serializer.serialize_bytes(&buf)
}

fn dory_deserialize<'de, T: DoryDeserialize, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<T, D::Error> {
    let buf: Vec<u8> = Deserialize::deserialize(deserializer)?;
    T::deserialize_compressed(&buf[..]).map_err(serde::de::Error::custom)
}

fn canonical_serialize<T: ark_serialize::CanonicalSerialize, S: Serializer>(
    value: &T,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    let mut buf = Vec::new();
    value
        .serialize_compressed(&mut buf)
        .map_err(serde::ser::Error::custom)?;
    serializer.serialize_bytes(&buf)
}

fn canonical_deserialize<'de, T: ark_serialize::CanonicalDeserialize, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<T, D::Error> {
    let buf: Vec<u8> = Deserialize::deserialize(deserializer)?;
    T::deserialize_compressed(&buf[..]).map_err(serde::de::Error::custom)
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::Field;
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
        let commitment = crate::DoryScheme::commit(poly.evaluations(), &prover_setup);

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
            .map(|_| <Fr as Field>::random(&mut rng))
            .collect();
        let eval = poly.evaluate(&point);
        let commitment = crate::DoryScheme::commit(poly.evaluations(), &prover_setup);

        let mut prove_transcript = jolt_transcript::Blake2bTranscript::new(b"serde-vs");
        let proof = crate::DoryScheme::open(
            poly.evaluations(),
            &point,
            eval,
            &prover_setup,
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
            .map(|_| <Fr as Field>::random(&mut rng))
            .collect();
        let eval = poly.evaluate(&point);

        let mut transcript = jolt_transcript::Blake2bTranscript::new(b"serde-bp");
        let proof = crate::DoryScheme::open(
            poly.evaluations(),
            &point,
            eval,
            &prover_setup,
            &mut transcript,
        );

        let serialized = serde_json::to_vec(&proof).expect("serialize proof");
        let deserialized: DoryProof =
            serde_json::from_slice(&serialized).expect("deserialize proof");

        let verifier_setup = DoryVerifierSetup(prover_setup.0.to_verifier_setup());
        let commitment = crate::DoryScheme::commit(poly.evaluations(), &prover_setup);

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
}
