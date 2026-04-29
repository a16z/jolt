//! Wrapper types bridging dory-pcs to jolt-openings.

use std::io::Cursor;

use ark_bn254::{Fq12, Fr as ArkFrInner};
use ark_ff::{AdditiveGroup, Field as ArkField, PrimeField};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use dory::backends::arkworks::{ArkDoryProof, ArkG1, ArkworksProverSetup, ArkworksVerifierSetup};
use dory::primitives::serialization::{DoryDeserialize, DorySerialize};
use jolt_crypto::{Bn254G1, Bn254GT, HomomorphicCommitment};
use jolt_transcript::{AppendToTranscript, Transcript};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::scheme::{ark_to_jolt_gt, jolt_gt_ref_to_ark, ArkGT};

pub const MAX_SERIALIZED_PROOF_ROUNDS: usize = 64;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DoryCommitment(pub Bn254GT);

impl Serialize for DoryCommitment {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        dory_serialize(jolt_gt_ref_to_ark(&self.0), serializer)
    }
}

impl<'de> Deserialize<'de> for DoryCommitment {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let buf: Vec<u8> = Deserialize::deserialize(deserializer)?;
        let ark_gt: ArkGT =
            DoryDeserialize::deserialize_compressed(&buf[..]).map_err(serde::de::Error::custom)?;
        validate_gt(&ark_gt).map_err(serde::de::Error::custom)?;
        Ok(Self(ark_to_jolt_gt(&ark_gt)))
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
        validate_proof_round_count(&buf).map_err(serde::de::Error::custom)?;
        let proof: ArkDoryProof = CanonicalDeserialize::deserialize_compressed(&buf[..])
            .map_err(serde::de::Error::custom)?;
        validate_proof_shape(&proof).map_err(serde::de::Error::custom)?;
        Ok(Self(proof))
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
pub struct DoryHint(pub Vec<Bn254G1>);

#[derive(Clone)]
pub struct DoryPartialCommitment {
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

fn validate_gt(gt: &ArkGT) -> Result<(), &'static str> {
    if gt.0 == Fq12::ZERO {
        return Err("GT element is zero (not in r-torsion subgroup)");
    }

    if gt.0.pow(ArkFrInner::MODULUS) != Fq12::ONE {
        return Err("GT element is not in the r-torsion subgroup");
    }

    Ok(())
}

fn validate_proof_round_count(buf: &[u8]) -> Result<(), String> {
    let mut cursor = Cursor::new(buf);
    let _: ArkGT = CanonicalDeserialize::deserialize_compressed(&mut cursor)
        .map_err(|e| format!("invalid Dory proof VMV c: {e}"))?;
    let _: ArkGT = CanonicalDeserialize::deserialize_compressed(&mut cursor)
        .map_err(|e| format!("invalid Dory proof VMV d2: {e}"))?;
    let _: ArkG1 = CanonicalDeserialize::deserialize_compressed(&mut cursor)
        .map_err(|e| format!("invalid Dory proof VMV e1: {e}"))?;

    let num_rounds: u32 = CanonicalDeserialize::deserialize_compressed(&mut cursor)
        .map_err(|e| format!("invalid Dory proof round count: {e}"))?;
    let num_rounds = num_rounds as usize;
    if num_rounds > MAX_SERIALIZED_PROOF_ROUNDS {
        return Err(format!(
            "Dory proof round count ({num_rounds}) exceeds maximum ({MAX_SERIALIZED_PROOF_ROUNDS})"
        ));
    }

    Ok(())
}

fn validate_proof_shape(proof: &ArkDoryProof) -> Result<(), String> {
    let num_rounds = proof.first_messages.len();
    if num_rounds > MAX_SERIALIZED_PROOF_ROUNDS {
        return Err(format!(
            "Dory proof round count ({num_rounds}) exceeds maximum ({MAX_SERIALIZED_PROOF_ROUNDS})"
        ));
    }
    if proof.second_messages.len() != num_rounds {
        return Err(format!(
            "Dory proof has mismatched round vectors: first={} second={}",
            num_rounds,
            proof.second_messages.len()
        ));
    }
    if proof.sigma != num_rounds {
        return Err(format!(
            "Dory proof sigma ({}) does not match round count ({num_rounds})",
            proof.sigma
        ));
    }
    if proof.nu > proof.sigma {
        return Err(format!(
            "Dory proof has invalid matrix shape: nu={} sigma={}",
            proof.nu, proof.sigma
        ));
    }
    let _ = proof
        .nu
        .checked_add(proof.sigma)
        .ok_or_else(|| "Dory proof dimension overflow".to_string())?;
    Ok(())
}

#[cfg(test)]
#[expect(clippy::expect_used, reason = "tests may panic on assertion failures")]
mod tests {
    use super::*;
    use ark_bn254::{Fq12, Fr as ArkFrInner};
    use ark_ff::{AdditiveGroup, Field as ArkField, PrimeField};
    use ark_serialize::CanonicalSerialize;
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
        let (commitment, _) = crate::DoryScheme::commit(poly.evaluations(), &prover_setup);

        let serialized = serde_json::to_vec(&commitment).expect("serialize commitment");
        let deserialized: DoryCommitment =
            serde_json::from_slice(&serialized).expect("deserialize commitment");

        assert_eq!(commitment, deserialized);
    }

    #[test]
    fn dory_commitment_rejects_zero_gt() {
        let mut bytes = Vec::new();
        Fq12::ZERO
            .serialize_compressed(&mut bytes)
            .expect("serialize zero GT");
        let encoded = serde_json::to_vec(&bytes).expect("encode bytes");

        let result = serde_json::from_slice::<DoryCommitment>(&encoded);
        assert!(result.is_err(), "zero GT must be rejected");
    }

    #[test]
    fn dory_commitment_rejects_non_torsion_gt() {
        let candidate = Fq12::from(2u64);
        assert_ne!(
            candidate.pow(ArkFrInner::MODULUS),
            Fq12::ONE,
            "test candidate should not be in the r-torsion subgroup"
        );

        let mut bytes = Vec::new();
        candidate
            .serialize_compressed(&mut bytes)
            .expect("serialize non-torsion GT");
        let encoded = serde_json::to_vec(&bytes).expect("encode bytes");

        let result = serde_json::from_slice::<DoryCommitment>(&encoded);
        assert!(result.is_err(), "non-r-torsion GT must be rejected");
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
            .map(|_| <Fr as Field>::random(&mut rng))
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
    fn dory_proof_rejects_oversized_round_count_before_full_decode() {
        let num_vars = 2;
        let mut rng = ChaCha20Rng::seed_from_u64(403);

        let prover_setup = crate::DoryScheme::setup_prover(num_vars);
        let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
        let point: Vec<Fr> = (0..num_vars)
            .map(|_| <Fr as Field>::random(&mut rng))
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
        CanonicalSerialize::serialize_compressed(&proof.0.vmv_message.c, &mut prefix)
            .expect("serialize c");
        CanonicalSerialize::serialize_compressed(&proof.0.vmv_message.d2, &mut prefix)
            .expect("serialize d2");
        CanonicalSerialize::serialize_compressed(&proof.0.vmv_message.e1, &mut prefix)
            .expect("serialize e1");

        let mut oversized_rounds = Vec::new();
        CanonicalSerialize::serialize_compressed(&u32::MAX, &mut oversized_rounds)
            .expect("serialize round count");
        let start = prefix.len();
        let end = start + oversized_rounds.len();
        bytes[start..end].copy_from_slice(&oversized_rounds);

        let encoded = serde_json::to_vec(&bytes).expect("encode proof bytes");
        let result = serde_json::from_slice::<DoryProof>(&encoded);
        assert!(result.is_err(), "oversized round count must be rejected");
    }
}
