//! Wrapper types for dory-pcs types used in the public API.
//!
//! These wrappers bridge between `dory-pcs`'s arkworks backend types and
//! the trait bounds required by `jolt-openings::CommitmentScheme`. In particular,
//! dory-pcs types use arkworks `CanonicalSerialize`/`CanonicalDeserialize` while
//! `jolt-openings` traits require `serde::Serialize`/`serde::Deserialize`. The
//! wrappers provide serde implementations by delegating to canonical serialization.

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use dory::backends::arkworks::{
    ArkDoryProof, ArkFr, ArkG1, ArkGT, ArkworksProverSetup, ArkworksVerifierSetup,
};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Commitment produced by the Dory scheme.
///
/// Wraps a `dory-pcs` pairing target element `ArkGT` (an element of the
/// target group $\mathbb{G}_T$ of the BN254 pairing). The commitment is
/// computed as a multi-pairing of row-level Pedersen commitments in $\mathbb{G}_1$
/// with the SRS generators in $\mathbb{G}_2$.
#[derive(Clone, Debug, PartialEq)]
pub struct DoryCommitment(pub ArkGT);

impl Serialize for DoryCommitment {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        canonical_serialize(&self.0, serializer)
    }
}

impl<'de> Deserialize<'de> for DoryCommitment {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        canonical_deserialize(deserializer).map(Self)
    }
}

/// Opening proof for a single polynomial.
///
/// Wraps the `dory-pcs` proof structure, which contains the inner-product
/// argument transcript for a Dory opening at a given evaluation point.
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

/// Batched opening proof for multiple polynomials.
///
/// A vector of individual Dory proofs, one per polynomial in the batch.
/// Batch verification uses random linear combination (RLC) to reduce
/// multiple claims to a single check.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DoryBatchedProof(pub Vec<DoryProof>);

/// Prover-side structured reference string (SRS) for Dory.
///
/// Contains the $\mathbb{G}_1$ and $\mathbb{G}_2$ generator vectors used for
/// computing Pedersen commitments and generating opening proofs.
#[derive(Clone)]
pub struct DoryProverSetup(pub ArkworksProverSetup);

// SAFETY: `ArkworksProverSetup` contains only group element vectors (`Vec<ArkG1>`,
// `Vec<ArkG2>`) which are `Send + Sync`. The inner `ProverSetup<BN254>` lacks the
// auto-trait markers only because of arkworks type-level plumbing, not due to any
// actual thread-safety hazard.
unsafe impl Send for DoryProverSetup {}
// SAFETY: Same rationale as `Send` impl above.
unsafe impl Sync for DoryProverSetup {}

/// Verifier-side structured reference string (SRS) for Dory.
///
/// A subset of the prover SRS sufficient for verifying opening proofs.
/// Serializable for transmission to verifiers.
#[derive(Clone)]
pub struct DoryVerifierSetup(pub ArkworksVerifierSetup);

// SAFETY: `ArkworksVerifierSetup` contains only group elements, field elements,
// and pairing target elements, all of which are plain data without interior
// mutability. The missing auto-traits are an artifact of arkworks generics.
unsafe impl Send for DoryVerifierSetup {}
// SAFETY: Same rationale as `Send` impl above.
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
///
/// During commitment, Dory computes a Pedersen commitment per row of the
/// polynomial coefficient matrix. These row commitments can be reused when
/// generating an opening proof, avoiding redundant MSM computation.
#[derive(Clone, Debug)]
pub struct DoryHint(pub Vec<ArkG1>);

/// Partial commitment state accumulated during streaming.
///
/// Stores the row-level $\mathbb{G}_1$ commitments computed so far during
/// a streaming commitment session.
#[derive(Clone)]
pub struct DoryPartialCommitment {
    /// Row commitments accumulated from processed chunks.
    pub row_commitments: Vec<ArkG1>,
}

// SAFETY: `ArkG1` elements are plain curve points (projective coordinates)
// with no interior mutability. The missing auto-traits are inherited from
// arkworks type parameters, not due to any actual thread-safety hazard.
unsafe impl Send for DoryPartialCommitment {}
// SAFETY: `ArkG1` elements are plain curve points with no interior mutability.
unsafe impl Sync for DoryPartialCommitment {}

/// Converts a `jolt_field::Field`-compatible `ark_bn254::Fr` to the dory-pcs `ArkFr` wrapper.
///
/// # Safety
///
/// `ArkFr` is `#[repr(transparent)]` over `ark_bn254::Fr`, guaranteeing
/// identical memory layout.
#[inline]
pub fn jolt_fr_to_ark(f: &ark_bn254::Fr) -> ArkFr {
    // SAFETY: ArkFr is repr(transparent) over ark_bn254::Fr.
    unsafe { std::mem::transmute_copy(f) }
}

/// Converts a dory-pcs `ArkFr` back to an `ark_bn254::Fr`.
///
/// # Safety
///
/// Same layout guarantee as [`jolt_fr_to_ark`].
#[inline]
pub fn ark_to_jolt_fr(ark: &ArkFr) -> ark_bn254::Fr {
    // SAFETY: ArkFr is repr(transparent) over ark_bn254::Fr.
    unsafe { std::mem::transmute_copy(ark) }
}

/// Serializes an arkworks `CanonicalSerialize` type via serde by encoding to bytes.
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

/// Deserializes an arkworks `CanonicalDeserialize` type via serde by decoding from bytes.
fn canonical_deserialize<'de, T: CanonicalDeserialize, D: Deserializer<'de>>(
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
    use jolt_poly::{DensePolynomial, MultilinearPolynomial};
    use jolt_transcript::Transcript;
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    type Fr = ark_bn254::Fr;

    #[test]
    fn dory_commitment_serde_round_trip() {
        let num_vars = 3;
        let mut rng = ChaCha20Rng::seed_from_u64(400);

        let prover_setup = crate::DoryScheme::setup_prover(num_vars);
        let poly = DensePolynomial::<Fr>::random(num_vars, &mut rng);
        let commitment = crate::DoryScheme::commit(&poly, &prover_setup);

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

        // Verify functional equivalence: both setups produce the same verification result
        let mut rng = ChaCha20Rng::seed_from_u64(401);
        let prover_setup = crate::DoryScheme::setup_prover(num_vars);

        let poly = DensePolynomial::<Fr>::random(num_vars, &mut rng);
        let point: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
        let eval = MultilinearPolynomial::evaluate(&poly, &point);
        let commitment = crate::DoryScheme::commit(&poly, &prover_setup);

        let mut prove_transcript = jolt_transcript::Blake2bTranscript::new(b"serde-vs");
        let proof =
            crate::DoryScheme::prove(&poly, &point, eval, &prover_setup, &mut prove_transcript);

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
    fn dory_batched_proof_serde_round_trip() {
        let num_vars = 2;
        let mut rng = ChaCha20Rng::seed_from_u64(402);

        let prover_setup = crate::DoryScheme::setup_prover(num_vars);

        let poly = DensePolynomial::<Fr>::random(num_vars, &mut rng);
        let point: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
        let eval = MultilinearPolynomial::evaluate(&poly, &point);

        let mut transcript = jolt_transcript::Blake2bTranscript::new(b"serde-bp");
        let proof = crate::DoryScheme::prove(&poly, &point, eval, &prover_setup, &mut transcript);

        let batched = DoryBatchedProof(vec![proof]);

        let serialized = serde_json::to_vec(&batched).expect("serialize batched proof");
        let deserialized: DoryBatchedProof =
            serde_json::from_slice(&serialized).expect("deserialize batched proof");

        assert_eq!(batched.0.len(), deserialized.0.len());

        // Verify the deserialized proof is functionally valid
        let verifier_setup = DoryVerifierSetup(prover_setup.0.to_verifier_setup());
        let commitment = crate::DoryScheme::commit(&poly, &prover_setup);

        let mut verify_transcript = jolt_transcript::Blake2bTranscript::new(b"serde-bp");
        let result = crate::DoryScheme::verify(
            &commitment,
            &point,
            eval,
            &deserialized.0[0],
            &verifier_setup,
            &mut verify_transcript,
        );
        assert!(
            result.is_ok(),
            "deserialized batched proof must verify correctly"
        );
    }
}
