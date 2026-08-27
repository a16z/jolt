//! Wrapper types bridging dory-pcs to jolt-openings.

use std::io::Cursor;

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use dory::backends::arkworks::{
    ArkDoryProof, ArkG1, ArkG2, ArkGT, ArkworksProverSetup, ArkworksVerifierSetup,
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

/// Byte-size cap on a serialized proof, checked before any parsing. A
/// well-formed proof is ~4.7 KiB per round (12 group elements, GT-dominated)
/// plus a small fixed prefix/suffix, so `MAX_SERIALIZED_PROOF_ROUNDS` rounds
/// stay well under 512 KiB.
pub const MAX_SERIALIZED_PROOF_BYTES: usize = 512 * 1024;

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

impl<F: jolt_field::JoltField> HomomorphicCommitment<F> for DoryCommitment {
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
        if buf.len() > MAX_SERIALIZED_PROOF_BYTES {
            return Err(serde::de::Error::custom(format!(
                "Dory proof ({} bytes) exceeds maximum ({MAX_SERIALIZED_PROOF_BYTES})",
                buf.len()
            )));
        }
        validate_proof_round_count(&buf).map_err(serde::de::Error::custom)?;
        let mut cursor = Cursor::new(&buf[..]);
        let proof =
            ArkDoryProof::deserialize_compressed(&mut cursor).map_err(serde::de::Error::custom)?;
        // Canonical encoding: a valid parse must consume the entire buffer.
        if cursor.position() != buf.len() as u64 {
            return Err(serde::de::Error::custom(
                "Dory proof encoding has trailing bytes",
            ));
        }
        Ok(Self(proof))
    }
}

/// Affine view of the full setup `g1_vec`, shared by every proof over one
/// setup. Empty when `JOLT_DORY_SETUP_PREP=0`.
pub(crate) type AffineG1Table = std::sync::Arc<Vec<ark_bn254::G1Affine>>;

/// The prover SRS plus its prove-path tables — the Miller-prepared G2 table
/// and the affine G1 bases — built eagerly by `DoryScheme::setup_prover`
/// (both empty under `JOLT_DORY_SETUP_PREP=0`). Owning the tables on the
/// setup object — one setup = one URS — makes prefix-borrowing sound by
/// construction, unlike dory-pcs's global prepared-point cache (see
/// `DoryScheme::setup_prover`).
#[derive(Clone)]
pub struct DoryProverSetup(
    pub ArkworksProverSetup,
    pub(crate) crate::tier2::PreparedG2Table,
    pub(crate) AffineG1Table,
);

#[derive(Clone)]
pub struct DoryVerifierSetup(pub ArkworksVerifierSetup);

impl Serialize for DoryVerifierSetup {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        canonical_serialize(&self.0, serializer)
    }
}

impl<'de> Deserialize<'de> for DoryVerifierSetup {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let buf: Vec<u8> = Deserialize::deserialize(deserializer)?;
        validate_verifier_setup_structure(&buf).map_err(serde::de::Error::custom)?;
        ArkworksVerifierSetup::deserialize_compressed(&buf[..])
            .map_err(serde::de::Error::custom)
            .map(Self)
    }
}

/// Commit-time auxiliary data reused when opening: the tier-1 row
/// commitments and the tier-2 blind. Fields are public for the
/// `combine_hints` device hook (`hint_hook`), which recombines rows outside
/// this crate.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DoryHint {
    pub row_commitments: Vec<Bn254G1>,
    pub commit_blind: Fr,
}

impl DoryHint {
    pub fn new(row_commitments: Vec<Bn254G1>, commit_blind: Fr) -> Self {
        Self {
            row_commitments,
            commit_blind,
        }
    }
}

#[derive(Clone)]
pub struct DoryPartialCommitment {
    pub row_commitments: Vec<Bn254G1>,
    /// Affine SRS bases cached lazily for the primitive-typed feed paths
    /// (`feed_u64`/`feed_i128`), which call arkworks `msm_u64`/`msm_i128`
    /// against affine bases. Grown on demand to the widest fed row.
    pub(crate) scalar_affine_bases: Option<Vec<ark_bn254::G1Affine>>,
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

/// Caps each GT vector in a serialized verifier setup. The delta/chi tables
/// hold `max_num_rounds + 1` entries, and `MAX_SERIALIZED_PROOF_ROUNDS`
/// bounds the rounds any supported proof can use.
const MAX_SETUP_GT_VECTOR_LEN: usize = MAX_SERIALIZED_PROOF_ROUNDS + 1;

/// Pre-validates a serialized `ArkworksVerifierSetup` before delegating to
/// the upstream parser, whose `Vec<T>` deserialization reads a u64 length
/// prefix and calls `Vec::with_capacity(len)` before reading any element —
/// an attacker-supplied length near `u64::MAX` would abort or OOM.
///
/// Wire layout (dory-pcs `derive(DorySerialize)` on `VerifierSetup`, fields
/// in declaration order): five u64-length-prefixed `Vec<GT>` (`delta_1l`,
/// `delta_1r`, `delta_2l`, `delta_2r`, `chi`), then fixed-size `g1_0`,
/// `g2_0`, `h1`, `h2`, `ht`, and `max_log_n` as u64. All group encodings are
/// fixed-width, so the whole structure can be measured without allocating.
fn validate_verifier_setup_structure(buf: &[u8]) -> Result<(), String> {
    // All three encodings are fixed-width; measure via placeholder values.
    let gt_size = ArkGT(Default::default()).compressed_size();
    let g1_size = ArkG1::default().compressed_size();
    let g2_size = ArkG2::default().compressed_size();

    let mut offset = 0usize;
    for field in ["delta_1l", "delta_1r", "delta_2l", "delta_2r", "chi"] {
        let len_bytes: [u8; 8] = buf
            .get(offset..offset + 8)
            .and_then(|b| b.try_into().ok())
            .ok_or_else(|| format!("truncated Dory verifier setup: missing {field} length"))?;
        let len = u64::from_le_bytes(len_bytes);
        if len > MAX_SETUP_GT_VECTOR_LEN as u64 {
            return Err(format!(
                "Dory verifier setup {field} length ({len}) exceeds maximum ({MAX_SETUP_GT_VECTOR_LEN})"
            ));
        }
        // len <= 65 and gt_size is a few hundred bytes: no overflow.
        offset += 8 + (len as usize) * gt_size;
    }

    let fixed_tail = 2 * g1_size + 2 * g2_size + gt_size + 8;
    let expected = offset.saturating_add(fixed_tail);
    if buf.len() != expected {
        return Err(format!(
            "Dory verifier setup length mismatch: expected {expected} bytes, got {}",
            buf.len()
        ));
    }
    Ok(())
}

/// Pre-validates the round count from the proof's wire bytes before invoking
/// the upstream `CanonicalDeserialize`, which calls `Vec::with_capacity(num_rounds)`
/// and would OOM on attacker-supplied lengths near `u32::MAX`.
///
/// The prefix elements are parsed with `Validate::No`: this scan only needs
/// their wire width to locate the round count, and the real parse that
/// follows re-reads them with full (expensive, for GT) subgroup validation.
fn validate_proof_round_count(buf: &[u8]) -> Result<(), String> {
    use ark_serialize::{Compress, Validate};
    let mut cursor = Cursor::new(buf);
    let _: ArkGT =
        CanonicalDeserialize::deserialize_with_mode(&mut cursor, Compress::Yes, Validate::No)
            .map_err(|e| format!("invalid Dory proof VMV.c: {e}"))?;
    let _: ArkGT =
        CanonicalDeserialize::deserialize_with_mode(&mut cursor, Compress::Yes, Validate::No)
            .map_err(|e| format!("invalid Dory proof VMV.d2: {e}"))?;
    let _: ArkG1 =
        CanonicalDeserialize::deserialize_with_mode(&mut cursor, Compress::Yes, Validate::No)
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
#[expect(
    clippy::expect_used,
    clippy::unwrap_used,
    reason = "tests may panic on assertion failures"
)]
#[expect(clippy::indexing_slicing, reason = "tests index fixture data")]
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
        let (commitment, _) = crate::DoryScheme::commit(poly.evaluations(), &prover_setup).unwrap();

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
        let (commitment, hint) =
            crate::DoryScheme::commit(poly.evaluations(), &prover_setup).unwrap();

        let mut prove_transcript = jolt_transcript::Blake2bTranscript::new(b"serde-vs");
        let proof = crate::DoryScheme::open(
            &poly,
            &point,
            eval,
            &prover_setup,
            Some(hint),
            &mut prove_transcript,
        )
        .unwrap();

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

    /// Wraps `bytes` in the outer serde byte layer and asserts `T`'s
    /// deserializer rejects them with `needle` in the error message.
    fn assert_rejected_with<T: for<'de> Deserialize<'de>>(bytes: &[u8], needle: &str) {
        let encoded = serde_json::to_vec(&bytes).expect("encode crafted bytes");
        let err = serde_json::from_slice::<T>(&encoded)
            .err()
            .expect("malformed input must be rejected");
        assert!(err.to_string().contains(needle), "{err}");
    }

    #[test]
    fn dory_verifier_setup_rejects_huge_vector_length_prefix() {
        // A crafted length prefix must be rejected before the upstream parser
        // calls Vec::with_capacity(len) on it.
        assert_rejected_with::<DoryVerifierSetup>(&u64::MAX.to_le_bytes(), "exceeds maximum");
    }

    #[test]
    fn dory_verifier_setup_rejects_truncated_buffer() {
        assert_rejected_with::<DoryVerifierSetup>(&[0u8; 4], "truncated");
    }

    #[test]
    fn dory_verifier_setup_rejects_trailing_bytes() {
        let verifier_setup = crate::DoryScheme::setup_verifier(2);
        let mut bytes = Vec::new();
        verifier_setup
            .0
            .serialize_compressed(&mut bytes)
            .expect("serialize verifier setup");
        bytes.push(0);
        assert_rejected_with::<DoryVerifierSetup>(&bytes, "length mismatch");
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
            crate::DoryScheme::open(&poly, &point, eval, &prover_setup, None, &mut transcript)
                .unwrap();

        let serialized = serde_json::to_vec(&proof).expect("serialize proof");
        let deserialized: DoryProof =
            serde_json::from_slice(&serialized).expect("deserialize proof");

        let verifier_setup = DoryVerifierSetup(prover_setup.0.to_verifier_setup());
        let (commitment, _) = crate::DoryScheme::commit(poly.evaluations(), &prover_setup).unwrap();

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
    fn dory_proof_rejects_oversized_buffer() {
        let bytes = vec![0u8; MAX_SERIALIZED_PROOF_BYTES + 1];
        assert_rejected_with::<DoryProof>(&bytes, "exceeds maximum");
    }

    #[test]
    fn dory_proof_rejects_trailing_bytes() {
        let num_vars = 2;
        let mut rng = ChaCha20Rng::seed_from_u64(404);

        let prover_setup = crate::DoryScheme::setup_prover(num_vars);
        let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
        let point: Vec<Fr> = (0..num_vars)
            .map(|_| <Fr as Field>::random(&mut rng))
            .collect();
        let eval = poly.evaluate(&point);

        let mut transcript = jolt_transcript::Blake2bTranscript::new(b"serde-trailing");
        let proof =
            crate::DoryScheme::open(&poly, &point, eval, &prover_setup, None, &mut transcript)
                .unwrap();

        let mut bytes = Vec::new();
        proof
            .0
            .serialize_compressed(&mut bytes)
            .expect("serialize proof");
        bytes.push(0);
        assert_rejected_with::<DoryProof>(&bytes, "trailing bytes");
    }

    #[test]
    fn dory_proof_rejects_oversized_round_count() {
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
            crate::DoryScheme::open(&poly, &point, eval, &prover_setup, None, &mut transcript)
                .unwrap();

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
