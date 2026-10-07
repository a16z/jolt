use std::io::Cursor;

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use dory::backends::arkworks::{ArkG1, ArkG2, ArkGT, ArkworksProverSetup, ArkworksVerifierSetup};
use jolt_crypto::{Bn254G1, Bn254GT, HomomorphicCommitment};
use jolt_field::{CanonicalBytes, CanonicalDecode, Fr};
use serde::{de::Error, Deserialize, Deserializer, Serialize, Serializer};

/// Bounds the rounds any supported proof can use, and so the verifier setup's
/// per-round tables. Dory runs `ceil(num_vars / 2)` rounds, so 64 covers
/// polynomials up to 2^128 evaluations.
const MAX_PROOF_ROUNDS: usize = 64;

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

/// The commitment's GT element is its transcript atom.
impl CanonicalBytes for DoryCommitment {
    const NUM_BYTES: usize = Bn254GT::NUM_BYTES;

    fn to_bytes_le(&self, out: &mut [u8]) {
        self.0.to_bytes_le(out);
    }
}

impl ::spongefish::Encoding<[u8]> for DoryCommitment {
    fn encode(&self) -> impl AsRef<[u8]> {
        ::jolt_field::narg::encode(self)
    }
}

impl CanonicalDecode for DoryCommitment {
    fn from_bytes_le_checked(bytes: &[u8]) -> Option<Self> {
        Bn254GT::from_bytes_le_checked(bytes).map(Self)
    }
}

impl ::spongefish::NargDeserialize for DoryCommitment {
    fn deserialize_from_narg(buf: &mut &[u8]) -> ::spongefish::VerificationResult<Self> {
        ::jolt_field::narg::deserialize(buf)
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

#[derive(Clone)]
pub struct DoryProverSetup(pub ArkworksProverSetup);

impl Serialize for DoryProverSetup {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        canonical_serialize(&self.0, serializer)
    }
}

impl<'de> Deserialize<'de> for DoryProverSetup {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let buf: Vec<u8> = Deserialize::deserialize(deserializer)?;
        let mut cursor = Cursor::new(&buf[..]);
        let setup =
            ArkworksProverSetup::deserialize_compressed(&mut cursor).map_err(Error::custom)?;
        if cursor.position() != buf.len() as u64 {
            return Err(Error::custom(
                "Dory prover setup encoding has trailing bytes",
            ));
        }
        Ok(Self(setup))
    }
}

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
        validate_verifier_setup_structure(&buf).map_err(Error::custom)?;
        ArkworksVerifierSetup::deserialize_compressed(&buf[..])
            .map_err(Error::custom)
            .map(Self)
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
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
/// hold `max_num_rounds + 1` entries.
const MAX_SETUP_GT_VECTOR_LEN: usize = MAX_PROOF_ROUNDS + 1;

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

#[cfg(test)]
#[expect(
    clippy::expect_used,
    clippy::unwrap_used,
    reason = "tests may panic on assertion failures"
)]
mod tests {
    use super::*;
    use jolt_field::Field;
    use jolt_openings::CommitmentScheme;
    use jolt_poly::Polynomial;
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

        let mut prove_transcript = crate::test_support::prover(b"serde-vs");
        crate::DoryScheme::open(
            &poly,
            &point,
            eval,
            &prover_setup,
            Some(hint),
            &mut prove_transcript,
        )
        .unwrap();

        let narg = prove_transcript.finish();
        let mut verify_transcript = crate::test_support::verifier(b"serde-vs", &narg);
        let result = crate::DoryScheme::verify(
            &commitment,
            &point,
            eval,
            &deserialized,
            &mut verify_transcript,
        );
        assert!(
            result.is_ok(),
            "deserialized verifier setup must verify correctly"
        );
    }

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
}
