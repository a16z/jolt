//! Wrapper types bridging dory-pcs to jolt-openings.

use std::io::Cursor;

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use dory::backends::arkworks::{
    ArkDoryProof, ArkG1, ArkGT, ArkworksProverSetup, ArkworksVerifierSetup,
};
use dory::primitives::transcript::Transcript as DoryTranscript;
use jolt_crypto::{Bn254G1, Bn254G2, Bn254GT, HomomorphicCommitment};
use jolt_field::{Fr, FromPrimitiveInt, Invertible};
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

impl HomomorphicCommitment<Fr> for DoryCommitment {
    #[inline]
    fn add(c1: &Self, c2: &Self) -> Self {
        Self(<Bn254GT as HomomorphicCommitment<Fr>>::add(&c1.0, &c2.0))
    }

    #[inline]
    fn linear_combine(c1: &Self, c2: &Self, scalar: &Fr) -> Self {
        Self(HomomorphicCommitment::linear_combine(&c1.0, &c2.0, scalar))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct DoryProof(pub ArkDoryProof);

impl Eq for DoryProof {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DoryVmvArtifacts {
    pub c: Bn254GT,
    pub d2: Bn254GT,
    pub e1: Bn254G1,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DoryZkArtifacts {
    pub e2: Option<Bn254G2>,
    pub y_com: Option<Bn254G1>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DoryScalarProductProofArtifacts {
    pub p1: Bn254GT,
    pub p2: Bn254GT,
    pub q: Bn254GT,
    pub r: Bn254GT,
    pub e1: Bn254G1,
    pub e2: Bn254G2,
    pub r1: Fr,
    pub r2: Fr,
    pub r3: Fr,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DoryVerifierTranscriptScalars {
    pub reduce_rounds: Vec<DoryReduceRoundTranscriptScalars>,
    pub gamma: Fr,
    pub gamma_inverse: Fr,
    pub scalar_product_sigma_c: Option<Fr>,
    pub d: Fr,
    pub d_inverse: Fr,
    pub d_squared: Fr,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DoryReduceRoundTranscriptScalars {
    pub beta: Fr,
    pub beta_inverse: Fr,
    pub alpha: Fr,
    pub alpha_inverse: Fr,
    pub alpha_beta: Fr,
    pub alpha_inverse_beta_inverse: Fr,
    pub s1_fold_factor: Fr,
    pub s2_fold_factor: Fr,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DoryReduceRoundArtifacts {
    pub first: DoryFirstReduceArtifacts,
    pub second: DorySecondReduceArtifacts,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DoryFirstReduceArtifacts {
    pub d1_left: Bn254GT,
    pub d1_right: Bn254GT,
    pub d2_left: Bn254GT,
    pub d2_right: Bn254GT,
    pub e1_beta: Bn254G1,
    pub e2_beta: Bn254G2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DorySecondReduceArtifacts {
    pub c_plus: Bn254GT,
    pub c_minus: Bn254GT,
    pub e1_plus: Bn254G1,
    pub e1_minus: Bn254G1,
    pub e2_plus: Bn254G2,
    pub e2_minus: Bn254G2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DoryFinalArtifacts {
    pub e1: Bn254G1,
    pub e2: Bn254G2,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DoryVerifierSetupArtifacts {
    pub chi: Vec<Bn254GT>,
    pub delta_1l: Vec<Bn254GT>,
    pub delta_1r: Vec<Bn254GT>,
    pub delta_2l: Vec<Bn254GT>,
    pub delta_2r: Vec<Bn254GT>,
    pub g1_0: Bn254G1,
    pub g2_0: Bn254G2,
    pub h1: Bn254G1,
    pub h2: Bn254G2,
    pub ht: Bn254GT,
}

impl DoryProof {
    pub fn point_len(&self) -> usize {
        self.0.nu + self.0.sigma
    }

    pub fn reduce_round_count(&self) -> usize {
        self.0.sigma
    }

    pub fn first_reduce_message_count(&self) -> usize {
        self.0.first_messages.len()
    }

    pub fn second_reduce_message_count(&self) -> usize {
        self.0.second_messages.len()
    }

    pub fn has_canonical_reduce_round_shape(&self) -> bool {
        self.first_reduce_message_count() == self.reduce_round_count()
            && self.second_reduce_message_count() == self.reduce_round_count()
    }

    pub fn vmv_artifacts(&self) -> DoryVmvArtifacts {
        DoryVmvArtifacts {
            c: crate::scheme::ark_to_jolt_gt(&self.0.vmv_message.c),
            d2: crate::scheme::ark_to_jolt_gt(&self.0.vmv_message.d2),
            e1: crate::scheme::ark_to_jolt_g1(self.0.vmv_message.e1),
        }
    }

    pub fn zk_artifacts(&self) -> DoryZkArtifacts {
        DoryZkArtifacts {
            e2: self
                .0
                .e2
                .as_ref()
                .map(|e2| crate::scheme::ark_to_jolt_g2(*e2)),
            y_com: self
                .0
                .y_com
                .as_ref()
                .map(|y_com| crate::scheme::ark_to_jolt_g1(*y_com)),
        }
    }

    pub fn has_transparent_opening_artifacts(&self) -> bool {
        self.0.e2.is_none()
            && self.0.y_com.is_none()
            && self.0.sigma1_proof.is_none()
            && self.0.sigma2_proof.is_none()
            && self.0.scalar_product_proof.is_none()
    }

    pub fn has_zk_opening_artifacts(&self) -> bool {
        self.0.e2.is_some()
            && self.0.y_com.is_some()
            && self.0.sigma1_proof.is_some()
            && self.0.sigma2_proof.is_some()
            && self.0.scalar_product_proof.is_some()
    }

    pub fn scalar_product_artifacts(&self) -> Option<DoryScalarProductProofArtifacts> {
        self.0
            .scalar_product_proof
            .as_ref()
            .map(|scalar_product| DoryScalarProductProofArtifacts {
                p1: crate::scheme::ark_to_jolt_gt(&scalar_product.p1),
                p2: crate::scheme::ark_to_jolt_gt(&scalar_product.p2),
                q: crate::scheme::ark_to_jolt_gt(&scalar_product.q),
                r: crate::scheme::ark_to_jolt_gt(&scalar_product.r),
                e1: crate::scheme::ark_to_jolt_g1(scalar_product.e1),
                e2: crate::scheme::ark_to_jolt_g2(scalar_product.e2),
                r1: crate::scheme::ark_to_jolt_fr(&scalar_product.r1),
                r2: crate::scheme::ark_to_jolt_fr(&scalar_product.r2),
                r3: crate::scheme::ark_to_jolt_fr(&scalar_product.r3),
            })
    }

    pub fn reduce_round_artifacts(&self) -> Vec<DoryReduceRoundArtifacts> {
        self.0
            .first_messages
            .iter()
            .zip(&self.0.second_messages)
            .map(|(first, second)| DoryReduceRoundArtifacts {
                first: DoryFirstReduceArtifacts {
                    d1_left: crate::scheme::ark_to_jolt_gt(&first.d1_left),
                    d1_right: crate::scheme::ark_to_jolt_gt(&first.d1_right),
                    d2_left: crate::scheme::ark_to_jolt_gt(&first.d2_left),
                    d2_right: crate::scheme::ark_to_jolt_gt(&first.d2_right),
                    e1_beta: crate::scheme::ark_to_jolt_g1(first.e1_beta),
                    e2_beta: crate::scheme::ark_to_jolt_g2(first.e2_beta),
                },
                second: DorySecondReduceArtifacts {
                    c_plus: crate::scheme::ark_to_jolt_gt(&second.c_plus),
                    c_minus: crate::scheme::ark_to_jolt_gt(&second.c_minus),
                    e1_plus: crate::scheme::ark_to_jolt_g1(second.e1_plus),
                    e1_minus: crate::scheme::ark_to_jolt_g1(second.e1_minus),
                    e2_plus: crate::scheme::ark_to_jolt_g2(second.e2_plus),
                    e2_minus: crate::scheme::ark_to_jolt_g2(second.e2_minus),
                },
            })
            .collect()
    }

    pub fn final_artifacts(&self) -> DoryFinalArtifacts {
        DoryFinalArtifacts {
            e1: crate::scheme::ark_to_jolt_g1(self.0.final_message.e1),
            e2: crate::scheme::ark_to_jolt_g2(self.0.final_message.e2),
        }
    }

    pub fn verifier_transcript_scalars<T>(
        &self,
        transcript: &T,
        point: &[Fr],
    ) -> DoryVerifierTranscriptScalars
    where
        T: Transcript<Challenge = Fr> + Clone,
    {
        let mut fork = (*transcript).clone();
        let mut dory_transcript = crate::transcript::JoltToDoryTranscript::new(&mut fork);
        let proof = &self.0;
        let dory_point = point.iter().rev().copied().collect::<Vec<_>>();
        let s1_coords = &dory_point[..proof.sigma.min(dory_point.len())];
        let s2_point_end = self.point_len().min(dory_point.len());
        let s2_raw = if proof.sigma <= s2_point_end {
            &dory_point[proof.sigma..s2_point_end]
        } else {
            &[]
        };

        dory_transcript.append_serde(b"vmv_c", &proof.vmv_message.c);
        dory_transcript.append_serde(b"vmv_d2", &proof.vmv_message.d2);
        dory_transcript.append_serde(b"vmv_e1", &proof.vmv_message.e1);

        if let (Some(e2), Some(y_com)) = (&proof.e2, &proof.y_com) {
            dory_transcript.append_serde(b"vmv_e2", e2);
            dory_transcript.append_serde(b"vmv_y_com", y_com);
            if let Some(sigma1) = &proof.sigma1_proof {
                dory_transcript.append_serde(b"sigma1_a1", &sigma1.a1);
                dory_transcript.append_serde(b"sigma1_a2", &sigma1.a2);
                let _ = dory_transcript.challenge_scalar(b"sigma1_c");
            }
            if let Some(sigma2) = &proof.sigma2_proof {
                dory_transcript.append_serde(b"sigma2_a", &sigma2.a);
                let _ = dory_transcript.challenge_scalar(b"sigma2_c");
            }
        }

        let mut reduce_rounds = Vec::with_capacity(proof.first_messages.len());
        for (first_msg, second_msg) in proof.first_messages.iter().zip(&proof.second_messages) {
            dory_transcript.append_serde(b"d1_left", &first_msg.d1_left);
            dory_transcript.append_serde(b"d1_right", &first_msg.d1_right);
            dory_transcript.append_serde(b"d2_left", &first_msg.d2_left);
            dory_transcript.append_serde(b"d2_right", &first_msg.d2_right);
            dory_transcript.append_serde(b"e1_beta", &first_msg.e1_beta);
            dory_transcript.append_serde(b"e2_beta", &first_msg.e2_beta);
            let beta = crate::scheme::ark_to_jolt_fr(&dory_transcript.challenge_scalar(b"beta"));
            let beta_inverse = beta.inverse().unwrap_or_default();

            dory_transcript.append_serde(b"c_plus", &second_msg.c_plus);
            dory_transcript.append_serde(b"c_minus", &second_msg.c_minus);
            dory_transcript.append_serde(b"e1_plus", &second_msg.e1_plus);
            dory_transcript.append_serde(b"e1_minus", &second_msg.e1_minus);
            dory_transcript.append_serde(b"e2_plus", &second_msg.e2_plus);
            dory_transcript.append_serde(b"e2_minus", &second_msg.e2_minus);
            let alpha = crate::scheme::ark_to_jolt_fr(&dory_transcript.challenge_scalar(b"alpha"));
            let alpha_inverse = alpha.inverse().unwrap_or_default();
            let alpha_beta = alpha * beta;
            let alpha_inverse_beta_inverse = alpha_inverse * beta_inverse;
            let coordinate_index = proof.sigma.saturating_sub(reduce_rounds.len() + 1);
            let s1_coord = s1_coords.get(coordinate_index).copied().unwrap_or_default();
            let s2_coord = s2_raw.get(coordinate_index).copied().unwrap_or_default();
            let one = Fr::from_u64(1);
            let s1_fold_factor = alpha * (one - s1_coord) + s1_coord;
            let s2_fold_factor = alpha_inverse * (one - s2_coord) + s2_coord;

            reduce_rounds.push(DoryReduceRoundTranscriptScalars {
                beta,
                beta_inverse,
                alpha,
                alpha_inverse,
                alpha_beta,
                alpha_inverse_beta_inverse,
                s1_fold_factor,
                s2_fold_factor,
            });
        }

        let gamma = crate::scheme::ark_to_jolt_fr(&dory_transcript.challenge_scalar(b"gamma"));
        let gamma_inverse = gamma.inverse().unwrap_or_default();

        let scalar_product_sigma_c = if let Some(scalar_product) = &proof.scalar_product_proof {
            dory_transcript.append_serde(b"sigma_p1", &scalar_product.p1);
            dory_transcript.append_serde(b"sigma_p2", &scalar_product.p2);
            dory_transcript.append_serde(b"sigma_q", &scalar_product.q);
            dory_transcript.append_serde(b"sigma_r", &scalar_product.r);
            Some(crate::scheme::ark_to_jolt_fr(
                &dory_transcript.challenge_scalar(b"sigma_c"),
            ))
        } else {
            None
        };

        dory_transcript.append_serde(b"final_e1", &proof.final_message.e1);
        dory_transcript.append_serde(b"final_e2", &proof.final_message.e2);
        let d = crate::scheme::ark_to_jolt_fr(&dory_transcript.challenge_scalar(b"d"));
        let d_inverse = d.inverse().unwrap_or_default();
        let d_squared = d * d;

        DoryVerifierTranscriptScalars {
            reduce_rounds,
            gamma,
            gamma_inverse,
            scalar_product_sigma_c,
            d,
            d_inverse,
            d_squared,
        }
    }
}

impl DoryVerifierTranscriptScalars {
    pub fn has_valid_inverse_relations(&self) -> bool {
        let one = Fr::from_u64(1);
        self.reduce_rounds.iter().all(|round| {
            round.beta * round.beta_inverse == one
                && round.alpha * round.alpha_inverse == one
                && round.alpha_beta == round.alpha * round.beta
                && round.alpha_inverse_beta_inverse == round.alpha_inverse * round.beta_inverse
        }) && self.gamma * self.gamma_inverse == one
            && self.d * self.d_inverse == one
            && self.d_squared == self.d * self.d
    }

    pub fn has_valid_replay_relations_for_point(&self, point: &[Fr]) -> bool {
        if !self.has_valid_inverse_relations() {
            return false;
        }

        let sigma = self.reduce_rounds.len();
        let dory_point = point.iter().rev().copied().collect::<Vec<_>>();
        let s1_coords = &dory_point[..sigma.min(dory_point.len())];
        let s2_raw = if sigma <= dory_point.len() {
            &dory_point[sigma..]
        } else {
            &[]
        };
        let one = Fr::from_u64(1);

        self.reduce_rounds
            .iter()
            .enumerate()
            .all(|(round_index, round)| {
                let coordinate_index = sigma.saturating_sub(round_index + 1);
                let s1_coord = s1_coords.get(coordinate_index).copied().unwrap_or_default();
                let s2_coord = s2_raw.get(coordinate_index).copied().unwrap_or_default();
                round.s1_fold_factor == round.alpha * (one - s1_coord) + s1_coord
                    && round.s2_fold_factor == round.alpha_inverse * (one - s2_coord) + s2_coord
            })
    }
}

impl DoryVerifierSetup {
    pub fn max_reduce_rounds(&self) -> usize {
        self.0.chi.len().saturating_sub(1)
    }

    pub fn has_consistent_artifact_lengths(&self) -> bool {
        let expected_len = self.0.chi.len();
        expected_len > 0
            && self.0.delta_1l.len() == expected_len
            && self.0.delta_1r.len() == expected_len
            && self.0.delta_2l.len() == expected_len
            && self.0.delta_2r.len() == expected_len
    }

    pub fn supports_reduce_round_count(&self, reduce_rounds: usize) -> bool {
        self.has_consistent_artifact_lengths() && reduce_rounds <= self.max_reduce_rounds()
    }

    pub fn artifacts(&self) -> DoryVerifierSetupArtifacts {
        DoryVerifierSetupArtifacts {
            chi: self
                .0
                .chi
                .iter()
                .map(crate::scheme::ark_to_jolt_gt)
                .collect(),
            delta_1l: self
                .0
                .delta_1l
                .iter()
                .map(crate::scheme::ark_to_jolt_gt)
                .collect(),
            delta_1r: self
                .0
                .delta_1r
                .iter()
                .map(crate::scheme::ark_to_jolt_gt)
                .collect(),
            delta_2l: self
                .0
                .delta_2l
                .iter()
                .map(crate::scheme::ark_to_jolt_gt)
                .collect(),
            delta_2r: self
                .0
                .delta_2r
                .iter()
                .map(crate::scheme::ark_to_jolt_gt)
                .collect(),
            g1_0: crate::scheme::ark_to_jolt_g1(self.0.g1_0),
            g2_0: crate::scheme::ark_to_jolt_g2(self.0.g2_0),
            h1: crate::scheme::ark_to_jolt_g1(self.0.h1),
            h2: crate::scheme::ark_to_jolt_g2(self.0.h2),
            ht: crate::scheme::ark_to_jolt_gt(&self.0.ht),
        }
    }
}

impl Serialize for DoryProof {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        canonical_serialize(&self.0, serializer)
    }
}

impl AppendToTranscript for DoryProof {
    fn append_to_transcript<T: Transcript>(&self, transcript: &mut T) {
        append_canonical_to_transcript(&self.0, transcript);
    }

    fn transcript_payload_len(&self) -> Option<u64> {
        Some(self.0.compressed_size() as u64)
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

impl AppendToTranscript for DoryVerifierSetup {
    fn append_to_transcript<T: Transcript>(&self, transcript: &mut T) {
        append_canonical_to_transcript(&self.0, transcript);
    }

    fn transcript_payload_len(&self) -> Option<u64> {
        Some(self.0.compressed_size() as u64)
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

#[expect(clippy::expect_used, reason = "serialization into Vec cannot fail")]
fn append_canonical_to_transcript<V: CanonicalSerialize, T: Transcript>(
    value: &V,
    transcript: &mut T,
) {
    let mut buf = Vec::with_capacity(value.compressed_size());
    value
        .serialize_compressed(&mut buf)
        .expect("Dory canonical serialization cannot fail");
    transcript.append_bytes(&buf);
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
    use jolt_openings::{CommitmentScheme, ZkOpeningScheme};
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
    fn dory_verifier_setup_reports_reduce_round_capacity() {
        let mut verifier_setup = crate::DoryScheme::setup_verifier(4);

        assert!(verifier_setup.has_consistent_artifact_lengths());
        assert_eq!(verifier_setup.max_reduce_rounds(), 2);
        assert!(verifier_setup.supports_reduce_round_count(2));
        assert!(!verifier_setup.supports_reduce_round_count(3));

        let _ = verifier_setup.0.delta_1l.pop();

        assert!(!verifier_setup.has_consistent_artifact_lengths());
        assert!(!verifier_setup.supports_reduce_round_count(1));
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
    fn verifier_transcript_scalars_replay_dory_challenge_schedule() {
        let num_vars = 4;
        let mut rng = ChaCha20Rng::seed_from_u64(404);
        let prover_setup = crate::DoryScheme::setup_prover(num_vars);
        let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
        let point: Vec<Fr> = (0..num_vars)
            .map(|_| <Fr as RandomSampling>::random(&mut rng))
            .collect();
        let eval = poly.evaluate(&point);
        let (commitment, hint) = crate::DoryScheme::commit(poly.evaluations(), &prover_setup);

        let mut prove_transcript = jolt_transcript::Blake2bTranscript::new(b"scalar-replay");
        let proof = crate::DoryScheme::open(
            &poly,
            &point,
            eval,
            &prover_setup,
            Some(hint),
            &mut prove_transcript,
        );
        let verifier_setup = DoryVerifierSetup(prover_setup.0.to_verifier_setup());
        let mut verify_transcript = jolt_transcript::Blake2bTranscript::new(b"scalar-replay");
        crate::DoryScheme::verify(
            &commitment,
            &point,
            eval,
            &proof,
            &verifier_setup,
            &mut verify_transcript,
        )
        .expect("proof verifies with matching transcript");

        let transcript = jolt_transcript::Blake2bTranscript::<Fr>::new(b"scalar-replay");
        let before = transcript.state();
        let scalars = proof.verifier_transcript_scalars(&transcript, &point);
        let again = proof.verifier_transcript_scalars(&transcript, &point);

        assert_eq!(transcript.state(), before);
        assert_eq!(scalars, again);
        assert_eq!(scalars.reduce_rounds.len(), proof.reduce_round_count());
        assert!(scalars.scalar_product_sigma_c.is_none());
        assert!(scalars.has_valid_inverse_relations());
        assert!(scalars.has_valid_replay_relations_for_point(&point));

        let one = Fr::from_u64(1);
        let first_round = scalars.reduce_rounds[0];
        assert_eq!(
            first_round.s1_fold_factor,
            first_round.alpha * (one - point[2]) + point[2]
        );
        assert_eq!(
            first_round.s2_fold_factor,
            first_round.alpha_inverse * (one - point[0]) + point[0]
        );

        let changed = proof.verifier_transcript_scalars(
            &jolt_transcript::Blake2bTranscript::<Fr>::new(b"scalar-domain"),
            &point,
        );
        assert_ne!(scalars.d, changed.d);
    }

    #[test]
    fn verifier_transcript_scalars_detect_invalid_inverse_relations() {
        let num_vars = 4;
        let mut rng = ChaCha20Rng::seed_from_u64(406);
        let prover_setup = crate::DoryScheme::setup_prover(num_vars);
        let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
        let point: Vec<Fr> = (0..num_vars)
            .map(|_| <Fr as RandomSampling>::random(&mut rng))
            .collect();
        let eval = poly.evaluate(&point);
        let (_commitment, hint) = crate::DoryScheme::commit(poly.evaluations(), &prover_setup);

        let mut transcript = jolt_transcript::Blake2bTranscript::new(b"scalar-inverses");
        let proof = crate::DoryScheme::open(
            &poly,
            &point,
            eval,
            &prover_setup,
            Some(hint),
            &mut transcript,
        );
        let replay_transcript = jolt_transcript::Blake2bTranscript::<Fr>::new(b"scalar-inverses");
        let mut scalars = proof.verifier_transcript_scalars(&replay_transcript, &point);
        assert!(scalars.has_valid_inverse_relations());
        assert!(scalars.has_valid_replay_relations_for_point(&point));

        scalars.reduce_rounds[0].beta_inverse = Fr::from_u64(0);
        assert!(!scalars.has_valid_inverse_relations());
        assert!(!scalars.has_valid_replay_relations_for_point(&point));

        let mut scalars = proof.verifier_transcript_scalars(&replay_transcript, &point);
        scalars.d_inverse = Fr::from_u64(0);
        assert!(!scalars.has_valid_inverse_relations());

        let mut scalars = proof.verifier_transcript_scalars(&replay_transcript, &point);
        scalars.reduce_rounds[0].alpha_beta += Fr::from_u64(1);
        assert!(!scalars.has_valid_inverse_relations());

        let mut scalars = proof.verifier_transcript_scalars(&replay_transcript, &point);
        scalars.reduce_rounds[0].s1_fold_factor += Fr::from_u64(1);
        assert!(scalars.has_valid_inverse_relations());
        assert!(!scalars.has_valid_replay_relations_for_point(&point));
    }

    #[test]
    fn zk_proof_exposes_scalar_product_artifacts() {
        let num_vars = 4;
        let mut rng = ChaCha20Rng::seed_from_u64(405);
        let prover_setup = crate::DoryScheme::setup_prover(num_vars);
        let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
        let point: Vec<Fr> = (0..num_vars)
            .map(|_| <Fr as RandomSampling>::random(&mut rng))
            .collect();
        let eval = poly.evaluate(&point);
        let (_commitment, hint) =
            <crate::DoryScheme as ZkOpeningScheme>::commit_zk(poly.evaluations(), &prover_setup);

        let mut transcript = jolt_transcript::Blake2bTranscript::new(b"zk-artifacts");
        let (proof, _hiding_commitment, _blind) =
            crate::DoryScheme::open_zk(&poly, &point, eval, &prover_setup, hint, &mut transcript);

        assert!(proof.has_zk_opening_artifacts());
        assert!(!proof.has_transparent_opening_artifacts());
        assert!(proof.zk_artifacts().e2.is_some());
        assert!(proof.zk_artifacts().y_com.is_some());
        assert!(proof.scalar_product_artifacts().is_some());
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
