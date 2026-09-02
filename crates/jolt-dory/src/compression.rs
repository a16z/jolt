use std::fmt::{Display, Formatter};

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use dory::backends::arkworks::{ArkDoryProof, ArkFr, ArkG1, ArkG2, ArkGT};
use dory::{
    FirstReduceMessage, ScalarProductMessage, ScalarProductProof, SecondReduceMessage, Sigma1Proof,
    Sigma2Proof, VMVMessage,
};
use jolt_crypto::{
    compress_gt, decompress_gt, CompressedBn254GT, GtCompressionError, COMPRESSED_GT_SIZE,
};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::scheme::{ark_to_jolt_gt, jolt_gt_to_ark};
use crate::types::{MAX_SERIALIZED_PROOF_BYTES, MAX_SERIALIZED_PROOF_ROUNDS};
use crate::{DoryCommitment, DoryProof};

const G1_SIZE: usize = 32;
const G2_SIZE: usize = 64;
const FR_SIZE: usize = 32;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DoryCompressionError {
    Gt(GtCompressionError),
    InvalidGroupEncoding,
    InvalidProofShape,
    InvalidOptionTag,
    RoundCountMismatch,
    TooManyRounds,
    ProofTooLarge,
    Truncated,
    TrailingBytes,
    IntegerOverflow,
}

impl Display for DoryCompressionError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Gt(error) => Display::fmt(error, f),
            Self::InvalidGroupEncoding => f.write_str("invalid compressed Dory group encoding"),
            Self::InvalidProofShape => f.write_str("invalid compressed Dory proof shape"),
            Self::InvalidOptionTag => f.write_str("invalid compressed Dory option tag"),
            Self::RoundCountMismatch => {
                f.write_str("compressed Dory proof round-vector lengths differ")
            }
            Self::TooManyRounds => f.write_str("compressed Dory proof has too many rounds"),
            Self::ProofTooLarge => f.write_str("compressed Dory proof exceeds the size limit"),
            Self::Truncated => f.write_str("truncated compressed Dory proof"),
            Self::TrailingBytes => f.write_str("compressed Dory proof has trailing bytes"),
            Self::IntegerOverflow => f.write_str("compressed Dory proof integer does not fit u32"),
        }
    }
}

impl std::error::Error for DoryCompressionError {}

impl From<GtCompressionError> for DoryCompressionError {
    fn from(error: GtCompressionError) -> Self {
        Self::Gt(error)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompressedDoryProof(DoryProof);

impl CompressedDoryProof {
    pub fn from_native(proof: &DoryProof) -> Result<Self, DoryCompressionError> {
        validate_proof_shape(&proof.0)?;
        Ok(Self(proof.clone()))
    }

    #[must_use]
    pub const fn as_native(&self) -> &DoryProof {
        &self.0
    }

    #[must_use]
    pub fn into_native(self) -> DoryProof {
        self.0
    }

    pub fn encoded_len(&self) -> Result<usize, DoryCompressionError> {
        encode_proof(&self.0).map(|bytes| bytes.len())
    }
}

impl Serialize for CompressedDoryProof {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let bytes = encode_proof(&self.0).map_err(serde::ser::Error::custom)?;
        serializer.serialize_bytes(&bytes)
    }
}

impl<'de> Deserialize<'de> for CompressedDoryProof {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let bytes = Vec::<u8>::deserialize(deserializer)?;
        decode_proof(&bytes)
            .map(Self)
            .map_err(serde::de::Error::custom)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CompressedDoryArtifacts {
    pub commitments: Vec<CompressedBn254GT>,
    pub proof: CompressedDoryProof,
}

impl CompressedDoryArtifacts {
    pub fn from_native(
        commitments: &[DoryCommitment],
        proof: &DoryProof,
    ) -> Result<Self, DoryCompressionError> {
        Ok(Self {
            commitments: commitments
                .iter()
                .map(|commitment| CompressedBn254GT::from_gt(&commitment.0))
                .collect(),
            proof: CompressedDoryProof::from_native(proof)?,
        })
    }

    pub fn into_native(self) -> Result<(Vec<DoryCommitment>, DoryProof), DoryCompressionError> {
        let commitments = self
            .commitments
            .iter()
            .map(CompressedBn254GT::decompress)
            .map(|result| result.map(DoryCommitment).map_err(Into::into))
            .collect::<Result<Vec<_>, DoryCompressionError>>()?;
        Ok((commitments, self.proof.into_native()))
    }
}

fn validate_proof_shape(proof: &ArkDoryProof) -> Result<(), DoryCompressionError> {
    if proof.first_messages.len() != proof.second_messages.len() {
        return Err(DoryCompressionError::RoundCountMismatch);
    }
    if proof.first_messages.len() > MAX_SERIALIZED_PROOF_ROUNDS {
        return Err(DoryCompressionError::TooManyRounds);
    }
    let _mode = proof
        .mode()
        .map_err(|_| DoryCompressionError::InvalidProofShape)?;
    Ok(())
}

fn encode_proof(proof: &DoryProof) -> Result<Vec<u8>, DoryCompressionError> {
    validate_proof_shape(&proof.0)?;
    let proof = &proof.0;
    let mut bytes = Vec::new();

    write_gt(&proof.vmv_message.c, &mut bytes);
    write_gt(&proof.vmv_message.d2, &mut bytes);
    write_group(&proof.vmv_message.e1, &mut bytes)?;
    write_u32(proof.first_messages.len(), &mut bytes)?;
    for message in &proof.first_messages {
        write_gt(&message.d1_left, &mut bytes);
        write_gt(&message.d1_right, &mut bytes);
        write_gt(&message.d2_left, &mut bytes);
        write_gt(&message.d2_right, &mut bytes);
        write_group(&message.e1_beta, &mut bytes)?;
        write_group(&message.e2_beta, &mut bytes)?;
    }
    for message in &proof.second_messages {
        write_gt(&message.c_plus, &mut bytes);
        write_gt(&message.c_minus, &mut bytes);
        write_group(&message.e1_plus, &mut bytes)?;
        write_group(&message.e1_minus, &mut bytes)?;
        write_group(&message.e2_plus, &mut bytes)?;
        write_group(&message.e2_minus, &mut bytes)?;
    }
    write_option(
        proof.final_message.as_ref(),
        &mut bytes,
        |message, bytes| {
            write_group(&message.e1, bytes)?;
            write_group(&message.e2, bytes)
        },
    )?;
    write_u32(proof.nu, &mut bytes)?;
    write_u32(proof.sigma, &mut bytes)?;
    write_option(proof.e2.as_ref(), &mut bytes, write_group)?;
    write_option(proof.y_com.as_ref(), &mut bytes, write_group)?;
    write_option(proof.sigma1_proof.as_ref(), &mut bytes, |sigma, bytes| {
        write_group(&sigma.a1, bytes)?;
        write_group(&sigma.a2, bytes)?;
        write_group(&sigma.z1, bytes)?;
        write_group(&sigma.z2, bytes)?;
        write_group(&sigma.z3, bytes)
    })?;
    write_option(proof.sigma2_proof.as_ref(), &mut bytes, |sigma, bytes| {
        write_gt(&sigma.a, bytes);
        write_group(&sigma.z1, bytes)?;
        write_group(&sigma.z2, bytes)
    })?;
    write_option(
        proof.scalar_product_proof.as_ref(),
        &mut bytes,
        |scalar, bytes| {
            write_gt(&scalar.p1, bytes);
            write_gt(&scalar.p2, bytes);
            write_gt(&scalar.q, bytes);
            write_gt(&scalar.r, bytes);
            write_group(&scalar.e1, bytes)?;
            write_group(&scalar.e2, bytes)?;
            write_group(&scalar.r1, bytes)?;
            write_group(&scalar.r2, bytes)?;
            write_group(&scalar.r3, bytes)
        },
    )?;

    if bytes.len() > MAX_SERIALIZED_PROOF_BYTES {
        return Err(DoryCompressionError::ProofTooLarge);
    }
    Ok(bytes)
}

fn decode_proof(bytes: &[u8]) -> Result<DoryProof, DoryCompressionError> {
    if bytes.len() > MAX_SERIALIZED_PROOF_BYTES {
        return Err(DoryCompressionError::ProofTooLarge);
    }
    let mut reader = Reader::new(bytes);
    let vmv_message = VMVMessage {
        c: reader.read_gt()?,
        d2: reader.read_gt()?,
        e1: reader.read_g1()?,
    };
    let num_rounds = reader.read_u32()? as usize;
    if num_rounds > MAX_SERIALIZED_PROOF_ROUNDS {
        return Err(DoryCompressionError::TooManyRounds);
    }

    let mut first_messages = Vec::with_capacity(num_rounds);
    for _ in 0..num_rounds {
        first_messages.push(FirstReduceMessage {
            d1_left: reader.read_gt()?,
            d1_right: reader.read_gt()?,
            d2_left: reader.read_gt()?,
            d2_right: reader.read_gt()?,
            e1_beta: reader.read_g1()?,
            e2_beta: reader.read_g2()?,
        });
    }
    let mut second_messages = Vec::with_capacity(num_rounds);
    for _ in 0..num_rounds {
        second_messages.push(SecondReduceMessage {
            c_plus: reader.read_gt()?,
            c_minus: reader.read_gt()?,
            e1_plus: reader.read_g1()?,
            e1_minus: reader.read_g1()?,
            e2_plus: reader.read_g2()?,
            e2_minus: reader.read_g2()?,
        });
    }

    let final_message = reader.read_option(|reader| {
        Ok(ScalarProductMessage {
            e1: reader.read_g1()?,
            e2: reader.read_g2()?,
        })
    })?;
    let nu = reader.read_u32()? as usize;
    let sigma = reader.read_u32()? as usize;
    let e2 = reader.read_option(Reader::read_g2)?;
    let y_com = reader.read_option(Reader::read_g1)?;
    let sigma1_proof = reader.read_option(|reader| {
        Ok(Sigma1Proof {
            a1: reader.read_g2()?,
            a2: reader.read_g1()?,
            z1: reader.read_fr()?,
            z2: reader.read_fr()?,
            z3: reader.read_fr()?,
        })
    })?;
    let sigma2_proof = reader.read_option(|reader| {
        Ok(Sigma2Proof {
            a: reader.read_gt()?,
            z1: reader.read_fr()?,
            z2: reader.read_fr()?,
        })
    })?;
    let scalar_product_proof = reader.read_option(|reader| {
        Ok(ScalarProductProof {
            p1: reader.read_gt()?,
            p2: reader.read_gt()?,
            q: reader.read_gt()?,
            r: reader.read_gt()?,
            e1: reader.read_g1()?,
            e2: reader.read_g2()?,
            r1: reader.read_fr()?,
            r2: reader.read_fr()?,
            r3: reader.read_fr()?,
        })
    })?;
    if !reader.is_finished() {
        return Err(DoryCompressionError::TrailingBytes);
    }

    let proof = DoryProof(ArkDoryProof {
        vmv_message,
        first_messages,
        second_messages,
        final_message,
        nu,
        sigma,
        e2,
        y_com,
        sigma1_proof,
        sigma2_proof,
        scalar_product_proof,
    });
    validate_proof_shape(&proof.0)?;
    Ok(proof)
}

fn write_gt(value: &ArkGT, bytes: &mut Vec<u8>) {
    bytes.extend_from_slice(&compress_gt(&ark_to_jolt_gt(value)));
}

fn write_group<T: CanonicalSerialize>(
    value: &T,
    bytes: &mut Vec<u8>,
) -> Result<(), DoryCompressionError> {
    value
        .serialize_compressed(bytes)
        .map_err(|_| DoryCompressionError::InvalidGroupEncoding)
}

fn write_u32(value: usize, bytes: &mut Vec<u8>) -> Result<(), DoryCompressionError> {
    let value = u32::try_from(value).map_err(|_| DoryCompressionError::IntegerOverflow)?;
    bytes.extend_from_slice(&value.to_le_bytes());
    Ok(())
}

fn write_option<T>(
    value: Option<&T>,
    bytes: &mut Vec<u8>,
    write: impl FnOnce(&T, &mut Vec<u8>) -> Result<(), DoryCompressionError>,
) -> Result<(), DoryCompressionError> {
    if let Some(value) = value {
        bytes.push(1);
        write(value, bytes)
    } else {
        bytes.push(0);
        Ok(())
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Reader<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8], DoryCompressionError> {
        let end = self
            .offset
            .checked_add(len)
            .ok_or(DoryCompressionError::Truncated)?;
        let value = self
            .bytes
            .get(self.offset..end)
            .ok_or(DoryCompressionError::Truncated)?;
        self.offset = end;
        Ok(value)
    }

    fn read_gt(&mut self) -> Result<ArkGT, DoryCompressionError> {
        let bytes: [u8; COMPRESSED_GT_SIZE] = self
            .take(COMPRESSED_GT_SIZE)?
            .try_into()
            .map_err(|_| DoryCompressionError::Truncated)?;
        decompress_gt(&bytes)
            .map(|gt| jolt_gt_to_ark(&gt))
            .map_err(Into::into)
    }

    fn read_canonical<T: CanonicalDeserialize>(
        &mut self,
        size: usize,
    ) -> Result<T, DoryCompressionError> {
        T::deserialize_compressed(self.take(size)?)
            .map_err(|_| DoryCompressionError::InvalidGroupEncoding)
    }

    fn read_g1(&mut self) -> Result<ArkG1, DoryCompressionError> {
        self.read_canonical(G1_SIZE)
    }

    fn read_g2(&mut self) -> Result<ArkG2, DoryCompressionError> {
        self.read_canonical(G2_SIZE)
    }

    fn read_fr(&mut self) -> Result<ArkFr, DoryCompressionError> {
        self.read_canonical(FR_SIZE)
    }

    fn read_u32(&mut self) -> Result<u32, DoryCompressionError> {
        let bytes: [u8; 4] = self
            .take(4)?
            .try_into()
            .map_err(|_| DoryCompressionError::Truncated)?;
        Ok(u32::from_le_bytes(bytes))
    }

    fn read_option<T>(
        &mut self,
        read: impl FnOnce(&mut Self) -> Result<T, DoryCompressionError>,
    ) -> Result<Option<T>, DoryCompressionError> {
        let tag = self
            .take(1)?
            .first()
            .copied()
            .ok_or(DoryCompressionError::Truncated)?;
        match tag {
            0 => Ok(None),
            1 => read(self).map(Some),
            _ => Err(DoryCompressionError::InvalidOptionTag),
        }
    }

    fn is_finished(&self) -> bool {
        self.offset == self.bytes.len()
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "tests may fail loudly")]
mod tests {
    use jolt_field::{Fr, RandomSampling};
    use jolt_openings::CommitmentScheme;
    use jolt_poly::Polynomial;
    use jolt_transcript::Transcript;
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    use crate::{DoryScheme, DoryVerifierSetup};

    use super::*;

    fn proof_case(num_vars: usize) -> (DoryCommitment, DoryProof, Vec<Fr>, Fr, DoryVerifierSetup) {
        let mut rng = ChaCha20Rng::seed_from_u64(73);
        let setup = DoryScheme::setup_prover(num_vars);
        let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
        let point: Vec<Fr> = (0..num_vars)
            .map(|_| <Fr as RandomSampling>::random(&mut rng))
            .collect();
        let eval = poly.evaluate(&point);
        let (commitment, hint) = DoryScheme::commit(poly.evaluations(), &setup).unwrap();
        let mut transcript = jolt_transcript::Blake2bTranscript::new(b"compressed-dory");
        let proof =
            DoryScheme::open(&poly, &point, eval, &setup, Some(hint), &mut transcript).unwrap();
        let verifier_setup = DoryVerifierSetup(setup.0.to_verifier_setup());
        (commitment, proof, point, eval, verifier_setup)
    }

    #[test]
    fn artifacts_round_trip_and_verify_natively() {
        let (commitment, proof, point, eval, verifier_setup) = proof_case(8);
        let commitments = vec![commitment.clone(); 41];
        let compressed = CompressedDoryArtifacts::from_native(&commitments, &proof).unwrap();
        assert_eq!(
            compressed.proof.encoded_len().unwrap(),
            402 + 1056 * proof.0.first_messages.len()
        );

        let encoded =
            bincode::serde::encode_to_vec(&compressed, bincode::config::standard()).unwrap();
        let native =
            bincode::serde::encode_to_vec((&commitments, &proof), bincode::config::standard())
                .unwrap();
        assert!(encoded.len() < native.len());

        let (decoded, consumed): (CompressedDoryArtifacts, usize) =
            bincode::serde::decode_from_slice(&encoded, bincode::config::standard()).unwrap();
        assert_eq!(consumed, encoded.len());
        let (recovered_commitments, recovered_proof) = decoded.into_native().unwrap();
        assert_eq!(recovered_commitments, commitments);
        assert_eq!(recovered_proof, proof);

        let mut transcript = jolt_transcript::Blake2bTranscript::new(b"compressed-dory");
        DoryScheme::verify(
            recovered_commitments.first().unwrap(),
            &point,
            eval,
            &recovered_proof,
            &verifier_setup,
            &mut transcript,
        )
        .unwrap();
    }

    #[test]
    fn proof_serde_rejects_invalid_gt() {
        let (_, proof, _, _, _) = proof_case(4);
        let compressed = CompressedDoryProof::from_native(&proof).unwrap();
        let encoded =
            bincode::serde::encode_to_vec(&compressed, bincode::config::standard()).unwrap();
        let (mut inner, _): (Vec<u8>, usize) =
            bincode::serde::decode_from_slice(&encoded, bincode::config::standard()).unwrap();
        for byte in inner.iter_mut().take(COMPRESSED_GT_SIZE) {
            *byte = u8::MAX;
        }
        let malformed = bincode::serde::encode_to_vec(inner, bincode::config::standard()).unwrap();
        assert!(bincode::serde::decode_from_slice::<CompressedDoryProof, _>(
            &malformed,
            bincode::config::standard()
        )
        .is_err());
    }
}
