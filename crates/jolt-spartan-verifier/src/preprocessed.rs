//! Conditional clear v2 preprocessing and SPARK verification.
//! Joint online extraction, ROM composition, bounded decoding and ZK remain gates.
use crate::SpartanError;
use blake2::{digest::consts::U32, Blake2b, Digest};
use jolt_crypto::Bn254G1;
use jolt_field::{CanonicalBytes, Fr, One, Zero};
use jolt_hyperkzg::{HyperKZGError, HyperKZGProof, HyperKZGScheme, HyperKZGVerifierSetup};
use jolt_openings::{CommitmentScheme, OpeningsError};
use jolt_poly::EqPolynomial;
use jolt_sumcheck::SumcheckError;
use jolt_transcript::{Bn254WideBlake2bTranscript, Label, Transcript};
use serde::{Deserialize, Serialize};
use thiserror::Error;

mod protocol;
pub mod sparse;
pub use protocol::PreprocessedProof;

#[derive(Debug, Error)]
pub enum MatrixError {
    #[error("preprocessed relation check failed: {0}")]
    Relation(&'static str),
    #[error(transparent)]
    Sumcheck(#[from] SumcheckError<Fr>),
    #[error(transparent)]
    Spartan(#[from] SpartanError<Fr>),
    #[error("invalid preprocessed matrix dimensions")]
    Shape,
    #[error("application key identity or imported setup mismatch")]
    Identity,
    #[error("matrix query or public input length mismatch")]
    Query,
    #[error(transparent)]
    Setup(#[from] HyperKZGError),
    #[error(transparent)]
    Opening(#[from] OpeningsError),
}

/// Checked dimensions. Counts include ONE; private columns are reindexed from zero.
#[derive(Clone, Copy, Debug)]
pub struct MatrixShape {
    pub rows: usize,
    pub columns: usize,
    pub public_columns: usize,
    pub padded_rows: usize,
    pub padded_private: usize,
    pub operations: usize,
    pub memory: usize,
    pub public_slabs: usize,
}
impl MatrixShape {
    pub fn new(
        rows: usize,
        columns: usize,
        public_columns: usize,
        max_private_nnz: usize,
    ) -> Result<Self, MatrixError> {
        if rows == 0 || public_columns == 0 || public_columns >= columns {
            return Err(MatrixError::Shape);
        }
        let padded_rows = rows
            .max(2)
            .checked_next_power_of_two()
            .ok_or(MatrixError::Shape)?;
        let padded_private = (columns - public_columns)
            .max(2)
            .checked_next_power_of_two()
            .ok_or(MatrixError::Shape)?;
        let operations = max_private_nnz
            .max(2)
            .checked_next_power_of_two()
            .ok_or(MatrixError::Shape)?;
        let public_slabs = public_columns
            .checked_mul(3)
            .and_then(usize::checked_next_power_of_two)
            .ok_or(MatrixError::Shape)?;
        let _ = operations.checked_mul(3).ok_or(MatrixError::Shape)?;
        Ok(Self {
            rows,
            columns,
            public_columns,
            padded_rows,
            padded_private,
            operations,
            memory: padded_rows.max(padded_private),
            public_slabs,
        })
    }
    fn validate(&self) -> Result<(), MatrixError> {
        let expected = Self::new(
            self.rows,
            self.columns,
            self.public_columns,
            self.operations,
        )?;
        if self.padded_rows != expected.padded_rows
            || self.padded_private != expected.padded_private
            || self.operations != expected.operations
            || self.memory != expected.memory
            || self.public_slabs != expected.public_slabs
        {
            return Err(MatrixError::Shape);
        }
        Ok(())
    }
}

/// Opaque application-authenticated identities, in normative wire order.
#[derive(Clone, Copy, Debug)]
pub struct MatrixApplicationIds {
    pub circuit: [u8; 32],
    pub profile: [u8; 32],
    pub public_schema: [u8; 32],
    pub table: [u8; 32],
}

/// Immutable key from trusted preprocessing. Does not authenticate its own provenance.
#[derive(Clone, Debug)]
pub struct ComputationKey {
    shape: MatrixShape,
    application: MatrixApplicationIds,
    matrix_digest: [u8; 32],
    setup_id: [u8; 32],
    setup_digest: [u8; 32],
    num_powers: u64,
    max_public_degree: u64,
    commitments: [Bn254G1; 3],
}

/// Public assignment and the outer sumcheck's matrix-evaluation query.
pub struct PublicColumnQuery<'a> {
    pub inputs: &'a [Fr],
    pub point: &'a [Fr],
    pub matrix_weights: [Fr; 3],
}

/// Public-column evaluations and an actual ordinary PCS opening.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PublicColumnProof {
    pub evaluations: Vec<Fr>,
    pub opening: HyperKZGProof,
}

impl ComputationKey {
    pub fn new(
        shape: MatrixShape,
        application: MatrixApplicationIds,
        matrix_digest: [u8; 32],
        commitments: [Bn254G1; 3],
        setup: &HyperKZGVerifierSetup,
    ) -> Result<Self, MatrixError> {
        shape.validate()?;
        let binding = setup.binding()?;
        for length in [
            shape.public_slabs.checked_mul(shape.padded_rows),
            shape.operations.checked_mul(16),
            shape.memory.checked_mul(2),
        ] {
            let length = length.ok_or(MatrixError::Shape)?;
            if u64::try_from(length).map_err(|_| MatrixError::Shape)? > binding.num_powers {
                return Err(MatrixError::Shape);
            }
        }
        Ok(Self {
            shape,
            application,
            matrix_digest,
            setup_id: binding.setup_id,
            setup_digest: Self::digest(&binding.canonical_bytes),
            num_powers: binding.num_powers,
            max_public_degree: binding.max_public_degree,
            commitments,
        })
    }

    /// Parameterized unkeyed BLAKE2b-256, not a truncated 512-bit digest.
    pub fn digest(bytes: &[u8]) -> [u8; 32] {
        Blake2b::<U32>::digest(bytes).into()
    }

    /// Exact 476-byte v2 encoding. Constructor validation pins shape and setup.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = b"JOLT-SPARK-KEY\0\0".to_vec();
        for id in [2u32, 1, 1, 1, 1, 1, 1] {
            out.extend(id.to_le_bytes());
        }
        let mut modulus = (-Fr::one()).to_bytes_le_vec();
        for byte in &mut modulus {
            let (value, carry) = byte.overflowing_add(1);
            *byte = value;
            if !carry {
                break;
            }
        }
        out.extend(modulus);
        for count in [
            self.shape.rows,
            self.shape.columns,
            self.shape.public_columns,
            self.shape.padded_rows,
            self.shape.padded_private,
            self.shape.operations,
            self.shape.memory,
            self.shape.public_slabs,
        ] {
            out.extend((count as u64).to_le_bytes());
        }
        out.extend(self.num_powers.to_le_bytes());
        out.extend(self.max_public_degree.to_le_bytes());
        for digest in [
            self.application.circuit,
            self.application.profile,
            self.application.public_schema,
            self.application.table,
            self.matrix_digest,
            self.setup_id,
            self.setup_digest,
        ] {
            out.extend(digest);
        }
        for commitment in &self.commitments {
            out.extend(commitment.compressed_bytes());
        }
        out
    }
    pub fn id(&self) -> [u8; 32] {
        Self::digest(&self.canonical_bytes())
    }
    pub fn shape(&self) -> MatrixShape {
        self.shape
    }

    fn authenticate(
        &self,
        expected_id: &[u8; 32],
        setup: &HyperKZGVerifierSetup,
    ) -> Result<(), MatrixError> {
        let binding = setup.binding()?;
        if self.id() != *expected_id
            || binding.setup_id != self.setup_id
            || binding.num_powers != self.num_powers
            || binding.max_public_degree != self.max_public_degree
            || Self::digest(&binding.canonical_bytes) != self.setup_digest
        {
            return Err(MatrixError::Identity);
        }
        Ok(())
    }

    /// v2 initialization through the retained post-tau outer marker. The caller
    /// supplies a fresh wide384 transcript in the v2 application domain.
    pub fn begin(
        &self,
        expected_id: &[u8; 32],
        setup: &HyperKZGVerifierSetup,
        public_inputs: &[Fr],
        witness_commitment: &Bn254G1,
        transcript: &mut Bn254WideBlake2bTranscript,
    ) -> Result<Vec<Fr>, MatrixError> {
        self.authenticate(expected_id, setup)?;
        if public_inputs.len() != self.shape.public_columns - 1 {
            return Err(MatrixError::Query);
        }
        transcript.append(&Label(b"computation-key"));
        transcript.append_bytes(expected_id);
        transcript.append_values(b"public-inputs", public_inputs);
        transcript.append_labeled(b"witness-commitment", witness_commitment);
        let tau = transcript.challenge_vector(self.shape.padded_rows.trailing_zeros() as usize);
        transcript.append_labeled(b"spartan-outer", &Fr::zero());
        Ok(tau)
    }

    /// Shared claim-before-selector schedule consumed by prover and verifier.
    pub fn public_opening_query(
        &self,
        rx: &[Fr],
        evaluations: &[Fr],
        transcript: &mut Bn254WideBlake2bTranscript,
    ) -> Result<(Vec<Fr>, Fr), MatrixError> {
        if rx.len() != self.shape.padded_rows.trailing_zeros() as usize
            || evaluations.len() != self.shape.public_columns * 3
        {
            return Err(MatrixError::Query);
        }
        transcript.append(&Label(b"matrix-public-v2"));
        transcript.append_values(b"public-columns", evaluations);
        transcript.append(&Label(b"public-selector"));
        let mut point =
            transcript.challenge_vector(self.shape.public_slabs.trailing_zeros() as usize);
        let weights = EqPolynomial::new(point.clone()).evaluations();
        let value = weights.iter().zip(evaluations).map(|(a, b)| *a * b).sum();
        point.extend_from_slice(rx);
        Ok((point, value))
    }

    /// Authenticate the public-column contribution only. This does not verify
    /// the outer sumcheck, private matrices, or witness opening.
    pub fn verify_public(
        &self,
        expected_id: &[u8; 32],
        setup: &HyperKZGVerifierSetup,
        query: PublicColumnQuery<'_>,
        proof: &PublicColumnProof,
        transcript: &mut Bn254WideBlake2bTranscript,
    ) -> Result<Fr, MatrixError> {
        self.authenticate(expected_id, setup)?;
        if query.inputs.len() != self.shape.public_columns - 1 {
            return Err(MatrixError::Query);
        }
        let (point, value) =
            self.public_opening_query(query.point, &proof.evaluations, transcript)?;
        let commitment = self.commitments.first().ok_or(MatrixError::Shape)?;
        HyperKZGScheme::verify(commitment, &point, value, &proof.opening, setup, transcript)?;
        let public = std::iter::once(Fr::one())
            .chain(query.inputs.iter().copied())
            .collect::<Vec<_>>();
        Ok(proof
            .evaluations
            .chunks_exact(self.shape.public_columns)
            .zip(query.matrix_weights)
            .map(|(columns, weight)| {
                weight * columns.iter().zip(&public).map(|(a, b)| *a * b).sum::<Fr>()
            })
            .sum())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_crypto::{Bn254, JoltGroup};

    #[test]
    fn normative_serialization_only_key_vector() {
        // Opaque digests from the independently generated wire-format fixture,
        // deliberately not an authenticated relation or an SRS.
        let key = ComputationKey {
            shape: MatrixShape {
                rows: 4,
                columns: 8,
                public_columns: 3,
                padded_rows: 4,
                padded_private: 8,
                operations: 2,
                memory: 8,
                public_slabs: 16,
            },
            application: MatrixApplicationIds {
                circuit: [1; 32],
                profile: [2; 32],
                public_schema: [3; 32],
                table: [4; 32],
            },
            matrix_digest: [5; 32],
            setup_id: [6; 32],
            setup_digest: [7; 32],
            num_powers: 64,
            max_public_degree: 63,
            commitments: [
                Bn254::g1_generator(),
                Bn254::g1_generator().scalar_mul(&(-Fr::one())),
                Bn254::g1_generator().scalar_mul(&Fr::zero()),
            ],
        };
        assert_eq!(
            key.canonical_bytes(),
            include_bytes!("fixtures/preprocessed-key-v2.bin").as_slice()
        );
        assert_eq!(
            key.id(),
            [
                137, 206, 218, 31, 91, 119, 183, 244, 240, 139, 47, 234, 122, 233, 32, 185, 209,
                157, 15, 26, 225, 187, 214, 109, 89, 154, 162, 166, 56, 5, 44, 27
            ]
        );
    }
}
