//! Full clear v2 algebraic verifier. Joint extraction/ROM and ZK remain separate gates.
use super::sparse::{SparseMatrixProof, SparseQuery};
use super::{ComputationKey, MatrixError, PublicColumnProof, PublicColumnQuery};
use crate::{key::check_outer_relation, SpartanKey, INNER_DEGREE, OUTER_DEGREE};
use jolt_crypto::Bn254G1;
use jolt_field::{Fr, Zero};
use jolt_hyperkzg::{HyperKZGProof, HyperKZGScheme, HyperKZGVerifierSetup};
use jolt_openings::CommitmentScheme;
use jolt_sumcheck::{
    BooleanHypercube, CompressedSumcheckProof, SumcheckClaim, SUMCHECK_ROUND_TRANSCRIPT_LABEL,
};
use jolt_transcript::{Bn254WideBlake2bTranscript, Transcript};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PreprocessedProof {
    pub witness_commitment: Bn254G1,
    pub outer: CompressedSumcheckProof<Fr>,
    pub outer_evaluations: [Fr; 3],
    pub public: PublicColumnProof,
    pub inner: CompressedSumcheckProof<Fr>,
    pub private_values: [Fr; 3],
    pub sparse: SparseMatrixProof,
    pub witness_evaluation: Fr,
    pub witness_opening: HyperKZGProof,
}
impl ComputationKey {
    pub fn outer_weights(
        evaluations: &[Fr; 3],
        transcript: &mut Bn254WideBlake2bTranscript,
    ) -> [Fr; 3] {
        transcript.append_values(b"outer-evaluations", evaluations);
        [
            transcript.challenge(),
            transcript.challenge(),
            transcript.challenge(),
        ]
    }
    pub fn inner_claim(
        evaluations: [Fr; 3],
        weights: [Fr; 3],
        public: Fr,
        transcript: &mut Bn254WideBlake2bTranscript,
    ) -> Fr {
        let claim = evaluations
            .iter()
            .zip(weights)
            .map(|(a, b)| *a * b)
            .sum::<Fr>()
            - public;
        transcript.append_labeled(b"spartan-inner", &claim);
        claim
    }
    /// Complete clear v2 relation checks with real PCS verification. Conditional
    /// algebraic prototype, not a certified SNARK. `expected_id` is application
    /// authenticated. Container decoding and setup provenance remain external.
    pub fn verify(
        &self,
        expected_id: &[u8; 32],
        public_inputs: &[Fr],
        proof: &PreprocessedProof,
        setup: &HyperKZGVerifierSetup,
    ) -> Result<(), MatrixError> {
        let mut transcript = Bn254WideBlake2bTranscript::new(b"spartan-preprocessed-clear-v2");
        let tau = self.begin(
            expected_id,
            setup,
            public_inputs,
            &proof.witness_commitment,
            &mut transcript,
        )?;
        let row_vars = self.shape.padded_rows.trailing_zeros() as usize;
        let outer = proof.outer.verify(
            &SumcheckClaim {
                num_vars: row_vars,
                degree: OUTER_DEGREE,
                claimed_sum: Fr::zero(),
            },
            BooleanHypercube,
            SUMCHECK_ROUND_TRANSCRIPT_LABEL,
            &mut transcript,
        )?;
        check_outer_relation(
            row_vars,
            &tau,
            outer.point.as_slice(),
            outer.value,
            proof.outer_evaluations,
        )?;
        let weights = Self::outer_weights(&proof.outer_evaluations, &mut transcript);
        let public = self.verify_public(
            expected_id,
            setup,
            PublicColumnQuery {
                inputs: public_inputs,
                point: outer.point.as_slice(),
                matrix_weights: weights,
            },
            &proof.public,
            &mut transcript,
        )?;
        let claim = Self::inner_claim(proof.outer_evaluations, weights, public, &mut transcript);
        let inner = proof.inner.verify(
            &SumcheckClaim {
                num_vars: self.shape.padded_private.trailing_zeros() as usize,
                degree: INNER_DEGREE,
                claimed_sum: claim,
            },
            BooleanHypercube,
            SUMCHECK_ROUND_TRANSCRIPT_LABEL,
            &mut transcript,
        )?;
        self.verify_sparse(
            SparseQuery {
                rows: outer.point.as_slice(),
                columns: inner.point.as_slice(),
                values: proof.private_values,
            },
            &proof.sparse,
            setup,
            &mut transcript,
        )?;
        let linear = weights
            .iter()
            .zip(proof.private_values)
            .map(|(a, b)| *a * b)
            .sum::<Fr>();
        if inner.value != linear * proof.witness_evaluation {
            return Err(MatrixError::Relation("private witness product"));
        }
        SpartanKey::append_witness_evaluation(proof.witness_evaluation, &mut transcript);
        HyperKZGScheme::verify(
            &proof.witness_commitment,
            inner.point.as_slice(),
            proof.witness_evaluation,
            &proof.witness_opening,
            setup,
            &mut transcript,
        )?;
        Ok(())
    }
}
