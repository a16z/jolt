//! The stage 7 `UnsignedIncChunkReconstruction` sumcheck instance (packed
//! path).
//!
//! Consumes the stage-6b chunk/msb booleanity openings and the `FusedInc`
//! claim from the stage-6a `IncVirtualization` phase, and re-opens every
//! chunk column plus the msb at its own bound `(address ‖ cycle)` point
//! (`chunk_width + log_t` rounds) — the packed final claims for those slots.
//! One γ batches its four duties (per-chunk hamming and booleanity-reduction
//! legs, the msb reduction leg, and the shifted decode tying the columns to
//! `2^64 + FusedInc`).

use jolt_claims::protocols::jolt::lattice::relations::chunk_reconstruction::{
    ChunkReconstruction as ChunkReconstructionSymbolic, ChunkReconstructionChallenges,
    ChunkReconstructionDimensions, ChunkReconstructionInputClaims, ChunkReconstructionOutputClaims,
};
use jolt_claims::protocols::jolt::{
    JoltDerivedId, JoltRelationId, UnsignedIncChunkReconstructionPublic,
};
use jolt_claims::SymbolicSumcheck;
use jolt_field::Field;
use jolt_poly::{try_eq_mle, IdentityPolynomial, MultilinearEvaluation};

use crate::stages::relations::ConcreteSumcheck;
use crate::VerifierError;

fn chunk_public_failed(reason: impl ToString) -> VerifierError {
    VerifierError::StageClaimPublicInputFailed {
        stage: JoltRelationId::UnsignedIncChunkReconstruction,
        reason: reason.to_string(),
    }
}

/// The `UnsignedIncChunkReconstruction` member: the γ-batched
/// hamming/reduction/decode fold over the unsigned-inc chunk and msb columns.
pub struct ChunkReconstruction<F: Field> {
    symbolic: ChunkReconstructionSymbolic,
    dimensions: ChunkReconstructionDimensions,
    /// The stage-6 booleanity point (`r_address ‖ r_cycle`) the reduction legs
    /// compare against.
    r_booleanity: Vec<F>,
    /// The `IncVirtualization` cycle point anchoring the hamming and
    /// shifted-decode legs.
    r_inc_cycle: Vec<F>,
}

impl<F: Field> ChunkReconstruction<F> {
    pub fn new(
        dimensions: ChunkReconstructionDimensions,
        r_booleanity: Vec<F>,
        r_inc_cycle: Vec<F>,
    ) -> Self {
        Self {
            symbolic: ChunkReconstructionSymbolic::new(dimensions),
            dimensions,
            r_booleanity,
            r_inc_cycle,
        }
    }
}

impl<F: Field> ConcreteSumcheck<F> for ChunkReconstruction<F> {
    type Symbolic = ChunkReconstructionSymbolic;

    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }

    fn derive_opening_points(
        &self,
        sumcheck_point: &[F],
        _input_points: &ChunkReconstructionInputClaims<Vec<F>>,
    ) -> Result<ChunkReconstructionOutputClaims<Vec<F>>, VerifierError> {
        // Cycle variables bind low, so the reversed sumcheck point is
        // `(r_address ‖ r_cycle)`; the msb has no address variables and opens
        // at the cycle tail.
        let opening_point = sumcheck_point.iter().rev().copied().collect::<Vec<_>>();
        let r_cycle = opening_point[self.dimensions.chunking.chunk_width()..].to_vec();
        Ok(ChunkReconstructionOutputClaims {
            chunks: vec![opening_point; self.dimensions.chunking.chunk_count()],
            msb: r_cycle,
        })
    }

    fn derive_output_term(
        &self,
        id: &JoltDerivedId,
        _input_points: &ChunkReconstructionInputClaims<Vec<F>>,
        output_points: &ChunkReconstructionOutputClaims<Vec<F>>,
        _challenges: &ChunkReconstructionChallenges<F>,
    ) -> Result<F, VerifierError> {
        let JoltDerivedId::UnsignedIncChunkReconstruction(public) = id else {
            return Err(VerifierError::MissingStageClaimDerived { id: *id });
        };
        let opening_point = output_points
            .chunks()
            .first()
            .ok_or_else(|| chunk_public_failed("reconstruction produced no chunk openings"))?;
        let chunk_width = self.dimensions.chunking.chunk_width();
        let (r_address, r_cycle) = opening_point.split_at(chunk_width);
        if self.r_booleanity.len() != opening_point.len() {
            return Err(chunk_public_failed(format!(
                "booleanity point has {} variables, expected {}",
                self.r_booleanity.len(),
                opening_point.len()
            )));
        }
        let (bool_address, bool_cycle) = self.r_booleanity.split_at(chunk_width);
        match public {
            UnsignedIncChunkReconstructionPublic::EqBooleanityAddress => {
                try_eq_mle(r_address, bool_address).map_err(chunk_public_failed)
            }
            UnsignedIncChunkReconstructionPublic::EqBooleanityCycle => {
                try_eq_mle(r_cycle, bool_cycle).map_err(chunk_public_failed)
            }
            UnsignedIncChunkReconstructionPublic::EqIncCycle => {
                try_eq_mle(r_cycle, &self.r_inc_cycle).map_err(chunk_public_failed)
            }
            UnsignedIncChunkReconstructionPublic::IdentityAtAddress => {
                Ok(IdentityPolynomial::new(r_address.len()).evaluate(r_address))
            }
        }
    }
}
