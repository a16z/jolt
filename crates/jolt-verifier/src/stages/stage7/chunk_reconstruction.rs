//! The stage 7 lattice `UnsignedIncChunkReconstruction` sumcheck instance.
//!
//! Consumes the stage-6 chunk/msb booleanity openings and the `FusedInc`
//! opening, and re-opens every chunk column at a fresh address point
//! (`log_k_chunk` rounds), proving in one gamma-batched fold that the chunks
//! are unit vectors (hamming leg), match their booleanity-point claims
//! (`EqBooleanityAddress` leg), and decode to the low 64 bits of the shifted
//! fused increment (`IdentityAtAddress` leg).

use jolt_claims::protocols::jolt::lattice::identity_mle;
use jolt_claims::protocols::jolt::lattice::relations::chunk_reconstruction::{
    ChunkReconstruction as ChunkReconstructionSymbolic, ChunkReconstructionChallenges,
    ChunkReconstructionInputClaims, ChunkReconstructionOutputClaims,
};
use jolt_claims::protocols::jolt::lattice::UnsignedIncChunking;
use jolt_claims::protocols::jolt::{
    JoltDerivedId, JoltRelationId, UnsignedIncChunkReconstructionPublic,
};
use jolt_claims::SymbolicSumcheck;
use jolt_field::Field;
use jolt_poly::try_eq_mle;

use crate::stages::relations::{ConcreteSumcheck, GetPoint, OpeningClaim};
use crate::VerifierError;

pub struct ChunkReconstruction<F: Field> {
    symbolic: ChunkReconstructionSymbolic,
    chunking: UnsignedIncChunking,
    /// The stage-6 booleanity cycle suffix; the re-opened chunk points are
    /// `reverse(sumcheck_point) ++ r_cycle`.
    r_cycle: Vec<F>,
    /// The stage-6a booleanity address prefix the `EqBooleanityAddress`
    /// public compares against.
    r_booleanity_address: Vec<F>,
}

impl<F: Field> ChunkReconstruction<F> {
    pub fn new(
        chunking: UnsignedIncChunking,
        r_cycle: Vec<F>,
        r_booleanity_address: Vec<F>,
    ) -> Self {
        Self {
            symbolic: ChunkReconstructionSymbolic::new(chunking),
            chunking,
            r_cycle,
            r_booleanity_address,
        }
    }
}

fn public_input_failed(reason: impl ToString) -> VerifierError {
    VerifierError::StageClaimPublicInputFailed {
        stage: JoltRelationId::UnsignedIncChunkReconstruction,
        reason: reason.to_string(),
    }
}

impl<F: Field> ConcreteSumcheck<F> for ChunkReconstruction<F> {
    type Symbolic = ChunkReconstructionSymbolic;

    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }

    fn derive_opening_points<C: GetPoint<F>>(
        &self,
        sumcheck_point: &[F],
        _inputs: &ChunkReconstructionInputClaims<C>,
    ) -> Result<ChunkReconstructionOutputClaims<Vec<F>>, VerifierError> {
        let r_address = sumcheck_point.iter().rev().copied().collect::<Vec<_>>();
        let opening_point = [r_address.as_slice(), self.r_cycle.as_slice()].concat();
        Ok(ChunkReconstructionOutputClaims {
            chunks: vec![opening_point; self.chunking.chunk_count()],
        })
    }

    fn derive_output_term<C: GetPoint<F>>(
        &self,
        id: &JoltDerivedId,
        _inputs: &ChunkReconstructionInputClaims<C>,
        outputs: &ChunkReconstructionOutputClaims<OpeningClaim<F>>,
        _challenges: &ChunkReconstructionChallenges<F>,
    ) -> Result<F, VerifierError> {
        let JoltDerivedId::UnsignedIncChunkReconstruction(public) = id else {
            return Err(VerifierError::MissingStageClaimDerived { id: *id });
        };
        let opening_point = outputs
            .chunks
            .first()
            .map(GetPoint::point)
            .ok_or_else(|| public_input_failed("reconstruction produced no chunk openings"))?;
        let r_address = &opening_point[..self.chunking.chunk_width()];
        match public {
            UnsignedIncChunkReconstructionPublic::EqBooleanityAddress => {
                try_eq_mle(r_address, &self.r_booleanity_address).map_err(public_input_failed)
            }
            UnsignedIncChunkReconstructionPublic::IdentityAtAddress => Ok(identity_mle(r_address)),
        }
    }
}
