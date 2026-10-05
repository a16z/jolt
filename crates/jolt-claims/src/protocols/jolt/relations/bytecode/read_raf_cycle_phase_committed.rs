use jolt_field::Ring;

use super::{BytecodeReadRafCycleShape, BytecodeReadRafInputClaims, BytecodeReadRafOutputClaims};
use crate::protocols::jolt::geometry::bytecode::{
    bytecode_read_raf_address_phase_opening, read_raf_cycle_output_committed,
};
use crate::protocols::jolt::{
    BytecodeReadRafChallenge, JoltChallengeId, JoltDerivedId, JoltExpr, JoltOpeningId,
    JoltRelationId,
};
use crate::{opening, SumcheckChallenges, SymbolicSumcheck};

#[derive(Clone, Copy, Debug, PartialEq, Eq, SumcheckChallenges)]
#[cfg_attr(feature = "allocative", derive(::allocative::Allocative))]
pub struct BytecodeReadRafCyclePhaseCommittedChallenges<F> {
    #[challenge(BytecodeReadRafChallenge::Gamma)]
    pub gamma: F,
}

/// Committed-program cycle phase: the per-stage Val factors come from the
/// `BytecodeValClaim(s)` openings staged at the end of the address phase
/// instead of public bytecode-table evaluations.
#[derive(Clone)]
pub struct ReadRafCyclePhaseCommitted {
    shape: BytecodeReadRafCycleShape,
}

impl SymbolicSumcheck for ReadRafCyclePhaseCommitted {
    type RelationId = JoltRelationId;
    type OpeningId = JoltOpeningId;
    type DerivedId = JoltDerivedId;
    type ChallengeId = JoltChallengeId;
    type Shape = BytecodeReadRafCycleShape;
    type Challenges<F> = BytecodeReadRafCyclePhaseCommittedChallenges<F>;
    type Inputs<C> = BytecodeReadRafInputClaims<C>;
    type Outputs<C> = BytecodeReadRafOutputClaims<C>;

    fn new(shape: BytecodeReadRafCycleShape) -> Self {
        Self { shape }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::BytecodeReadRaf
    }

    fn rounds(&self) -> usize {
        self.shape.0.log_t()
    }

    fn degree(&self) -> usize {
        self.shape.0.num_committed_ra_polys() + 1
    }

    fn input_expression<F: Ring>(&self) -> JoltExpr<F> {
        opening(bytecode_read_raf_address_phase_opening())
    }

    fn output_expression<F: Ring>(&self) -> JoltExpr<F> {
        read_raf_cycle_output_committed(self.shape.0, self.shape.1)
    }
}
