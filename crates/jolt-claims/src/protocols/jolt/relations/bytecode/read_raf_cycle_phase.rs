use jolt_field::Ring;

use super::{BytecodeReadRafCycleShape, BytecodeReadRafInputClaims, BytecodeReadRafOutputClaims};
use crate::protocols::jolt::geometry::bytecode::{
    bytecode_read_raf_address_phase_opening, read_raf_cycle_output,
};
use crate::protocols::jolt::{
    BytecodeReadRafChallenge, JoltChallengeId, JoltDerivedId, JoltExpr, JoltOpeningId,
    JoltRelationId,
};
use crate::{opening, SumcheckChallenges, SymbolicSumcheck};

#[derive(Clone, Copy, Debug, SumcheckChallenges)]
#[cfg_attr(feature = "allocative", derive(::allocative::Allocative))]
pub struct BytecodeReadRafCyclePhaseChallenges<F> {
    #[challenge(BytecodeReadRafChallenge::Gamma)]
    pub gamma: F,
}

/// The cycle phase of the bytecode read-RAF sumcheck: starts from the staged
/// address-phase opening and reduces to the bytecode-table cycle output.
#[derive(Clone)]
pub struct ReadRafCyclePhase {
    shape: BytecodeReadRafCycleShape,
}

impl SymbolicSumcheck for ReadRafCyclePhase {
    type RelationId = JoltRelationId;
    type OpeningId = JoltOpeningId;
    type DerivedId = JoltDerivedId;
    type ChallengeId = JoltChallengeId;
    type Shape = BytecodeReadRafCycleShape;
    type Challenges<F> = BytecodeReadRafCyclePhaseChallenges<F>;
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
        read_raf_cycle_output(self.shape.0, self.shape.1)
    }
}
