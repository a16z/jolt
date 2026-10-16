use jolt_field::Ring;

use crate::protocols::jolt::geometry::bytecode::{
    read_raf_address_input_fold, read_raf_cycle_output, BytecodeReadRafDimensions,
};
use crate::protocols::jolt::geometry::claim_reductions::bytecode::NUM_BYTECODE_VAL_STAGES;
use crate::protocols::jolt::{
    BytecodeReadRafChallenge, JoltChallengeId, JoltDerivedId, JoltExpr, JoltOpeningId,
    JoltRelationId,
};
use crate::{SumcheckChallenges, SymbolicSumcheck};

/// Fiat-Shamir challenges drawn by the full bytecode read-RAF sumcheck: the
/// batching `gamma` plus the five per-stage gammas folding the staged claims.
#[derive(Clone, Copy, Debug, SumcheckChallenges)]
#[cfg_attr(feature = "allocative", derive(::allocative::Allocative))]
pub struct BytecodeReadRafChallenges<F> {
    #[challenge(BytecodeReadRafChallenge::Gamma)]
    pub gamma: F,
    #[challenge(BytecodeReadRafChallenge::Stage1Gamma)]
    pub stage1_gamma: F,
    #[challenge(BytecodeReadRafChallenge::Stage2Gamma)]
    pub stage2_gamma: F,
    #[challenge(BytecodeReadRafChallenge::Stage3Gamma)]
    pub stage3_gamma: F,
    #[challenge(BytecodeReadRafChallenge::Stage4Gamma)]
    pub stage4_gamma: F,
    #[challenge(BytecodeReadRafChallenge::Stage5Gamma)]
    pub stage5_gamma: F,
}

/// The full bytecode read-RAF sumcheck: folds the five staged claims plus the
/// Spartan outer/shift PC openings against the bytecode-table cycle output.
#[derive(Clone)]
pub struct ReadRaf {
    shape: BytecodeReadRafDimensions,
}

impl SymbolicSumcheck for ReadRaf {
    type RelationId = JoltRelationId;
    type OpeningId = JoltOpeningId;
    type DerivedId = JoltDerivedId;
    type ChallengeId = JoltChallengeId;
    type Shape = BytecodeReadRafDimensions;
    type Challenges<F> = BytecodeReadRafChallenges<F>;
    type Inputs<C> = crate::NoInputs<C>;
    type Outputs<C> = crate::NoOutputs<C>;

    fn new(shape: BytecodeReadRafDimensions) -> Self {
        Self { shape }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::BytecodeReadRaf
    }

    fn rounds(&self) -> usize {
        self.shape.sumcheck_rounds()
    }

    fn degree(&self) -> usize {
        self.shape.num_committed_ra_polys() + 1
    }

    fn input_expression<F: Ring>(&self) -> JoltExpr<F> {
        read_raf_address_input_fold(Vec::new())
    }

    fn output_expression<F: Ring>(&self) -> JoltExpr<F> {
        read_raf_cycle_output(self.shape, NUM_BYTECODE_VAL_STAGES)
    }
}
