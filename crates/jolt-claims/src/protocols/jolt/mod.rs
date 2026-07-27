pub mod geometry;
pub mod lattice;
pub mod relations;

mod ids;

pub use geometry::{
    claim_reductions::advice::AdviceClaimReductionLayout,
    claim_reductions::bytecode::BytecodeClaimReductionLayout,
    claim_reductions::precommitted::{
        PrecommittedClaimReduction, PrecommittedReductionDimensions, PrecommittedReductionLayout,
        PrecommittedSchedulingReference,
    },
    claim_reductions::program_image::ProgramImageClaimReductionLayout,
    dimensions::{
        CommitmentMatrixShape, JoltFormulaDimensions, JoltOneHotConfig, JoltOneHotDimensions,
        JoltReadWriteConfig, ReadWriteDimensions, TraceDimensions, TracePolynomialOrder,
    },
    error::{JoltFormulaDimensionsError, JoltFormulaPointError},
};
pub use ids::{
    AdviceClaimReductionPublic, BooleanityChallenge, BooleanityPublic,
    BytecodeChunkReconstructionChallenge, BytecodeChunkReconstructionPublic,
    BytecodeClaimReductionChallenge, BytecodeClaimReductionPublic, BytecodeReadRafChallenge,
    BytecodeReadRafPublic, BytecodeRegisterLane, HammingWeightClaimReductionChallenge,
    HammingWeightClaimReductionPublic, IncClaimReductionChallenge, IncClaimReductionPublic,
    InstructionClaimReductionChallenge, InstructionClaimReductionPublic, InstructionInputChallenge,
    InstructionInputPublic, InstructionRaVirtualizationChallenge,
    InstructionRaVirtualizationPublic, InstructionReadRafChallenge, InstructionReadRafPublic,
    JoltAdviceKind, JoltChallengeId, JoltCommittedPolynomial, JoltDerivedId, JoltExpr,
    JoltOpeningId, JoltPolynomialId, JoltRelationId, JoltVirtualPolynomial,
    ProgramImageClaimReductionPublic, ProgramImageReconstructionPublic, RamHammingBooleanityPublic,
    RamOutputCheckPublic, RamRaClaimReductionChallenge, RamRaClaimReductionPublic,
    RamRaVirtualizationPublic, RamRafEvaluationPublic, RamReadWriteChallenge, RamReadWritePublic,
    RamValCheckChallenge, RamValCheckPublic, RegistersClaimReductionChallenge,
    RegistersClaimReductionPublic, RegistersReadWriteChallenge, RegistersReadWritePublic,
    RegistersValEvaluationPublic, SpartanOuterPublic, SpartanProductVirtualizationPublic,
    SpartanShiftChallenge, SpartanShiftPublic, TrustedAdviceReconstructionPublic,
    UntrustedAdviceReconstructionChallenge, UntrustedAdviceReconstructionPublic,
};

// The empty claim structs contribute no claim-graph edges; these impls let
// analyses bound on `ClaimAdjacency` treat every relation uniformly.
impl<C> crate::ClaimAdjacency for crate::NoInputs<C> {
    type Id = JoltOpeningId;
    const EDGES: &'static [crate::ClaimEdge<JoltOpeningId>] = &[];
}

impl<C> crate::ClaimAdjacency for crate::NoOutputs<C> {
    type Id = JoltOpeningId;
    const EDGES: &'static [crate::ClaimEdge<JoltOpeningId>] = &[];
}
