//! The ordinary jolt protocol family: its id families, geometry, and relation
//! instantiations.
//!
//! Ownership rule: this module owns the jolt ids and instantiates the shared
//! Twist identities (`crate::twist`) with them; that module owns the
//! algebra. `protocols::field_inline` is a completely separate protocol family
//! — the two protocol modules never import each other, and their composition
//! happens only in `jolt-verifier` (pinned by the
//! `protocol_modules_are_import_disjoint` boundary test).

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
    SpartanShiftChallenge, SpartanShiftPublic,
};
