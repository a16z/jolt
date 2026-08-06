//! Isolated design model for the `RegistersValEvaluation` Metal backend.
//!
//! Nothing in this directory is registered with the production backend. The
//! reusable low-level kernels remain owned by `solinas::registers_val`.

mod oracle;
mod plan;

pub use oracle::{RegistersValOracle, RegistersValOracleError, RegistersValOutputs, RoundSamples};
pub use plan::{
    eight_x_accepts, five_x_accepts, KernelVariant, MetalPhase, PhaseProjection, PhaseWork,
    RegistersValConfig, RegistersValExecution, RegistersValFallback, RegistersValPlan,
    RegistersValPlanError, RegistersValRoofControls, RegistersValRound, RegistersValShape,
    ResidentBytes, RoundOwner, DEFAULT_CPU_TAIL_ELEMENTS, DEFAULT_TRACE_CUTOFF_ELEMENTS,
    FIELD_BYTES, FROZEN_COMPLETE_CPU_NS, FROZEN_EVALUATOR, FROZEN_REVISION,
    M4_MAX_COPY_BYTES_PER_SECOND, M4_MAX_DIRECT_PRODUCTS_PER_SECOND,
    M4_MAX_SIX_ACCUMULATOR_PRODUCTS_PER_SECOND, REGISTER_ADDRESS_DOMAIN, TARGET_EIGHT_X_NS,
    TARGET_FIVE_X_NS, TARGET_LOG_T,
};
