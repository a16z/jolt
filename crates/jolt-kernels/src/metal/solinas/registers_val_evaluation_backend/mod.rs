//! Isolated design model for the `RegistersValEvaluation` Metal backend.
//!
//! Nothing in this directory is registered with the production backend. The
//! reusable low-level kernels remain owned by `solinas::registers_val`.

mod abi;
mod model;
mod oracle;

pub use abi::{
    RegistersValAbiError, RegistersValDirectFirstDeviceLimits, RegistersValDirectFirstDispatch,
    RegistersValDirectFirstLaunch, RegistersValDirectFirstParams, RegistersValResidentInputAbi,
    DIRECT_FIRST_SAMPLES, DIRECT_FIRST_SIMD_WIDTH, PRODUCER_STAGE, RD_INDEX_ABSENT,
};

pub use model::{
    eight_x_accepts, five_x_accepts, seven_x_accepts, FixedBoundaryProjection, KernelVariant,
    MetalPhase, PhaseProjection, PhaseWork, ProducerProjection, RegistersValConfig,
    RegistersValExecution, RegistersValFallback, RegistersValPlan, RegistersValPlanError,
    RegistersValRoofControls, RegistersValRound, RegistersValShape, ResidentBytes, RoundOwner,
    DEFAULT_CPU_TAIL_ELEMENTS, DEFAULT_TRACE_CUTOFF_ELEMENTS, FIELD_BYTES, FROZEN_COMPLETE_CPU_NS,
    FROZEN_CPU_TAIL_2_16_NS, FROZEN_EVALUATOR, FROZEN_REVISION, HOST_OPTION_INDEX_BYTES,
    M4_MAX_COPY_BYTES_PER_SECOND, M4_MAX_DIRECT_PRODUCTS_PER_SECOND,
    M4_MAX_SIX_ACCUMULATOR_PRODUCTS_PER_SECOND, REGISTER_ADDRESS_DOMAIN,
    RESIDENT_INPUT_BYTES_PER_ROW, STAGE4_PUBLISH_BYTES_PER_ROW, STAGE4_SOURCE_BYTES_PER_ROW,
    TARGET_EIGHT_X_NS, TARGET_FIVE_X_NS, TARGET_LOG_T, TARGET_SEVEN_X_NS,
};
pub use oracle::{
    eq_evaluations, lt_evaluations, output_point, CubicCoefficients, RegistersValOracle,
    RegistersValOracleError, RegistersValOutputs, RoundSamples,
};
