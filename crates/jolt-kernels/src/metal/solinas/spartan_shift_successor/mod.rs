//! First-principles successor packet for the Spartan shift Metal member.
//!
//! This module is intentionally unregistered. It freezes the cross-stage
//! partial-carrier ABI, exact roof model, dense oracle, and MSL sketches before
//! shared stage-driver work begins.

pub mod abi;
pub mod model;
pub mod oracle;

pub const SOURCE: &str = include_str!("shader.metal");

pub use abi::{
    CarrierProducer, MidpointUpcLease, PartialCarrierHeader, PartialCarrierLease,
    ResidentBufferDescriptor, ResidualPlaneLease, SpartanShiftSuccessorAbiError,
    SpartanShiftSuccessorFlagWord, SpartanShiftSuccessorFoldParams, SpartanShiftSuccessorGeometry,
    SpartanShiftSuccessorPartialParams, SpartanShiftSuccessorReductionParams,
};
pub use model::{
    target_plan, work_plan, AttributionBoundary, MidpointPlan, PhaseWork, RoofBounds, WorkPlan,
};
pub use oracle::{
    attach_midpoint_upc, combine_q, direct_trace, factorized_initial_claim, fold_all_columns,
    fold_residual_columns, outer_component_tables, product_component_tables, DenseOutputs,
    DirectTrace, OuterComponentTables, PrefixQTables, ProductComponentTables,
    SpartanShiftSuccessorOracleError, SpartanShiftSuccessorRow,
};

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module: fail loudly")]
mod tests;
