//! Generic BlindFold instance, layout, and verifier-equation types.

mod error;
mod instance;
pub mod r1cs;

pub use error::{Error, LayoutError};
pub use instance::{Inputs, Instance, Stage, StageInput, VectorInputs, VectorStageInput};
pub use jolt_r1cs::{
    assert_claim_expr_eq, lower_claim_expr, ClaimLoweringError, ClaimSourceResolver,
    ClaimSourceTable, LinearCombination, R1csBuilder, Variable,
};
pub use r1cs::{
    allocate_layout, append as append_r1cs, build as build_r1cs, Layout, RoundLayout, StageLayout,
};
