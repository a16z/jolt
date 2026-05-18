//! Generic BlindFold claim, protocol, layout, and verifier-equation types.

mod error;
mod inputs;
mod instance_claims;
mod proof;
pub mod protocol;
pub mod r1cs;
mod relaxed;
mod verify;

pub use error::{Error, LayoutError, RelaxedError, VerificationError};
pub use inputs::{Inputs, StageInput, VectorInputs, VectorStageInput};
pub use instance_claims::{InstanceClaims, StageClaims};
pub use jolt_r1cs::{
    assert_claim_expr_eq, lower_claim_expr, ClaimLoweringError, ClaimSourceTable, ClaimSources,
    LinearCombination, R1csBuilder, Variable,
};
pub use proof::BlindFoldProof;
pub use protocol::{BlindFoldDimensions, BlindFoldProtocol, RowDimensions, WitnessRowLayout};
pub use r1cs::{
    allocate_layout, append as append_r1cs, build as build_r1cs, Layout, RoundLayout, StageLayout,
};
pub use relaxed::{RelaxedInstance, RelaxedWitness};
pub use verify::verify;
