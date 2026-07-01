//! Stage 3 verifier: Spartan shift, instruction input, and register reduction.

pub mod instruction_input;
pub mod outputs;
pub mod registers_claim_reduction;
pub mod spartan_shift;
mod verify;

pub use instruction_input::{
    InstructionInput, InstructionInputInputClaims, InstructionInputOutputClaims,
};
pub use outputs::{
    Stage3Challenges, Stage3ClearOutput, Stage3InputClaims, Stage3InputPoints, Stage3Output,
    Stage3OutputClaims, Stage3OutputPoints, Stage3Sumchecks, Stage3ZkOutput,
};
pub use registers_claim_reduction::{
    RegistersClaimReduction, RegistersClaimReductionInputClaims,
    RegistersClaimReductionOutputClaims,
};
pub use spartan_shift::{SpartanShift, SpartanShiftInputClaims, SpartanShiftOutputClaims};
pub use verify::{stage3_expected_final_claim, verify};
