//! Stage 4 verifier entry point.

pub mod outputs;
pub mod ram_val_check;
pub mod registers_read_write_checking;
mod verify;

pub use outputs::{
    Stage4Challenges, Stage4ClearOutput, Stage4Output, Stage4OutputClaims, Stage4ZkOutput,
};
pub use ram_val_check::{
    ram_val_check_advice_block, RamValCheck, RamValCheckAdviceBlock, RamValCheckAdviceClaims,
    RamValCheckInitialEvaluation, RamValCheckInputClaims, RamValCheckOutputClaims,
    VerifiedRamValCheckAdviceContribution,
};
pub use registers_read_write_checking::{
    RegistersReadWriteChecking, RegistersReadWriteInputClaims, RegistersReadWriteOutputClaims,
};
pub use verify::{
    append_ram_val_check_gamma_domain_separator, stage4_expected_final_claim,
    stage4_output_claims_with_points, verify,
};
