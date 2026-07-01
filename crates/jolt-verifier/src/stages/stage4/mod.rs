//! Stage 4 verifier entry point.

pub mod outputs;
pub mod ram_val_check;
pub mod registers_read_write_checking;
mod verify;

pub use outputs::{
    Stage4Challenges, Stage4ClearOutput, Stage4InputClaims, Stage4InputPoints, Stage4Output,
    Stage4OutputClaims, Stage4OutputPoints, Stage4Sumchecks, Stage4ZkOutput,
};
pub use ram_val_check::{
    append_ram_val_check_gamma_domain_separator, ram_val_check_advice_block, RamValCheck,
    RamValCheckAdviceBlock, RamValCheckInitialEvaluation, RamValCheckInputClaims,
    RamValCheckOutputClaims, VerifiedRamValCheckAdviceContribution,
};
pub use registers_read_write_checking::{
    RegistersReadWriteChecking, RegistersReadWriteInputClaims, RegistersReadWriteOutputClaims,
};
pub use verify::verify;
