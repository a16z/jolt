//! Stage 4 verifier entry point.

pub mod outputs;
pub mod ram_val_check;
pub mod registers_read_write_checking;
mod verify;

pub use outputs::{
    Stage4ClearOutput, Stage4Output, Stage4OutputClaims, Stage4OutputPoints, Stage4ZkOutput,
};
pub use ram_val_check::{
    ram_val_check_init_structure, RamValCheckInitStructure, RamValCheckInitialEvaluation,
    RamValCheckOutputClaims, VerifiedRamValCheckAdviceContribution,
};
pub use registers_read_write_checking::RegistersReadWriteOutputClaims;
pub use verify::{
    public_initial_ram_evaluation, stage4_input_points_from_upstream,
    stage4_input_values_from_upstream, verify,
};
