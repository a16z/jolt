//! Stage 4 verifier entry point.

pub mod outputs;
pub mod ram_val_check;
pub mod registers_read_write_checking;
mod verify;

pub use outputs::{
    Stage4ClearOutput, Stage4Output, Stage4OutputClaims, Stage4OutputPoints, Stage4ZkOutput,
};
pub use ram_val_check::{RamValCheckInitialEvaluation, RamValCheckOutputClaims};
pub use registers_read_write_checking::RegistersReadWriteOutputClaims;
pub use verify::verify;
