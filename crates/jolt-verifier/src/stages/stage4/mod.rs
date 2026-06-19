//! Stage 4 verifier entry point.

pub mod inputs;
pub mod outputs;
pub mod ram_val_check;
pub mod registers_read_write_checking;
mod verify;

pub use inputs::{deps, Deps, Stage4Claims};
pub use outputs::{Stage4ClearOutput, Stage4Output, Stage4ZkOutput};
pub use ram_val_check::{
    RamValCheck, RamValCheckAdviceClaims, RamValCheckInputClaims, RamValCheckOutputClaims,
};
pub use registers_read_write_checking::{
    RegistersReadWriteChecking, RegistersReadWriteInputClaims, RegistersReadWriteOutputClaims,
};
pub use verify::{
    append_ram_val_check_gamma_domain_separator, stage4_expected_final_claim, verify,
};
