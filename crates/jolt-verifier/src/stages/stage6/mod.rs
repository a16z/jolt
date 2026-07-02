//! Stage 6 verifier entry point.

pub mod batch;
pub mod booleanity;
pub mod bytecode_read_raf;
pub mod committed_reduction_cycle_phase;
pub mod inc_claim_reduction;
pub mod instruction_ra_virtualization;
pub mod outputs;
pub mod ram_hamming_booleanity;
pub mod ram_ra_virtualization;
pub mod verify;

pub use outputs::{Stage6ClearOutput, Stage6Output, Stage6ZkOutput};
pub use verify::verify;
