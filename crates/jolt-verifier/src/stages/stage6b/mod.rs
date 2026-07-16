//! Stage 6b (cycle-phase) verifier entry point.

pub mod batch;
pub mod booleanity;
pub mod bytecode_read_raf;
pub mod committed_reduction_cycle_phase;
#[cfg(feature = "akita")]
pub mod fused_inc_claim_reduction;
pub mod inc_claim_reduction;
pub mod instruction_ra_virtualization;
pub mod outputs;
pub mod ram_hamming_booleanity;
pub mod ram_ra_virtualization;
pub mod verify;

pub use outputs::{Stage6bClearOutput, Stage6bOutput, Stage6bZkOutput};
pub use verify::verify;
