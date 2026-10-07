//! Stage 6b (cycle-phase) verifier entry point.

pub mod batch;
pub mod booleanity;
#[cfg(feature = "akita-byte-link")]
mod byte_link;
pub mod bytecode_read_raf;
pub mod committed_reduction_cycle_phase;
pub mod inc_claim_reduction;
pub mod instruction_ra_virtualization;
pub mod outputs;
pub mod ram_hamming_booleanity;
pub mod ram_ra_virtualization;
pub mod verify;

pub use outputs::{Stage6bClearOutput, Stage6bOutput, Stage6bZkOutput};
#[cfg(not(feature = "akita-byte-link"))]
pub use verify::stage6b_opening_values;
pub use verify::{stage6b_input_points_from_upstream, stage6b_input_values_from_upstream, verify};
