//! [`ProverStage`](crate::stage::ProverStage) implementations.
//!
//! Each module implements one of the 7 sumcheck stages in the Jolt
//! proving pipeline, constructing claims and witnesses for the
//! batched sumcheck prover.

pub mod s1_spartan;
pub mod s2_product_virtual;
pub mod s2_ra_virtual;
pub mod s3_claim_reductions;
pub mod s3_instruction_input;
pub mod s3_shift;
pub mod s4_ram_rw;
pub mod s4_rw_checking;
pub mod s5_ram_checking;
pub mod s5_registers_val_eval;
pub mod s6_booleanity;
pub mod s6_ra_booleanity;
pub mod s7_hamming_reduction;
pub mod s8_opening;
pub mod s_bytecode_read_raf;
pub mod s_instruction_read_raf;
