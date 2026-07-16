//! Lattice-mode symbolic sumcheck relations: the additional (or variant)
//! relations layered on the base `jolt/` PIOP when committing through the
//! packed lattice witness. See `specs/lattice-claims.md`.

pub mod advice_reconstruction;
pub mod booleanity;
pub mod bytecode_reconstruction;
pub mod fused_inc_claim_reduction;
pub mod hamming_weight;
pub mod inc_virtualization;
pub mod program_image_reconstruction;
pub mod read_raf;
