//! Lattice-mode symbolic sumcheck relations: the additional (or variant)
//! relations layered on the base `jolt/` PIOP when committing through the
//! packed lattice witness. See `specs/lattice-claims.md`.

pub mod booleanity;
pub mod bytecode_reconstruction;
pub mod digit_zero;
pub mod program_image_reconstruction;
pub mod read_raf;
