//! Compute kernels for the modular Jolt prover: all heavy, transcript-free
//! compute behind the backend seam, so implementations swap without touching
//! protocol structure. `jolt-prover` holds orchestration only — every
//! field-element crunch lives here, behind a slot of the [`JoltBackend`]
//! registry it proves against.
//!
//! Kernel APIs consume witness oracles, field elements, and PCS setups —
//! never a transcript, never Fiat-Shamir — and return canonical values.
//! Sumcheck kernels implement `jolt_sumcheck::ProveRounds` and are added per
//! relation as stages demand them: each per-relation module at the crate
//! root defines the slot's object-safe factory/instance traits (the seam),
//! and its sibling under [`reference`] holds the reference implementation.
//! The [`NaiveSumcheckProver`] is the reference tier: it
//! interprets a relation's output `Expr` with polynomial-valued leaves,
//! making any relation whose leaves are multilinear provable at harness
//! scale with zero relation-specific code; optimized kernels are
//! equivalence-tested against it. Derived ids correspond one-to-one with
//! multilinears by design, so every batch member is naive-provable; the only
//! hand-written reference compute is the uni-skip first-round polynomials
//! (single univariate rounds over a centered integer domain — outside the
//! sumcheck round model entirely). See `specs/clean-slate-prover.md`,
//! "The backend seam".
//!
//! The commitment kernel streams PCS commitments of the committed witness
//! polynomials over the proof's shared embedding grid.

pub mod advice_claim_reduction;
mod backend;
pub mod booleanity;
pub mod bytecode_claim_reduction;
pub mod bytecode_read_raf;
mod commitment;
pub mod committed_program;
mod error;
pub mod hamming_weight_claim_reduction;
pub mod inc_claim_reduction;
pub mod instruction_claim_reduction;
pub mod instruction_input;
pub mod instruction_ra_virtualization;
pub mod instruction_read_raf;
pub mod opening;
pub mod precommitted_reduction;
pub mod program_image_claim_reduction;
pub mod ram_hamming_booleanity;
pub mod ram_output_check;
pub mod ram_ra_claim_reduction;
pub mod ram_ra_virtualization;
pub mod ram_raf_evaluation;
pub mod ram_read_write;
pub mod ram_val_check;
pub mod reference;
pub mod registers_claim_reduction;
pub mod registers_read_write;
pub mod registers_val_evaluation;
pub mod spartan_outer;
pub mod spartan_product;
pub mod spartan_shift;

pub use backend::{JoltBackend, ProofSession};
pub use commitment::{CommitWitness, CommitmentGrid, WitnessCommitment};
pub use error::KernelError;
/// Re-exported from `jolt-verifier` (its home since the generated prove
/// drivers must name it); the kernel crate keeps the path for downstream
/// stability.
pub use jolt_verifier::stages::relations::{SumcheckKernel, SumcheckKernelError};
pub use reference::naive::NaiveSumcheckProver;
pub use reference::ReferenceBackend;
