//! Compute kernels for the modular Jolt prover: all heavy, transcript-free
//! compute behind the backend seam, so implementations swap without touching
//! protocol structure. `jolt-prover` holds orchestration only — every
//! field-element crunch lives here, behind a slot of the [`JoltBackend`]
//! registry it proves against.
//!
//! Kernel APIs consume witness oracles, field elements, and PCS setups —
//! never a transcript, never Fiat-Shamir — and return canonical values.
//! Sumcheck kernels implement the fused `jolt_sumcheck::ProveRounds` round
//! contract and are minted per relation through ONE universal trait,
//! [`PrepareKernel`]: its typed request is the relation instance itself
//! (inside a `ProverInputs` bundle), so kernels read geometry off relation
//! accessors instead of restated constructor arguments. Non-oracle data
//! reaches a kernel through the two other channels `prepare` receives:
//! typed witness rows off the [`JoltVmWitnessPlane`](jolt_witness::protocols::jolt_vm::JoltVmWitnessPlane)
//! accessors, and [`ProofSession`] carries (prover-retained program data,
//! parked at proof start; cross-stage kernel state). Only the bespoke slots
//! keep hand-shaped trait modules at the crate root: the uni-skip handoffs
//! ([`spartan_outer`], [`spartan_product`]), the typed-row witness
//! ([`instruction_read_raf`]), the two-batch precommitted reduction family
//! ([`precommitted_reduction`]), commitment streaming, and the joint
//! opening. Reference implementations live under [`reference`].
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

mod backend;
mod commitment;
pub mod committed_program;
mod error;
pub mod instruction_read_raf;
pub mod opening;
pub mod precommitted_reduction;
pub mod reference;
pub mod spartan_outer;
pub mod spartan_product;

pub use backend::{JoltBackend, PrepareKernel, ProofSession, RetainedProgram};
pub use commitment::{CommitWitness, CommitmentGrid, WitnessCommitment};
pub use error::KernelError;
/// Re-exported from `jolt-verifier` (its home since the generated prove
/// drivers must name it); the kernel crate keeps the path for downstream
/// stability.
pub use jolt_verifier::stages::relations::{SumcheckKernel, SumcheckKernelError};
pub use reference::naive::NaiveSumcheckProver;
pub use reference::ReferenceBackend;
