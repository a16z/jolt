//! Verifier-side scalar arithmetic abstraction.
//!
//! `FieldBackend` lifts the verifier's field-level computation off the concrete
//! [`Field`](jolt_field::Field) so the same verification logic can run against
//! multiple targets:
//!
//! | Backend     | `Scalar`    | What it does                                          |
//! |-------------|-------------|-------------------------------------------------------|
//! | [`Native`]  | `F`         | Direct field ops; identical codegen to native code    |
//! | `Tracing`   | `NodeId`    | Records every op into an AST (recursion / Lean)       |
//! | `R1CSGen`   | `LcId`      | Emits R1CS constraints (replaces hand-rolled R1CS)    |
//!
//! Only [`Native`] and [`Tracing`] are implemented in this crate. R1CSGen
//! lives downstream; it only needs to provide `Scalar` and the trait
//! methods, no other API changes.
//!
//! # Why not just `Field`?
//!
//! The native implementation IS just `Field` with `Scalar = F` and identity
//! wrapping. The trait exists for backends where each operation needs to be
//! observed:
//!
//! - **Tracing** has to know that "this multiplication came from the verifier's
//!   eq evaluation" or "this scalar was sampled from the transcript at round 5".
//! - **R1CSGen** has to emit a fresh R1CS variable for every wrapped scalar
//!   and a constraint for every assertion.
//! - **Provenance** ([`ScalarOrigin`]) lets backends label inputs by source —
//!   public verifier-key data, untrusted proof data, transcript challenges.
//!
//! # Zero overhead
//!
//! [`Native`] is a unit struct. Every method is `#[inline(always)]` and forwards
//! directly to the underlying field operator. Rust's monomorphization erases
//! the trait, producing the same code as if the verifier had been hand-written
//! against `F` from the start.
//!
//! # Helpers
//!
//! [`eq_eval`], [`univariate_horner`], and the Lagrange/preprocessed-poly
//! helpers in [`mod@helpers`] compose with any backend without trait surface
//! bloat. They live here so this crate stays the single source of truth for
//! backend-aware primitives.

#![cfg_attr(not(test), warn(missing_docs))]

mod backend;
mod error;
pub mod helpers;
mod native;
pub mod tracing;
pub mod viz;

pub use backend::{FieldBackend, ScalarOrigin};
pub use error::BackendError;
pub use helpers::{eq_eval, eq_evals_table, pow_u64, univariate_horner};
pub use native::Native;
pub use tracing::{replay as replay_trace, AstAssertion, AstGraph, AstNodeId, AstOp, Tracing};
pub use viz::{to_dot, to_mermaid};
