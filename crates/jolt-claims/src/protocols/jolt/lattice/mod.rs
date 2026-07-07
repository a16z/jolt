//! Lattice (Akita) mode: the additional protocol semantics for running the
//! Jolt PIOP over a packed one-hot witness committed with a
//! non-homomorphic PCS. Design: `specs/lattice-claims.md`.
//!
//! This module names facts only — the packed polynomial set and its
//! canonical registration with `jolt-openings::PrefixPacking` (the single
//! source of truth for slot assignment), the extra relations, and the
//! final-opening map. Witness materialization, transcripts, and stage
//! orchestration live in the verifier/prover crates.
//!
//! # Vocabulary
//!
//! Nothing lattice-specific; everything is inherited:
//!
//! - Packing layer (`jolt-openings`): a packed witness holds **logical
//!   polynomials** in **slots**; a slot has a **prefix** and `num_vars`, and
//!   a logical point is the packed point's suffix.
//! - Per family the dimensions keep their existing names: `(address ‖
//!   cycle)` for the trace one-hots (`Ra` families, inc chunks, msb — base
//!   vocabulary), `(byte ‖ place ‖ word)` for byte-decomposed data (place
//!   `i` carries place value `256^i`, matching
//!   [`UnsignedIncChunking::place_value`]), and `(lane ‖ row)` for bytecode.
//! - The one cross-family convention: every logical polynomial's Boolean
//!   index is `(hot-value bits ‖ instance bits)`, msb-first, so the instance
//!   bits are always the packed point's suffix.
//! - **final claim** — claims flow through the relation DAG until, per
//!   polynomial, one claim remains that no relation consumes. In base mode
//!   the stage-8 RLC batch settles it; in lattice mode it is a packed slot's
//!   single `EvaluationClaim` ([`LatticeFinalOpening::Packed`] names its
//!   producing relation output), unless the polynomial is `Virtualized` —
//!   every claim on it consumed by a lattice relation — in which case it has
//!   none.

pub mod geometry;
pub mod packing;
pub mod relations;

pub use geometry::{
    LatticeGeometryError, UnsignedIncChunking, LATTICE_BYTECODE_VAL_STAGES, UNSIGNED_INC_BITS,
};
pub use packing::{
    final_opening, precommitted_packing, proof_packing, LatticeFinalOpening,
    PrecommittedPackingShape, ProofPackingShape,
};
