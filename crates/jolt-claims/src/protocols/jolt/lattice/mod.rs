//! Lattice (Akita) mode: the additional protocol semantics for running the
//! Jolt PIOP over a packed one-hot witness committed with a
//! non-homomorphic PCS. Design: `specs/lattice-claims.md`.
//!
//! This module names facts only — the canonical OneHotTrace selector layout,
//! auxiliary fixed-prefix layouts, extra relations,
//! and final-opening map. Witness generation, transcripts, and stage
//! orchestration live in the verifier and prover crates.
//!
//! # Vocabulary
//!
//! Nothing lattice-specific; everything is inherited:
//!
//! - Auxiliary packing layer (`jolt-openings`): a packed object holds **logical
//!   polynomials** in **slots**; a slot has a **prefix** and `num_vars`, and
//!   a logical point is the packed point's suffix.
//! - Per family the dimensions keep their existing names: `(address ‖
//!   cycle)` for the trace one-hots (`Ra` families, inc digits, carry — base
//!   vocabulary), `(byte ‖ place ‖ word)` for byte-decomposed data (place
//!   `i` carries place value `256^i`, matching
//!   [`BalancedIncChunking::place_value`]), and `(lane ‖ row)` for bytecode.
//! - The one cross-family convention: every logical polynomial's Boolean
//!   index is `(hot-value bits ‖ instance bits)`, msb-first, so the instance
//!   bits are always the logical point's suffix.
//! - **final claim** — claims flow through the relation DAG until, per
//!   polynomial, one claim remains that no relation consumes. In base mode
//!   the stage-8 RLC batch settles it; in lattice mode the semantic
//!   OneHotTrace claims are reduced at a random selector and the single
//!   physical polynomial is opened once. Auxiliary columns use packed-slot
//!   claims.

pub mod geometry;
pub mod packing;
pub mod relations;

pub use geometry::{
    BalancedIncChunking, LatticeGeometryError, FUSED_INC_BITS, LATTICE_BYTECODE_VAL_STAGES,
};
pub mod strategy;
pub use packing::{
    advice_packing_plan, one_hot_trace_columns, precommitted_packing_plan, OneHotTraceShape,
    PrecommittedPackingPlan, PrecommittedPackingShape, PrefixPackedObjectPlan,
    ADVICE_MAX_PHYSICAL_VARS, ADVICE_MIN_PHYSICAL_VARS,
};
pub use strategy::{
    OneHotTraceColumnRanges, OneHotTraceLayout, OneHotTraceLayoutPlan, OneHotTraceSetupShape,
    ONE_HOT_TRACE_LAYOUT,
};
