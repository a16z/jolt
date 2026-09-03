//! T1: the Blake3 transcript table — the Jolt verifier's Fiat-Shamir chain
//! from the first commitment absorb to the Dory `d` squeeze as half-G-step
//! rows over committed bit columns in 128-row compression cells, proven by two
//! head-aligned stage-A members: the degree-3 row relation
//! (`Σ_row eq(τ₁, row) · Σ_j γ_j C_j(row) = 0`) and the degree-3 wiring
//! zero-check binding every wired column to the committed word it copies,
//! pinning the protocol constants and the canonicality of every absorbed
//! field element.
//!
//! - [`blake3`]: the compression function with a half-step trace and the
//!   streaming keyed chain, byte-exact with `jolt_transcript::Blake3Transcript`.
//! - [`recorder`]: a transcript decorator logging a verifier run.
//! - [`schedule`]: the symbolic schedule (verifier key: byte identities,
//!   pins, wire rows) and the witness view of a proof's recorded run.
//! - [`layout`]: columns, the aligned quadratic row relation, `final_check`.
//! - [`wiring`]: the position table of the copy constraints, wired-column
//!   materialization, public inputs, the wiring statement's verifier side.
//! - [`table`]: witness generation.
//! - [`prover`] / [`wiring_prover`]: the two `jolt_sumcheck::prover::ProveRounds`
//!   members.
//! - [`terms`]: the Fiat–Shamir randomizers, the batched final relation as
//!   affine-form terms, virtual value columns, link row maps.
//! - [`adapter`]: T1 on the wrapper stream (columns, members, `TermExporter`).

pub mod adapter;
pub mod blake3;
pub mod layout;
pub mod prover;
pub mod recorder;
pub mod schedule;
pub mod table;
pub mod terms;
pub mod wiring;
pub mod wiring_prover;

pub use adapter::{Members, StreamColumns, StreamTermExporter};
pub use layout::{
    ColumnEvals, Relation, WiredWord, WordColumn, COMMITTED, CONSTRAINTS, DEGREE, WIRED_BITS,
    WIRED_WORDS,
};
pub use prover::HashTableProver;
pub use recorder::{Decoder, Event, Recorded, RecordingTranscript};
pub use schedule::{
    ByteSource, CellIndex, CellPlan, ElementKind, ItemClass, JoltSchedule, ScheduleError, Squeeze,
    SymbolicSchedule,
};
pub use table::HashTable;
pub use terms::{AffineForm, ColumnId, FinalContext, LinkMap, T1Challenges, Term};
pub use wiring::{
    PublicInputs, VkColumn, VkColumns, VkEvals, WiringStatement, MODULUS_HI, WIRING_TERMS,
};
pub use wiring_prover::WiringProver;
