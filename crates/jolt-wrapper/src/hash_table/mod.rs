//! T1: the Blake3 transcript table — the Jolt verifier's Fiat-Shamir chain
//! from the first commitment absorb to the Dory `d` squeeze as half-G-step
//! rows over committed bit columns, proven by one degree-3 sumcheck
//! (`Σ_row eq(τ, row) · Σ_j γ_j C_j(row) = 0`).
//!
//! - [`blake3`]: the compression function with a half-step trace and the
//!   streaming keyed chain, byte-exact with `jolt_transcript::Blake3Transcript`.
//! - [`recorder`]: a transcript decorator logging a verifier run.
//! - [`schedule`]: the Jolt run as a table segment; item classification for
//!   the link table.
//! - [`layout`]: columns, the aligned quadratic row relation, `final_check`.
//! - [`table`]: witness generation — rows, wiring feeds, links.
//! - [`prover`]: the row sumcheck as a `jolt_sumcheck::prover::ProveRounds`
//!   member.

pub mod blake3;
pub mod layout;
pub mod prover;
pub mod recorder;
pub mod schedule;
pub mod table;

pub use layout::{ColumnEvals, Relation, COMMITTED, CONSTRAINTS, DEGREE, WIRED_BITS, WIRED_WORDS};
pub use prover::HashTableProver;
pub use recorder::{Decoder, Event, Recorded, RecordingTranscript};
pub use schedule::{ElementKind, ItemClass, JoltSchedule, ScheduleError};
pub use table::{ChallengeLink, Feed, HashTable, MessageLink, MessageSource, RowFeeds};
