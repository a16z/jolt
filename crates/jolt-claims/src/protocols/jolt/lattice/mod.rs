//! Lattice (Akita) mode: the additional protocol semantics for running the
//! Jolt PIOP over a packed one-hot witness committed with a
//! non-homomorphic PCS. Design: `specs/lattice-claims.md`.
//!
//! This module names facts only — packed column ids and arities, decode-view
//! term lists, the extra relations, and the final-opening discharge map. Slot
//! assignment and the packed-opening reduction live in
//! `jolt-openings::PrefixPacking` (feed it [`packing::proof_packed_columns`]
//! verbatim); witness materialization, transcripts, and stage orchestration
//! live in the verifier/prover crates.

pub mod discharge;
pub mod geometry;
pub mod ids;
pub mod packing;
pub mod relations;
pub mod views;

pub use discharge::{final_opening, packed_column_leaf, LatticeFinalOpening};
pub use geometry::{
    LatticeGeometryError, UnsignedIncChunking, LATTICE_BYTECODE_VAL_STAGES, UNSIGNED_INC_BITS,
};
pub use ids::{
    AdviceBytesValidityChallenge, AdviceBytesValidityPublic, BytecodeRegisterLane,
    IncVirtualizationChallenge, IncVirtualizationPublic, LatticeColumn,
    UnsignedIncChunkReconstructionChallenge, UnsignedIncChunkReconstructionPublic,
};
pub use packing::{
    precommitted_packed_columns, proof_packed_columns, PrecommittedPackingShape, ProofPackingShape,
};
pub use views::{DecodeTerm, LatticeView};
