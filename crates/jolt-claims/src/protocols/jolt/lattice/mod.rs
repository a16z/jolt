//! Lattice (Akita) mode: the additional protocol semantics for running the
//! Jolt PIOP over a packed one-hot witness committed with a
//! non-homomorphic PCS. Design: `specs/lattice-claims.md`.
//!
//! This module names facts only — the packed column set and its canonical
//! registration with `jolt-openings::PrefixPacking` (the single source of
//! truth for slot assignment), reconstruction term lists, the extra
//! relations, and the final-opening map. Witness materialization, transcripts,
//! and stage orchestration live in the verifier/prover crates.

pub mod geometry;
pub mod packing;
pub mod relations;

pub use geometry::{
    LatticeGeometryError, UnsignedIncChunking, LATTICE_BYTECODE_VAL_STAGES, UNSIGNED_INC_BITS,
};
pub use packing::{
    advice_word_reconstruction_terms, byte_reconstruction_terms,
    bytecode_chunk_reconstruction_terms, final_opening, precommitted_packing,
    program_image_word_reconstruction_terms, proof_packing, BytecodeRegisterLane, LatticeColumn,
    LatticeFinalOpening, PrecommittedPackingShape, ProofPackingShape, ReconstructionTerm,
};
