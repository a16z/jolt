//! Lattice (Akita) mode: the additional protocol semantics for running the
//! Jolt PIOP over a packed one-hot witness committed with a
//! non-homomorphic PCS. Design: `specs/lattice-claims.md`.
//!
//! This module names facts only — the packed column set and its canonical
//! registration with `jolt-openings::PrefixPacking` (the single source of
//! truth for slot assignment), decode-view term lists, the extra relations,
//! and the final-opening discharge map. Witness materialization, transcripts,
//! and stage orchestration live in the verifier/prover crates.

pub mod geometry;
pub mod ids;
pub mod packing;
pub mod relations;

pub use geometry::{
    LatticeGeometryError, UnsignedIncChunking, LATTICE_BYTECODE_VAL_STAGES, UNSIGNED_INC_BITS,
};
pub use ids::{
    AdviceBytesValidityChallenge, AdviceBytesValidityPublic, IncVirtualizationChallenge,
    IncVirtualizationPublic, UnsignedIncChunkReconstructionChallenge,
    UnsignedIncChunkReconstructionPublic,
};
pub use packing::{
    advice_bytes_polynomial, advice_word_decode_terms, byte_decode_terms,
    bytecode_chunk_decode_terms, final_opening, packed_column_leaf, precommitted_packing,
    program_image_word_decode_terms, proof_packing, scaled_byte_decode_terms,
    weighted_symbol_terms, BytecodeRegisterLane, DecodeTerm, LatticeColumn, LatticeFinalOpening,
    PrecommittedPackingShape, ProofPackingShape,
};
