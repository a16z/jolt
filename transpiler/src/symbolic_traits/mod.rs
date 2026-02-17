//! Symbolic trait implementations for MleAst transpilation.
//!
//! This module contains implementations of jolt-core traits for `MleAst`, the symbolic
//! field type used during transpilation. These implementations allow the Jolt verifier
//! to run symbolically, recording operations as an AST instead of computing concrete values.
//!
//! # Modules
//!
//! - [`commitment_scheme`]: `CommitmentScheme` implementation (`AstCommitmentScheme`)
//! - [`opening_accumulator`]: `OpeningAccumulator` implementation (`MleOpeningAccumulator`)
//! - [`transcript`]: `Transcript` implementation (`PoseidonAstTranscript`)
//!
//! # Usage
//!
//! These types are used together to run the Jolt verifier symbolically:
//!
//! ```ignore
//! use transpiler::symbolic_traits::{
//!     AstCommitmentScheme, MleOpeningAccumulator, PoseidonAstTranscript
//! };
//! use jolt_core::zkvm::transpilable_verifier::TranspilableVerifier;
//!
//! let verifier = TranspilableVerifier::<
//!     MleAst,                    // Symbolic field (records operations)
//!     AstCommitmentScheme,       // Stub commitment scheme
//!     PoseidonAstTranscript,     // Symbolic Poseidon transcript
//!     MleOpeningAccumulator,     // Collects opening claims
//! >::new(...);
//!
//! verifier.verify(&proof, ...);  // Runs stages 1-7, records AST
//! ```

pub mod commitment_scheme;
pub mod opening_accumulator;
pub mod transcript;

pub use commitment_scheme::AstCommitmentScheme;
pub use opening_accumulator::MleOpeningAccumulator;
pub use transcript::PoseidonAstTranscript;
