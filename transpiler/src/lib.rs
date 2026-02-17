//! Transpiler for Jolt Verifier
//!
//! This crate transpiles Jolt's verifier (stages 1-7, all sumchecks) into circuit code
//! for various proving backends. Currently supported: gnark (Go/Groth16).
//!
//! # Architecture
//!
//! ```text
//! JoltProof (concrete Fr values)
//!     ↓ symbolize_proof()
//! JoltProof<MleAst> (symbolic variables)
//!     ↓ TranspilableVerifier::verify()
//! AST in NODE_ARENA (recorded operations)
//!     ↓ target-specific codegen
//! Circuit code (e.g., stages_circuit.go for gnark)
//!     ↓ target prover
//! Proof (e.g., 164 bytes Groth16)
//! ```
//!
//! # Target Backends
//!
//! Currently supported:
//! - **gnark**: Go/Groth16 circuit generation (~250 constraints per Poseidon hash)
//!
//! Future targets (not yet implemented):
//! - Circom
//! - Plonky2
//!
//! # Key Concepts
//!
//! ## Symbolic Execution with MleAst
//!
//! `MleAst` is a type that implements the `JoltField` trait but records operations
//! as an AST instead of computing them. When we run the verifier with `MleAst`,
//! every `+`, `*`, `-`, `==` operation creates AST nodes.
//!
//! ## Per-Constraint Expression Trees
//!
//! Each constraint (sumcheck assertion) gets its own isolated expression tree with
//! independent CSE (Common Subexpression Elimination) namespacing: constraint 0 uses
//! `cse_0_*`, constraint 1 uses `cse_1_*`, etc. This makes debugging easier - when a
//! constraint fails, all its `cse_N_*` variables are self-contained.
//!
//! ## Stages Covered
//!
//! This crate transpiles stages 1-7 (all sumcheck verifications):
//! - Stages 1-6: Standard sumcheck verifications
//! - Stage 7: HammingWeight claim reduction sumcheck
//!
//! Stage 8 (PCS/Hyrax) is NOT transpiled because it requires native elliptic curve
//! operations. For a complete recursive verifier, see `quangvdao/quang-jolt` which
//! uses Hyrax over Grumpkin with native curve operations.
//!
//! **Note**: Stage 7 does not include `AdviceClaimReduction` verifiers (they require
//! state management across stages 6-7). For proofs without advice, this is complete.
//!
//! # Transcript Feature Flags
//!
//! The transpiler must use the same transcript as proof generation:
//! - `--features transcript-poseidon`: Poseidon hash (SNARK-friendly, recommended)
//! - `--features transcript-keccak`: Keccak hash
//! - `--features transcript-blake2b`: Blake2b hash (default if none specified)
//!
//! **Note**: Only Poseidon-generated proofs can be efficiently verified in-circuit.
//!
//! # Module Overview
//!
//! - [`gnark_codegen`]: AST → Go/gnark code generation with CSE
//! - [`symbolic_proof`]: Convert concrete proofs to symbolic form
//! - [`symbolic_traits`]: Trait implementations for MleAst transpilation
//!   - [`symbolic_traits::poseidon`]: Poseidon transcript for symbolic Fiat-Shamir
//!   - [`symbolic_traits::opening_accumulator`]: Symbolic opening accumulator
//!   - [`symbolic_traits::ast_commitment_scheme`]: Stub commitment scheme for transpilation
//!
//! # Usage
//!
//! See `main.rs` for the full transpilation pipeline, or use the library directly:
//!
//! ```ignore
//! use transpiler::{symbolize_proof, gnark_codegen, PoseidonAstTranscript};
//!
//! let (symbolic_proof, accumulator, var_alloc) = symbolize_proof::<PoseidonAstTranscript>(&real_proof);
//! // ... run TranspilableVerifier::verify() ...
//! let circuit_code = gnark_codegen::generate_circuit_from_bundle(&bundle, "MyCircuit");
//! ```

pub mod gnark_codegen;
pub mod symbolic_proof;
pub mod symbolic_traits;

pub use gnark_codegen::{generate_circuit_from_bundle, sanitize_go_name};
pub use symbolic_proof::{symbolize_proof, VarAllocator};
pub use symbolic_traits::{AstCommitmentScheme, AstOpeningAccumulator, PoseidonAstTranscript};

// Re-export transcript types based on feature flags (matching jolt-core pattern)
// This allows main.rs to use the selected transcript without conditional imports

// Compile-time error if multiple transcript features are enabled
#[cfg(any(
    all(feature = "transcript-poseidon", feature = "transcript-keccak"),
    all(feature = "transcript-poseidon", feature = "transcript-blake2b"),
    all(feature = "transcript-keccak", feature = "transcript-blake2b"),
    all(
        feature = "transcript-poseidon",
        feature = "transcript-keccak",
        feature = "transcript-blake2b"
    )
))]
compile_error!("Cannot enable multiple transcript features simultaneously. Please choose exactly one of: 'transcript-poseidon', 'transcript-keccak', or 'transcript-blake2b'.");

/// The selected AST transcript type based on feature flags.
/// For symbolic execution, this determines which transcript implementation to use.
///
/// Note: Currently only Poseidon is implemented for symbolic execution.
/// Other transcript types will use PoseidonAstTranscript as a fallback
/// (the circuit still uses Poseidon internally regardless of proof transcript).
#[cfg(feature = "transcript-poseidon")]
pub type SelectedAstTranscript = PoseidonAstTranscript;

#[cfg(feature = "transcript-keccak")]
pub type SelectedAstTranscript = PoseidonAstTranscript; // TODO: KeccakAstTranscript when implemented

#[cfg(feature = "transcript-blake2b")]
pub type SelectedAstTranscript = PoseidonAstTranscript; // TODO: Blake2bAstTranscript when implemented

#[cfg(not(any(
    feature = "transcript-poseidon",
    feature = "transcript-keccak",
    feature = "transcript-blake2b"
)))]
pub type SelectedAstTranscript = PoseidonAstTranscript; // Default to Poseidon
