//! Gnark Transpiler for Jolt Verifier
//!
//! This crate transpiles Jolt's verifier (stages 1-6) into Gnark circuits for Groth16 proving.
//!
//! # Architecture
//!
//! ```text
//! JoltProof (concrete Fr values)
//!     ↓ symbolize_proof()
//! JoltProof<MleAst> (symbolic variables)
//!     ↓ TranspilableVerifier::verify()
//! AST in NODE_ARENA (recorded operations)
//!     ↓ generate_circuit_from_bundle()
//! stages_circuit.go (Gnark circuit code)
//!     ↓ go test / groth16.Prove()
//! Groth16 Proof (164 bytes) → EVM (~280k gas)
//! ```
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
//! constraint fails, all its `cse_N_*` variables are self-contained, so you can trace
//! through the expression tree without cross-referencing other sumchecks.
//!
//! ## Stages Covered
//!
//! This crate transpiles stages 1-6 (all sumcheck verifications). The PCS stage
//! is NOT transpiled because Dory uses pairings, which would add ~100M constraints
//! if emulated. For a complete recursive verifier, see `quangvdao/quang-jolt` which
//! uses Hyrax over Grumpkin with native curve operations.
//!
//! # Transcript Feature Flags
//!
//! The transpiler must use the same transcript as proof generation:
//! - `--features transcript-poseidon`: Poseidon hash (SNARK-friendly, recommended)
//! - `--features transcript-keccak`: Keccak hash
//! - `--features transcript-blake2b`: Blake2b hash (default if none specified)
//!
//! **Note**: Only Poseidon-generated proofs can be efficiently verified in-circuit.
//! Keccak/Blake2b are provided for compatibility but the generated circuit would
//! be infeasibly large (~150k constraints per hash vs ~250 for Poseidon).
//!
//! # Module Overview
//!
//! - [`codegen`]: AST → Go code generation with CSE
//! - [`symbolic_proof`]: Convert concrete proofs to symbolic form
//! - [`poseidon`]: Poseidon transcript for symbolic Fiat-Shamir
//! - [`mle_opening_accumulator`]: Symbolic opening accumulator
//! - [`ast_commitment_scheme`]: Stub commitment scheme for transpilation
//!
//! # Usage
//!
//! See `main.rs` for the full transpilation pipeline, or use the library directly:
//!
//! ```ignore
//! use gnark_transpiler::{symbolize_proof, generate_circuit_from_bundle, PoseidonAstTranscript};
//!
//! let (symbolic_proof, accumulator, var_alloc) = symbolize_proof::<PoseidonAstTranscript>(&real_proof);
//! // ... run TranspilableVerifier::verify() ...
//! let circuit_code = generate_circuit_from_bundle(&bundle, "MyCircuit");
//! ```

pub mod ast_commitment_scheme;
pub mod codegen;
pub mod mle_opening_accumulator;
pub mod poseidon;
pub mod symbolic_proof;

pub use ast_commitment_scheme::AstCommitmentScheme;
pub use codegen::{generate_circuit_from_bundle, sanitize_go_name};
pub use mle_opening_accumulator::MleOpeningAccumulator;
pub use poseidon::PoseidonAstTranscript;
pub use symbolic_proof::{extract_witness_values, symbolize_proof, VarAllocator};

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
