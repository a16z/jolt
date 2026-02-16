//! Gnark Transpiler for Jolt Verifier
//!
//! This crate transpiles Jolt's verifier (stages 1-6) into Gnark circuits for Groth16 proving.
//!
//! ## Architecture
//!
//! ```text
//! Jolt Verifier (Rust)
//!     ↓ (runtime introspection with zkLean's MleAst)
//! MLE AST (in global NODE_ARENA)
//!     ↓ (this crate)
//! Gnark Circuit (Go)
//!     ↓ (Gnark compiler)
//! Groth16 Proof → EVM (280k gas)
//! ```
//!
//! ## Strategy
//!
//! We reuse zkLean's infrastructure:
//! - `MleAst` implements `JoltField` trait
//! - Running verifier with `MleAst` builds AST automatically
//! - This crate generates Gnark code from the AST bundle
//!
//! ## Usage
//!
//! See `main.rs` for the full transpilation pipeline using `generate_circuit_from_bundle`.

pub mod ast_commitment_scheme;
pub mod codegen;
pub mod mle_opening_accumulator;
pub mod poseidon;
pub mod symbolic_proof;

pub use codegen::{generate_circuit_from_bundle, sanitize_go_name};
pub use ast_commitment_scheme::AstCommitmentScheme;
pub use mle_opening_accumulator::MleOpeningAccumulator;
pub use poseidon::PoseidonAstTranscript;
pub use symbolic_proof::{symbolize_proof, extract_witness_values, VarAllocator};
