//! Gnark Transpiler for Jolt Verifier
//!
//! This crate transpiles Jolt's Stage 1 verifier into Gnark circuits for Groth16 proving.
//!
//! ## Architecture
//!
//! ```text
//! Jolt Stage 1 Verifier (Rust)
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
//! - This crate just generates Gnark code from the AST
//!
//! ## Usage
//!
//! ```rust,ignore
//! use zklean_extractor::mle_ast::MleAst;
//! use gnark_transpiler::codegen::generate_circuit;
//!
//! // Execute verifier with MleAst (builds AST automatically)
//! let result: MleAst = verify_stage1::<MleAst>(...);
//!
//! // Generate Gnark circuit
//! let circuit = generate_circuit(result.root(), "Stage1Verifier");
//! std::fs::write("stage1.go", circuit).unwrap();
//! ```

pub mod ast_commitment_scheme;
pub mod ast_json;
pub mod codegen;
pub mod keccak;
pub mod mle_opening_accumulator;
pub mod poseidon;
pub mod symbolic_proof;
pub mod witness;

pub use ast_json::{export_stage1_ast, export_stage1_poseidon_ast, Stage1AstJson};
pub use codegen::{
    generate_circuit, generate_circuit_from_bundle, generate_gnark_expr, generate_stage1_circuit,
    generate_stage1_circuit_memoized, generate_stage1_circuit_with_cse,
    generate_stage1_circuit_with_global_cse, MemoizedCodeGen, sanitize_go_name,
};
pub use ast_commitment_scheme::AstCommitmentScheme;
pub use keccak::KeccakMleTranscript;
pub use mle_opening_accumulator::MleOpeningAccumulator;
pub use poseidon::PoseidonAstTranscript;
pub use symbolic_proof::{symbolize_proof, extract_witness_values, VarAllocator};
pub use witness::Stage1Witness;
