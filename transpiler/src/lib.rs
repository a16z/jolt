//! Transpiler for Jolt Verifier
//!
//! This crate transpiles Jolt's verifier (stages 1-7, all sumchecks) into circuit code
//! for various proving backends. Currently supported: gnark (Go/Groth16).
//!
//! # SCOPE (maintainer-accepted; matches the pre-spongefish transpiler)
//!
//! - **Non-ZK proofs ONLY.** ZK/BlindFold proofs (`proof.zk_mode == true`) are
//!   rejected up front ([`narg_parser::NargParseError::ZkProofUnsupported`]).
//! - **Poseidon transcript sponge ONLY.** Requires the `transcript-poseidon`
//!   feature; the Blake2b/Keccak byte sponges used elsewhere in Jolt are
//!   intentionally unsupported here ([`pipeline::PipelineError::WrongSpongeFeature`]).
//! - **Field-aligned absorption.** Each 32-byte NARG word is deserialized to one
//!   `Fr` and absorbed as a single field element — proof scalars are NEVER
//!   byte-decomposed in-circuit. The 31-byte chunk rule applies only to genuine
//!   byte STRINGS (the domain separator, GT commitment bytes) and is applied
//!   OUTSIDE the circuit.
//!
//! This restores the scope of the pre-spongefish transpiler (commit
//! `f3de3c9160498abdd7452740b37869ecbc60f611`, where `raw_append_scalar`
//! absorbed each scalar as itself).
//!
//! # Architecture
//!
//! ```text
//! JoltProof (NARG byte-string + structural fields)
//!     ↓ symbolize_proof(): claims → vars; NARG → frames (narg_parser)
//! TranspilableVerifier stages 1–7 over SymbolicVerifierFs
//!     (frames → witness vars at their exact read positions;
//!      absorbs/squeezes → sponge-layout AST nodes)
//!     ↓
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
//! - Stage 7: HammingWeight and AdviceClaimReduction sumchecks
//!
//! Stage 8 (PCS/Dory) is NOT transpiled because pairing operations are too expensive
//! in-circuit. The PCS choice may change, so this is deferred.
//!
//! # Transcript Feature Flags
//!
//! `transcript-poseidon` is the only transcript feature: the symbolic sponge layout
//! (and the Go gadget) model Poseidon, the sole sponge that is efficient in-circuit.
//! Featureless builds compile, but `run_symbolic_pipeline` returns
//! `WrongSpongeFeature` — the transpiler must use the same transcript as proof
//! generation.
//!
//! # Module Overview
//!
//! - [`narg_parser`]: split the proof's NARG byte-string into self-delimiting frames
//! - [`gnark_codegen`]: AST → Go/gnark code generation with CSE
//! - [`symbolic_proof`]: symbolize the proof's structural parts (claims, configs)
//! - [`symbolic_traits`]: Trait implementations for MleAst transpilation
//!   - [`symbolic_traits::verifier_fs`]: spongefish `VerifierFs` for symbolic Fiat-Shamir
//!   - [`symbolic_traits::opening_accumulator`]: Symbolic opening accumulator
//!   - [`symbolic_traits::ast_commitment_scheme`]: Stub commitment scheme for transpilation
//!
//! # Usage
//!
//! See `main.rs` for the full transpilation pipeline, or use the library directly:
//!
//! ```ignore
//! use transpiler::symbolic_proof::{symbolize_proof, SymbolizedProof};
//! use transpiler::symbolic_traits::{FieldAlignedLayout, SymbolicVerifierFs};
//!
//! let SymbolizedProof { parsed_narg, accumulator, proof_data } =
//!     symbolize_proof(&real_proof, &mut var_alloc)?;
//! // ... build SymbolicVerifierFs over the frames, drive TranspilableVerifier stages ...
//! let circuit_code = gnark_codegen::generate_circuit_from_bundle(&bundle, "MyCircuit");
//! ```

pub mod ast_evaluator;
pub mod gnark_codegen;
pub mod narg_parser;
pub mod pipeline;
#[cfg(test)]
pub(crate) mod poseidon_model;
pub mod symbolic_proof;
pub mod symbolic_traits;
pub mod symbolize;

pub use gnark_codegen::{generate_circuit_from_bundle, sanitize_go_name};
pub use symbolic_proof::{symbolize_proof, VarAllocator};
pub use symbolic_traits::{AstCommitmentScheme, AstCurve, AstOpeningAccumulator};
