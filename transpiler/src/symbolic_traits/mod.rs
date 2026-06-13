//! Symbolic trait implementations for MleAst transpilation.
//!
//! This module contains implementations of jolt-core traits for `MleAst`, the symbolic
//! field type used during transpilation. These implementations allow the Jolt verifier
//! to run symbolically, recording operations as an AST instead of computing concrete values.
//!
//! # Modules
//!
//! - [`commitment_scheme`]: `CommitmentScheme` implementation (`AstCommitmentScheme`)
//! - [`opening_accumulator`]: `OpeningAccumulator` implementation (`AstOpeningAccumulator`)
//! - [`verifier_fs`]: spongefish `VerifierFs` implementation (`SymbolicVerifierFs`)
//!
//! # Usage
//!
//! These types are used together to run the Jolt verifier symbolically:
//!
//! ```ignore
//! use transpiler::symbolic_traits::{
//!     AstCommitmentScheme, AstOpeningAccumulator, FieldAlignedLayout, SymbolicVerifierFs,
//! };
//! use jolt_core::zkvm::transpilable_verifier::TranspilableVerifier;
//!
//! // See `transpiler::pipeline::run_symbolic_pipeline` for the real driver.
//! let layout = FieldAlignedLayout::new(b"Jolt", &instance);
//! let mut fs = SymbolicVerifierFs::new(layout, parsed_narg, var_alloc);
//! let mut verifier = TranspilableVerifier::<MleAst, AstCurve, AstCommitmentScheme, AstOpeningAccumulator>::new(...)?;
//! verifier.verify_stage1(&mut fs)?; // ... stages 1-7, recording the AST
//! ```

pub mod ast_commitment_scheme;
pub mod ast_curve;
pub mod opening_accumulator;
pub mod verifier_fs;

pub use ast_commitment_scheme::AstCommitmentScheme;
pub use ast_curve::AstCurve;
pub use opening_accumulator::AstOpeningAccumulator;
pub use verifier_fs::{FieldAlignedLayout, FrameLabel, SymbolicSpongeLayout, SymbolicVerifierFs};
