//! Verifier wrapping infrastructure for Jolt.
//!
//! Provides symbolic execution, AST capture, and pluggable codegen backends for
//! transpiling the Jolt verifier into external proof systems (gnark/Groth16,
//! Spartan+HyperKZG, etc.).
//!
//! # Architecture
//!
//! ```text
//! Symbolic Execution (SymbolicField + PoseidonSymbolicTranscript)
//!      ↓ records operations
//! AST Arena (Node graph)
//!      ↓ captured as
//! AstBundle (constraints + inputs — backend-agnostic)
//!      ↓ consumed by
//! AstEmitter trait (pluggable codegen backend)
//!      ├── gnark/ → GnarkAstEmitter → Go circuit code (Groth16)
//!      └── future backends...
//! ```
//!
//! # Usage
//!
//! ```
//! use jolt_wrapper::arena::ArenaSession;
//! use jolt_wrapper::symbolic::SymbolicField;
//! use jolt_wrapper::bundle::VarAllocator;
//! use jolt_wrapper::gnark::{GnarkAstEmitter, generate_go_file, GoFileConfig};
//! use jolt_field::Field;
//!
//! // 1. Create arena session
//! let _session = ArenaSession::new();
//!
//! // 2. Symbolic execution
//! let x = SymbolicField::variable(0, "x");
//! let y = SymbolicField::variable(1, "y");
//! let constraint = x * y - SymbolicField::from_u64(42);
//!
//! // 3. Capture as bundle
//! let mut alloc = VarAllocator::new();
//! alloc.input("x");
//! alloc.input("y");
//! alloc.assert_zero(constraint.into_edge());
//! let bundle = alloc.finish();
//!
//! // 4. Emit Go code
//! let code = generate_go_file(&bundle, &GoFileConfig::default());
//! assert!(code.contains("api.Mul("));
//! ```

pub mod arena;
pub mod ast_emitter;
pub mod bundle;
pub mod gnark;
pub mod scalar_ops;
pub mod symbolic;
pub mod transcript;
pub mod tunneling;

// Re-exports for convenience
pub use arena::ArenaSession;
pub use ast_emitter::AstEmitter;
pub use bundle::AstBundle;
pub use symbolic::SymbolicField;
pub use transcript::PoseidonSymbolicTranscript;
