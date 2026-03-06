//! Gnark/Groth16 codegen backend.
//!
//! This module contains:
//!
//! - [`GnarkEmitter`] — `CircuitEmitter` impl for `jolt-ir` expressions
//! - [`GnarkAstEmitter`] — `AstEmitter` impl for arena-based AST
//! - [`MemoizedCodeGen`] — two-phase CSE + Go codegen
//! - [`go_file`] — complete Go circuit file generation

pub mod ast_emitter;
pub mod codegen;
pub mod emitter;
pub mod go_file;

pub use ast_emitter::GnarkAstEmitter;
pub use codegen::MemoizedCodeGen;
pub use emitter::{sanitize_go_name, GnarkEmitter};
pub use go_file::{generate_go_file, GoFileConfig};
