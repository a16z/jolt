//! Gnark Go code generation from `jolt-ir` expressions.
//!
//! Provides [`GnarkEmitter`], an implementation of [`CircuitEmitter`](jolt_ir::CircuitEmitter) that
//! produces gnark `frontend.API` calls as Go code strings.
//!
//! # Usage
//!
//! ```
//! use jolt_ir::{ExprBuilder, CircuitEmitter};
//! use jolt_gnark::GnarkEmitter;
//!
//! let b = ExprBuilder::new();
//! let h = b.opening(0);
//! let gamma = b.challenge(0);
//! let expr = b.build(gamma * (h * h - h));
//!
//! let mut emitter = GnarkEmitter::new();
//! let root = expr.to_circuit(&mut emitter);
//!
//! let code = emitter.finish(&root);
//! assert!(code.contains("api.Mul"));
//! assert!(code.contains("api.Sub"));
//! ```
//!
//! # Variable naming
//!
//! By default, openings are `circuit.Opening_N` and challenges are
//! `circuit.Challenge_N`. Custom names can be provided via
//! [`GnarkEmitter::with_opening_name`] and [`GnarkEmitter::with_challenge_name`].
//!
//! # Constant handling
//!
//! Small constants (fitting in `i64`) are emitted as Go integer literals.
//! Larger values use `bigInt("...")` — the caller is responsible for defining
//! the `bigInt` helper in the generated Go file.

mod emitter;

pub use emitter::{sanitize_go_name, GnarkEmitter};
