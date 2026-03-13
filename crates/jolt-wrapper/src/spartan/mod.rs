//! Spartan R1CS backend for the [`AstEmitter`](crate::AstEmitter) trait.
//!
//! Converts the wrapper's AST into R1CS constraints (`A · B = C`) suitable for
//! proving with Spartan or any R1CS-based proof system.
//!
//! # Wire representation
//!
//! Each wire is a [`LinearCombination<F>`](jolt_ir::LinearCombination). This
//! means **additions are free** (LCs merge without creating constraints) while
//! **multiplications emit constraints** (each mul allocates an auxiliary
//! variable). This minimizes constraint count.
//!
//! # Example
//!
//! ```
//! use jolt_wrapper::arena::ArenaSession;
//! use jolt_wrapper::symbolic::SymbolicField;
//! use jolt_wrapper::bundle::VarAllocator;
//! use jolt_wrapper::spartan::SpartanAstEmitter;
//! use jolt_field::{Field, Fr};
//!
//! let _session = ArenaSession::new();
//!
//! let x = SymbolicField::variable(0, "x");
//! let y = SymbolicField::variable(1, "y");
//!
//! let mut alloc = VarAllocator::new();
//! alloc.input("x");
//! alloc.input("y");
//! alloc.assert_zero((x * y).into_edge());
//! let bundle = alloc.finish();
//!
//! let mut emitter = SpartanAstEmitter::<Fr>::new();
//! bundle.emit(&mut emitter);
//! let circuit = emitter.finish();
//!
//! // Build witness and verify satisfaction
//! let witness = circuit.build_witness(&[(0, Fr::from_u64(0)), (1, Fr::from_u64(7))]);
//! assert!(circuit.is_satisfied(&witness));
//! ```

mod emitter;

pub use emitter::{InputMapping, SpartanAstEmitter, SpartanCircuit};
