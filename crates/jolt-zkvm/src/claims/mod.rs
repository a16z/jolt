//! IR-based claim definitions for all Jolt sumcheck instances.
//!
//! Re-exports from [`jolt_ir::zkvm::claims`] — the single source of truth.
//! See that module for documentation on each claim definition.

pub use jolt_ir::zkvm::claims::instruction;
pub use jolt_ir::zkvm::claims::ram;
pub use jolt_ir::zkvm::claims::reductions;
pub use jolt_ir::zkvm::claims::registers;
pub use jolt_ir::zkvm::claims::spartan;
