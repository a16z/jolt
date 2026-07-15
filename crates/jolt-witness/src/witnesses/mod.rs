//! Atomic witness values: one row-free newtype per witness.
//!
//! Each newtype carries the witness's natural scalar type; its derivation
//! from a trace row lives in exactly one `Extract` impl in the trace-backend
//! module. File grouping here is packaging convenience, not taxonomy —
//! nothing dispatches on modules.

mod flags;
mod lookups;
mod operands;
mod pc;
mod ram;
mod registers;

pub use flags::{
    InstructionFlag, InstructionRafFlag, LookupTableFlag, NextIsFirstInSequence, NextIsNoop,
    NextIsVirtual, OpFlag, ShouldBranch, ShouldJump,
};
pub use lookups::LookupOutput;
pub use operands::{
    Imm, LeftInstructionInput, LeftLookupOperand, Product, RightInstructionInput,
    RightLookupOperand,
};
pub use pc::{NextPc, NextUnexpandedPc, Pc, UnexpandedPc};
pub use ram::{RamAddress, RamHammingWeight, RamReadValue, RamWriteValue};
pub use registers::{RdWriteValue, Rs1Value, Rs2Value};
