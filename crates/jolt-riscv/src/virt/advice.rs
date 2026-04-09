//! Virtual advice and I/O instructions.
//!
//! These opcodes are runtime-managed by the tracer/emulator. Their
//! [`Instruction::execute`](crate::Instruction::execute) implementations
//! return placeholder zero values so the registry can still expose a uniform
//! trait object API for opcode/flag lookup.

use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

use crate::Instruction;

/// Virtual ADVICE: runtime-provided advice value. `execute` returns a placeholder 0.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(Advice, WriteLookupOutputToRD)]
pub struct VirtualAdvice;

impl Instruction for VirtualAdvice {
    #[inline]
    fn name(&self) -> &'static str {
        "VIRTUAL_ADVICE"
    }

    #[inline]
    fn execute(&self, _x: u64, _y: u64) -> u64 {
        0
    }
}

/// Virtual ADVICE_LEN: advice-tape length query. `execute` returns a placeholder 0.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(Advice, WriteLookupOutputToRD)]
pub struct VirtualAdviceLen;

impl Instruction for VirtualAdviceLen {
    #[inline]
    fn name(&self) -> &'static str {
        "VIRTUAL_ADVICE_LEN"
    }

    #[inline]
    fn execute(&self, _x: u64, _y: u64) -> u64 {
        0
    }
}

/// Virtual ADVICE_LOAD: advice-tape read. `execute` returns a placeholder 0.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(Advice, WriteLookupOutputToRD)]
pub struct VirtualAdviceLoad;

impl Instruction for VirtualAdviceLoad {
    #[inline]
    fn name(&self) -> &'static str {
        "VIRTUAL_ADVICE_LOAD"
    }

    #[inline]
    fn execute(&self, _x: u64, _y: u64) -> u64 {
        0
    }
}

/// Virtual HOST_IO: host I/O side-effect instruction. `execute` returns a placeholder 0.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
pub struct VirtualHostIO;

impl Instruction for VirtualHostIO {
    #[inline]
    fn name(&self) -> &'static str {
        "VIRTUAL_HOST_IO"
    }

    #[inline]
    fn execute(&self, _x: u64, _y: u64) -> u64 {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_managed_instructions_use_zero_placeholders() {
        assert_eq!(VirtualAdvice.execute(0, 0), 0);
        assert_eq!(VirtualAdviceLen.execute(0, 0), 0);
        assert_eq!(VirtualAdviceLoad.execute(0, 0), 0);
        assert_eq!(VirtualHostIO.execute(0, 0), 0);
    }
}
