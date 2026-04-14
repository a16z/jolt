//! RV64I system and synchronization instructions.
//!
//! These instructions produce side effects (syscalls, memory fences) that
//! are handled by the VM. Their `execute` returns 0 as a no-op.

use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

use crate::Instruction;

/// RV64I ECALL: environment call (syscall). Returns 0.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
pub struct Ecall;

impl Instruction for Ecall {
    #[inline]
    fn name(&self) -> &'static str {
        "ECALL"
    }

    #[inline]
    fn execute(&self, _x: u64, _y: u64) -> u64 {
        0
    }
}

/// RV64I EBREAK: breakpoint trap. Returns 0.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
pub struct Ebreak;

impl Instruction for Ebreak {
    #[inline]
    fn name(&self) -> &'static str {
        "EBREAK"
    }

    #[inline]
    fn execute(&self, _x: u64, _y: u64) -> u64 {
        0
    }
}

/// RV64I FENCE: memory ordering fence. Returns 0.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
pub struct Fence;

impl Instruction for Fence {
    #[inline]
    fn name(&self) -> &'static str {
        "FENCE"
    }

    #[inline]
    fn execute(&self, _x: u64, _y: u64) -> u64 {
        0
    }
}

/// No-operation pseudo-instruction. Returns 0.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[instruction(IsNoop)]
pub struct Noop;

impl Instruction for Noop {
    #[inline]
    fn name(&self) -> &'static str {
        "NOOP"
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
    fn system_instructions_return_zero() {
        assert_eq!(Ecall.execute(42, 99), 0);
        assert_eq!(Ebreak.execute(42, 99), 0);
        assert_eq!(Fence.execute(42, 99), 0);
        assert_eq!(Noop.execute(42, 99), 0);
    }
}
