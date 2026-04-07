//! RV64I system and synchronization instructions.
//!
//! These instructions produce side effects (syscalls, memory fences) that
//! are handled by the VM. Their `execute` returns 0 as a no-op.

define_instruction!(
    /// RV64I ECALL: environment call (syscall). Returns 0.
    Ecall, "ECALL",
    |_x, _y| 0,
);

define_instruction!(
    /// RV64I EBREAK: breakpoint trap. Returns 0.
    Ebreak, "EBREAK",
    |_x, _y| 0,
);

define_instruction!(
    /// RV64I FENCE: memory ordering fence. Returns 0.
    Fence, "FENCE",
    |_x, _y| 0,
);

define_instruction!(
    /// No-operation pseudo-instruction. Returns 0.
    Noop, "NOOP",
    |_x, _y| 0,
    instruction: [IsNoop],
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Instruction;

    #[test]
    fn system_instructions_return_zero() {
        assert_eq!(Ecall.execute(42, 99), 0);
        assert_eq!(Ebreak.execute(42, 99), 0);
        assert_eq!(Fence.execute(42, 99), 0);
        assert_eq!(Noop.execute(42, 99), 0);
    }
}
