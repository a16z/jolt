//! Virtual advice and I/O instructions.
//!
//! These opcodes are runtime-managed by the tracer/emulator. Their
//! [`Instruction::execute`](crate::Instruction::execute) implementations
//! return placeholder zero values so the registry can still expose a uniform
//! trait object API for opcode/flag lookup.

define_instruction!(
    /// Virtual ADVICE: runtime-provided advice value. `execute` returns a placeholder 0.
    VirtualAdvice, "VIRTUAL_ADVICE",
    |_x, _y| 0,
    circuit: [Advice, WriteLookupOutputToRD],
);

define_instruction!(
    /// Virtual ADVICE_LEN: advice-tape length query. `execute` returns a placeholder 0.
    VirtualAdviceLen, "VIRTUAL_ADVICE_LEN",
    |_x, _y| 0,
    circuit: [Advice, WriteLookupOutputToRD],
);

define_instruction!(
    /// Virtual ADVICE_LOAD: advice-tape read. `execute` returns a placeholder 0.
    VirtualAdviceLoad, "VIRTUAL_ADVICE_LOAD",
    |_x, _y| 0,
    circuit: [Advice, WriteLookupOutputToRD],
);

define_instruction!(
    /// Virtual HOST_IO: host I/O side-effect instruction. `execute` returns a placeholder 0.
    VirtualHostIO, "VIRTUAL_HOST_IO",
    |_x, _y| 0,
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Instruction;

    #[test]
    fn runtime_managed_instructions_use_zero_placeholders() {
        assert_eq!(VirtualAdvice.execute(0, 0), 0);
        assert_eq!(VirtualAdviceLen.execute(0, 0), 0);
        assert_eq!(VirtualAdviceLoad.execute(0, 0), 0);
        assert_eq!(VirtualHostIO.execute(0, 0), 0);
    }
}
