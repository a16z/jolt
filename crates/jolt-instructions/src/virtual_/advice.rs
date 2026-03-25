//! Virtual advice and I/O instructions.

use crate::opcodes;

define_instruction!(
    /// Virtual ADVICE: non-deterministic advice value. The prover supplies the value.
    VirtualAdvice, opcodes::VIRTUAL_ADVICE, "VIRTUAL_ADVICE",
    |_x, _y| 0,
    circuit: [Advice, WriteLookupOutputToRD],
    table: RangeCheck,
);

define_instruction!(
    /// Virtual ADVICE_LEN: returns the length of the advice data.
    VirtualAdviceLen, opcodes::VIRTUAL_ADVICE_LEN, "VIRTUAL_ADVICE_LEN",
    |_x, _y| 0,
    circuit: [Advice, WriteLookupOutputToRD],
    table: RangeCheck,
);

define_instruction!(
    /// Virtual ADVICE_LOAD: loads a value from the advice tape.
    VirtualAdviceLoad, opcodes::VIRTUAL_ADVICE_LOAD, "VIRTUAL_ADVICE_LOAD",
    |_x, _y| 0,
    circuit: [Advice, WriteLookupOutputToRD],
    table: RangeCheck,
);

define_instruction!(
    /// Virtual HOST_IO: host I/O operation. Returns 0.
    VirtualHostIO, opcodes::VIRTUAL_HOST_IO, "VIRTUAL_HOST_IO",
    |_x, _y| 0,
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Instruction;

    #[test]
    fn advice_returns_zero() {
        assert_eq!(VirtualAdvice.execute(0, 0), 0);
        assert_eq!(VirtualAdviceLen.execute(0, 0), 0);
        assert_eq!(VirtualAdviceLoad.execute(0, 0), 0);
        assert_eq!(VirtualHostIO.execute(0, 0), 0);
    }
}
