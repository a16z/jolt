//! Virtual byte manipulation instructions.

use crate::opcodes;

define_instruction!(
    /// Virtual REV8W: byte-reverse within the lower 32 bits.
    VirtualRev8W, opcodes::VIRTUAL_REV8W, "VIRTUAL_REV8W",
    |x, _y| {
        let w = x as u32;
        w.swap_bytes() as u64
    },
    circuit: [WriteLookupOutputToRD],
    instruction: [LeftOperandIsRs1Value],
    table: VirtualRev8W,
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Instruction;

    #[test]
    fn rev8w_basic() {
        assert_eq!(VirtualRev8W.execute(0x0102_0304, 0), 0x0403_0201);
    }

    #[test]
    fn rev8w_zero() {
        assert_eq!(VirtualRev8W.execute(0, 0), 0);
    }
}
