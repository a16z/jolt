//! RV64I store instructions that mask a value to the appropriate width.
//!
//! In Jolt's execution model, `x` is the value to store and the instruction
//! truncates it to the target memory width.

use crate::opcodes;

define_instruction!(
    /// RV64I SB: store byte (lowest 8 bits).
    Sb, opcodes::SB, "SB",
    |x, _y| x & 0xFF,
    circuit: [Store],
);

define_instruction!(
    /// RV64I SH: store halfword (lowest 16 bits).
    Sh, opcodes::SH, "SH",
    |x, _y| x & 0xFFFF,
    circuit: [Store],
);

define_instruction!(
    /// RV64I SW: store word (lowest 32 bits).
    Sw, opcodes::SW, "SW",
    |x, _y| x & 0xFFFF_FFFF,
    circuit: [Store],
);

define_instruction!(
    /// RV64I SD: store doubleword (full 64 bits). Identity operation.
    Sd, opcodes::SD, "SD",
    |x, _y| x,
    circuit: [Store],
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Instruction;

    #[test]
    fn sb_masks_to_byte() {
        assert_eq!(Sb.execute(0xDEAD_BEEF, 0), 0xEF);
    }

    #[test]
    fn sh_masks_to_halfword() {
        assert_eq!(Sh.execute(0xDEAD_BEEF, 0), 0xBEEF);
    }

    #[test]
    fn sw_masks_to_word() {
        assert_eq!(Sw.execute(0xDEAD_BEEF_CAFE_BABE, 0), 0xCAFE_BABE);
    }

    #[test]
    fn sd_identity() {
        assert_eq!(Sd.execute(0xDEAD_BEEF_CAFE_BABE, 0), 0xDEAD_BEEF_CAFE_BABE);
    }
}
