//! RV64I load instructions that extract and extend bytes from a memory word.
//!
//! In Jolt's execution model, `x` contains the loaded value from memory
//! and the instruction performs sign/zero extension to 64 bits.

use crate::opcodes;

define_instruction!(
    /// RV64I LB: load byte, sign-extended to 64 bits.
    Lb, opcodes::LB, "LB",
    |x, _y| (x as i8) as i64 as u64,
    circuit: [Load],
);

define_instruction!(
    /// RV64I LBU: load byte, zero-extended to 64 bits.
    Lbu, opcodes::LBU, "LBU",
    |x, _y| x & 0xFF,
    circuit: [Load],
);

define_instruction!(
    /// RV64I LH: load halfword (16 bits), sign-extended to 64 bits.
    Lh, opcodes::LH, "LH",
    |x, _y| (x as i16) as i64 as u64,
    circuit: [Load],
);

define_instruction!(
    /// RV64I LHU: load halfword, zero-extended to 64 bits.
    Lhu, opcodes::LHU, "LHU",
    |x, _y| x & 0xFFFF,
    circuit: [Load],
);

define_instruction!(
    /// RV64I LW: load word (32 bits), sign-extended to 64 bits.
    Lw, opcodes::LW, "LW",
    |x, _y| (x as i32) as i64 as u64,
    circuit: [Load],
);

define_instruction!(
    /// RV64I LWU: load word, zero-extended to 64 bits.
    Lwu, opcodes::LWU, "LWU",
    |x, _y| x & 0xFFFF_FFFF,
    circuit: [Load],
);

define_instruction!(
    /// RV64I LD: load doubleword (64 bits). Identity operation.
    Ld, opcodes::LD, "LD",
    |x, _y| x,
    circuit: [Load],
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Instruction;

    #[test]
    fn lb_sign_extends() {
        assert_eq!(Lb.execute(0x80, 0), 0xFFFF_FFFF_FFFF_FF80); // -128
        assert_eq!(Lb.execute(0x7F, 0), 0x7F); // +127
    }

    #[test]
    fn lbu_zero_extends() {
        assert_eq!(Lbu.execute(0x80, 0), 0x80);
        assert_eq!(Lbu.execute(0xFF_FF, 0), 0xFF);
    }

    #[test]
    fn lh_sign_extends() {
        assert_eq!(Lh.execute(0x8000, 0), 0xFFFF_FFFF_FFFF_8000);
        assert_eq!(Lh.execute(0x7FFF, 0), 0x7FFF);
    }

    #[test]
    fn lhu_zero_extends() {
        assert_eq!(Lhu.execute(0x8000, 0), 0x8000);
    }

    #[test]
    fn lw_sign_extends() {
        assert_eq!(Lw.execute(0x8000_0000, 0), 0xFFFF_FFFF_8000_0000);
        assert_eq!(Lw.execute(0x7FFF_FFFF, 0), 0x7FFF_FFFF);
    }

    #[test]
    fn lwu_zero_extends() {
        assert_eq!(Lwu.execute(0x8000_0000, 0), 0x8000_0000);
    }

    #[test]
    fn ld_identity() {
        assert_eq!(Ld.execute(0xDEAD_BEEF_CAFE_BABE, 0), 0xDEAD_BEEF_CAFE_BABE);
    }
}
