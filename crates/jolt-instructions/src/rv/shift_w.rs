//! RV64 W-suffix shift instructions operating on the lower 32 bits
//! with sign-extension of the result to 64 bits.
//! Shift amount is masked to 5 bits (0..31).

use crate::opcodes;

define_instruction!(
    /// RV64I SLLW: 32-bit shift left logical, sign-extended to 64 bits.
    SllW, opcodes::SLLW, "SLLW",
    |x, y| ((x as u32) << (y & 31)) as i32 as i64 as u64,
    circuit: [WriteLookupOutputToRD],
    instruction: [LeftOperandIsRs1Value, RightOperandIsRs2Value],
);

define_instruction!(
    /// RV64I SLLIW: 32-bit shift left logical by immediate, sign-extended.
    SllIW, opcodes::SLLIW, "SLLIW",
    |x, y| ((x as u32) << (y & 31)) as i32 as i64 as u64,
    circuit: [WriteLookupOutputToRD],
    instruction: [LeftOperandIsRs1Value, RightOperandIsImm],
);

define_instruction!(
    /// RV64I SRLW: 32-bit shift right logical, sign-extended to 64 bits.
    SrlW, opcodes::SRLW, "SRLW",
    |x, y| ((x as u32) >> (y & 31)) as i32 as i64 as u64,
    circuit: [WriteLookupOutputToRD],
    instruction: [LeftOperandIsRs1Value, RightOperandIsRs2Value],
);

define_instruction!(
    /// RV64I SRLIW: 32-bit shift right logical by immediate, sign-extended.
    SrlIW, opcodes::SRLIW, "SRLIW",
    |x, y| ((x as u32) >> (y & 31)) as i32 as i64 as u64,
    circuit: [WriteLookupOutputToRD],
    instruction: [LeftOperandIsRs1Value, RightOperandIsImm],
);

define_instruction!(
    /// RV64I SRAW: 32-bit shift right arithmetic, sign-extended to 64 bits.
    SraW, opcodes::SRAW, "SRAW",
    |x, y| ((x as i32) >> (y & 31)) as i64 as u64,
    circuit: [WriteLookupOutputToRD],
    instruction: [LeftOperandIsRs1Value, RightOperandIsRs2Value],
);

define_instruction!(
    /// RV64I SRAIW: 32-bit shift right arithmetic by immediate, sign-extended.
    SraIW, opcodes::SRAIW, "SRAIW",
    |x, y| ((x as i32) >> (y & 31)) as i64 as u64,
    circuit: [WriteLookupOutputToRD],
    instruction: [LeftOperandIsRs1Value, RightOperandIsImm],
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Instruction;

    #[test]
    fn sllw_sign_extends() {
        // Shifting 1 left by 31 gives 0x8000_0000 which is negative as i32
        let result = SllW.execute(1, 31);
        assert_eq!(result, 0xFFFF_FFFF_8000_0000);
    }

    #[test]
    fn sllw_masks_to_5_bits() {
        assert_eq!(SllW.execute(1, 32), 1); // 32 & 31 = 0
    }

    #[test]
    fn srlw_basic() {
        assert_eq!(SrlW.execute(0x8000_0000, 31), 1);
    }

    #[test]
    fn sraw_sign_extends() {
        let neg = 0xFFFF_FFFF_8000_0000u64; // -2^31 sign-extended
        let result = SraW.execute(neg, 1);
        assert_eq!(result, 0xFFFF_FFFF_C000_0000); // -2^30 sign-extended
    }

    #[test]
    fn immediate_variants_match() {
        assert_eq!(SllW.execute(42, 5), SllIW.execute(42, 5));
        assert_eq!(SrlW.execute(42, 5), SrlIW.execute(42, 5));
        assert_eq!(SraW.execute(42, 5), SraIW.execute(42, 5));
    }
}
