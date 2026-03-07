//! RV64I shift instructions operating on full 64-bit values.
//! Shift amount is masked to 6 bits (0..63) per the RISC-V spec.

use crate::opcodes;

define_instruction!(
    /// RV64I SLL: shift left logical. Shift amount from lower 6 bits of `y`.
    Sll, opcodes::SLL, "SLL",
    |x, y| x << (y & 63),
    circuit: [WriteLookupOutputToRD],
    instruction: [LeftOperandIsRs1Value, RightOperandIsRs2Value],
);

define_instruction!(
    /// RV64I SLLI: shift left logical by immediate. Immediate already masked.
    SllI, opcodes::SLLI, "SLLI",
    |x, y| x << (y & 63),
    circuit: [WriteLookupOutputToRD],
    instruction: [LeftOperandIsRs1Value, RightOperandIsImm],
);

define_instruction!(
    /// RV64I SRL: shift right logical. Shift amount from lower 6 bits of `y`.
    Srl, opcodes::SRL, "SRL",
    |x, y| x >> (y & 63),
    circuit: [WriteLookupOutputToRD],
    instruction: [LeftOperandIsRs1Value, RightOperandIsRs2Value],
);

define_instruction!(
    /// RV64I SRLI: shift right logical by immediate.
    SrlI, opcodes::SRLI, "SRLI",
    |x, y| x >> (y & 63),
    circuit: [WriteLookupOutputToRD],
    instruction: [LeftOperandIsRs1Value, RightOperandIsImm],
);

define_instruction!(
    /// RV64I SRA: shift right arithmetic. Preserves sign bit.
    Sra, opcodes::SRA, "SRA",
    |x, y| ((x as i64) >> (y & 63)) as u64,
    circuit: [WriteLookupOutputToRD],
    instruction: [LeftOperandIsRs1Value, RightOperandIsRs2Value],
);

define_instruction!(
    /// RV64I SRAI: shift right arithmetic by immediate.
    SraI, opcodes::SRAI, "SRAI",
    |x, y| ((x as i64) >> (y & 63)) as u64,
    circuit: [WriteLookupOutputToRD],
    instruction: [LeftOperandIsRs1Value, RightOperandIsImm],
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Instruction;

    #[test]
    fn sll_basic() {
        assert_eq!(Sll.execute(1, 10), 1024);
        assert_eq!(Sll.execute(1, 63), 1 << 63);
    }

    #[test]
    fn sll_masks_shift_amount() {
        // Shift by 64 should wrap to shift by 0
        assert_eq!(Sll.execute(1, 64), 1);
    }

    #[test]
    fn srl_basic() {
        assert_eq!(Srl.execute(1024, 10), 1);
        assert_eq!(Srl.execute(u64::MAX, 63), 1);
    }

    #[test]
    fn sra_sign_extends() {
        let neg = (-1024i64) as u64;
        let result = Sra.execute(neg, 4);
        assert_eq!(result, (-64i64) as u64);
    }

    #[test]
    fn sra_positive() {
        assert_eq!(Sra.execute(1024, 4), 64);
    }

    #[test]
    fn immediate_variants_match() {
        assert_eq!(Sll.execute(42, 5), SllI.execute(42, 5));
        assert_eq!(Srl.execute(42, 5), SrlI.execute(42, 5));
        assert_eq!(Sra.execute(42, 5), SraI.execute(42, 5));
    }
}
