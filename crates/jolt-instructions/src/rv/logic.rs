//! RV64I bitwise logic instructions.

use crate::opcodes;

define_instruction!(
    /// RV64I AND: bitwise AND of two registers.
    And, opcodes::AND, "AND",
    |x, y| x & y,
    circuit: [WriteLookupOutputToRD],
    instruction: [LeftOperandIsRs1Value, RightOperandIsRs2Value],
    table: And,
);

define_instruction!(
    /// RV64I ANDI: bitwise AND with sign-extended immediate.
    AndI, opcodes::ANDI, "ANDI",
    |x, y| x & y,
    circuit: [WriteLookupOutputToRD],
    instruction: [LeftOperandIsRs1Value, RightOperandIsImm],
    table: And,
);

define_instruction!(
    /// RV64I OR: bitwise OR of two registers.
    Or, opcodes::OR, "OR",
    |x, y| x | y,
    circuit: [WriteLookupOutputToRD],
    instruction: [LeftOperandIsRs1Value, RightOperandIsRs2Value],
    table: Or,
);

define_instruction!(
    /// RV64I ORI: bitwise OR with sign-extended immediate.
    OrI, opcodes::ORI, "ORI",
    |x, y| x | y,
    circuit: [WriteLookupOutputToRD],
    instruction: [LeftOperandIsRs1Value, RightOperandIsImm],
    table: Or,
);

define_instruction!(
    /// RV64I XOR: bitwise exclusive OR of two registers.
    Xor, opcodes::XOR, "XOR",
    |x, y| x ^ y,
    circuit: [WriteLookupOutputToRD],
    instruction: [LeftOperandIsRs1Value, RightOperandIsRs2Value],
    table: Xor,
);

define_instruction!(
    /// RV64I XORI: bitwise exclusive OR with sign-extended immediate.
    XorI, opcodes::XORI, "XORI",
    |x, y| x ^ y,
    circuit: [WriteLookupOutputToRD],
    instruction: [LeftOperandIsRs1Value, RightOperandIsImm],
    table: Xor,
);

define_instruction!(
    /// Zbb ANDN: bitwise AND-NOT. `rd = rs1 & ~rs2`.
    Andn, opcodes::ANDN, "ANDN",
    |x, y| x & !y,
    circuit: [WriteLookupOutputToRD],
    instruction: [LeftOperandIsRs1Value, RightOperandIsRs2Value],
    table: Andn,
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Instruction;

    #[test]
    fn and_basic() {
        assert_eq!(And.execute(0xFF00, 0x0FF0), 0x0F00);
        assert_eq!(And.execute(u64::MAX, 0), 0);
    }

    #[test]
    fn or_basic() {
        assert_eq!(Or.execute(0xFF00, 0x00FF), 0xFFFF);
        assert_eq!(Or.execute(0, 0), 0);
    }

    #[test]
    fn xor_basic() {
        assert_eq!(Xor.execute(0xFF, 0xFF), 0);
        assert_eq!(Xor.execute(0xFF, 0x00), 0xFF);
    }

    #[test]
    fn immediate_variants_match() {
        assert_eq!(And.execute(0xAB, 0xCD), AndI.execute(0xAB, 0xCD));
        assert_eq!(Or.execute(0xAB, 0xCD), OrI.execute(0xAB, 0xCD));
        assert_eq!(Xor.execute(0xAB, 0xCD), XorI.execute(0xAB, 0xCD));
    }

    #[test]
    fn andn_basic() {
        assert_eq!(Andn.execute(0xFF, 0x0F), 0xF0);
        assert_eq!(Andn.execute(0xFF, 0xFF), 0);
        assert_eq!(Andn.execute(0xFF, 0), 0xFF);
    }
}
