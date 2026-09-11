//! Implicit-carry arithmetic instructions.
//!
//! `ADDC` and `MULC` consume the previous row's implicit carry (the committed
//! `Carry` column) and, like `ADD` and `MUL`, export the high 64 bits of the
//! true arithmetic result as the next row's carry. The carry is
//! non-architectural: it is not part of the memory-checked register file.
//! See the implicit-carry spec in <https://github.com/a16z/jolt/issues/1710>.

use crate::jolt_instruction;

jolt_instruction!(
    /// Jolt ADDC: `rd = low_64(rs1 + rs2 + carry)`, carry-out = `high_64(rs1 + rs2 + carry)`.
    AddC,
    circuit flags: [
        AddOperands,
        WriteLookupOutputToRD,
        UsesCarry,
        ProducesCarry,
    ],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsRs2Value]
);

jolt_instruction!(
    /// Jolt MULC: `rd = low_64(rs1 * rs2 + carry)`, carry-out = `high_64(rs1 * rs2 + carry)`,
    /// over the unsigned 128-bit widening of the raw 64-bit words.
    MulC,
    circuit flags: [
        MultiplyOperands,
        WriteLookupOutputToRD,
        UsesCarry,
        ProducesCarry,
    ],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsRs2Value]
);
