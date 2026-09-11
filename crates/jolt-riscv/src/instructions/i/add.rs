use crate::jolt_instruction;

jolt_instruction!(
    /// RV64I ADD: `rd = rs1 + rs2` (wrapping).
    Add,
    circuit flags: [
        AddOperands,
        WriteLookupOutputToRD,
        #[cfg(feature = "implicit-carry")]
        ProducesCarry,
    ],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsRs2Value]
);
