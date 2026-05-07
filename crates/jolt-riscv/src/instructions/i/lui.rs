use crate::jolt_instruction;

jolt_instruction!(
    /// RV64I LUI: load upper immediate. Result is the immediate value itself.
    Lui,
    circuit flags: [AddOperands, WriteLookupOutputToRD],
    instruction flags: [RightOperandIsImm]
);
