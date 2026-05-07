use crate::jolt_instruction;

jolt_instruction!(
    /// Virtual ADVICE: runtime-provided advice value.
    VirtualAdvice,
    circuit flags: [Advice, WriteLookupOutputToRD],
    instruction flags: []
);
