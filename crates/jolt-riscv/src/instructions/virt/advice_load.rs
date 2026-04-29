use crate::jolt_instruction;

jolt_instruction!(
    /// Virtual ADVICE_LOAD: advice-tape read.
    VirtualAdviceLoad,
    circuit flags: [Advice, WriteLookupOutputToRD],
    instruction flags: []
);
