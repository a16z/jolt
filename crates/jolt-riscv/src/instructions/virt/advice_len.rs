use crate::jolt_instruction;

jolt_instruction!(
    /// Virtual ADVICE_LEN: advice-tape length query.
    VirtualAdviceLen,
    circuit flags: [Advice, WriteLookupOutputToRD],
    instruction flags: []
);
