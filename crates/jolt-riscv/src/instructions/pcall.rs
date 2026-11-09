//! PCALL virtual rows. Proving them is not supported yet.
use crate::jolt_instruction;

jolt_instruction!(PcallCall, circuit flags: [Assert], instruction flags: [LeftOperandIsRs1Value, RightOperandIsImm]);
jolt_instruction!(PcallRead, circuit flags: [Load], instruction flags: []);
jolt_instruction!(PcallWrite, circuit flags: [Store, Advice], instruction flags: []);
jolt_instruction!(PcallReturn, circuit flags: [Assert], instruction flags: [LeftOperandIsRs1Value, RightOperandIsImm]);
