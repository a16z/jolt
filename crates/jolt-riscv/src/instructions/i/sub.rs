use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

/// RV64I SUB: `rd = rs1 - rs2` (wrapping).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(SubtractOperands, WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct Sub<T = ()>(pub T);
