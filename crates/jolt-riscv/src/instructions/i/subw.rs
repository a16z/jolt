use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

/// RV64I SUBW: 32-bit subtract, sign-extended to 64 bits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(SubtractOperands, WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct SubW<T = ()>(pub T);
