use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

/// RV64M MULW: 32-bit multiply, sign-extended to 64 bits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(MultiplyOperands, WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct MulW<T = ()>(pub T);
