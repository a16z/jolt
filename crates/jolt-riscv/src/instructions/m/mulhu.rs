use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

/// RV64M MULHU: unsigned×unsigned multiply, upper 64 bits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(MultiplyOperands, WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct MulHU<T = ()>(pub T);
