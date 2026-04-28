use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

/// RV64M MULH: signed×signed multiply, upper 64 bits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct MulH<T = ()>(pub T);
