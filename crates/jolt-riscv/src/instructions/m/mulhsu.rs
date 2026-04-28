use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

/// RV64M MULHSU: signed×unsigned multiply, upper 64 bits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct MulHSU<T = ()>(pub T);
