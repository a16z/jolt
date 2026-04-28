use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

/// RV64M REMU: unsigned remainder. Returns `x` on division by zero.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct RemU<T = ()>(pub T);
