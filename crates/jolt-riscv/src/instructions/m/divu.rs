use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

/// RV64M DIVU: unsigned division. Returns `u64::MAX` on division by zero.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct DivU<T = ()>(pub T);
