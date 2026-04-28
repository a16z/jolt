use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

/// RV64M DIVW: 32-bit signed division, sign-extended to 64 bits.
///
/// Division by zero returns `u64::MAX`. Overflow (`i32::MIN / -1`) returns `i32::MIN` sign-extended.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct DivW<T = ()>(pub T);
