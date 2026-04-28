use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

/// RV64M DIVUW: 32-bit unsigned division, sign-extended to 64 bits.
/// Returns `u64::MAX` on division by zero.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct DivUW<T = ()>(pub T);
