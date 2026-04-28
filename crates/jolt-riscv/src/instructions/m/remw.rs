use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

/// RV64M REMW: 32-bit signed remainder, sign-extended to 64 bits.
/// Returns `x` (truncated to 32 bits, sign-extended) on division by zero.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct RemW<T = ()>(pub T);
