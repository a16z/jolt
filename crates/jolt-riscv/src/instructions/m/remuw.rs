use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

/// RV64M REMUW: 32-bit unsigned remainder, sign-extended to 64 bits.
/// Returns `x` (truncated to 32 bits, sign-extended) on division by zero.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct RemUW<T = ()>(pub T);
