use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

/// Virtual ASSERT_MULU_NO_OVERFLOW: checks unsigned multiply doesn't overflow.
/// Returns 1 if the upper XLEN bits of `x * y` are all zero.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(MultiplyOperands, Assert)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct AssertMulUNoOverflow<T = ()>(pub T);
