use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

/// Virtual ASSERT_VALID_UNSIGNED_REMAINDER: validates unsigned remainder.
/// Returns 1 if divisor is 0 or remainder < divisor.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(Assert)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct AssertValidUnsignedRemainder<T = ()>(pub T);
