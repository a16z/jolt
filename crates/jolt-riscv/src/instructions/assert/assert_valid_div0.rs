use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

/// Virtual ASSERT_VALID_DIV0: validates `(divisor, quotient)` for division-by-zero handling.
/// Returns 1 if the divisor is nonzero, or if the divisor is 0 and the quotient is MAX.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(Assert)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct AssertValidDiv0<T = ()>(pub T);
