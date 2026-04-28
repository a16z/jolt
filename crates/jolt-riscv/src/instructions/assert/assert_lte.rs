use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

/// Virtual ASSERT_LTE: returns 1 if `x <= y` (unsigned), 0 otherwise.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(Assert)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct AssertLte<T = ()>(pub T);
