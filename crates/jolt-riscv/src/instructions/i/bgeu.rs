use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

/// RV64I BGEU: branch if greater than or equal (unsigned).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value, Branch)]
pub struct BgeU<T = ()>(pub T);
