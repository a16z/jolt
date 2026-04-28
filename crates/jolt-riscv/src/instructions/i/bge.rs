use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

/// RV64I BGE: branch if greater than or equal (signed).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value, Branch)]
pub struct Bge<T = ()>(pub T);
