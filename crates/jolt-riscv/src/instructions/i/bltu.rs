use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

/// RV64I BLTU: branch if less than (unsigned).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value, Branch)]
pub struct BltU<T = ()>(pub T);
