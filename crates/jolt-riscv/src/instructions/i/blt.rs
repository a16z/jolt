use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

/// RV64I BLT: branch if less than (signed).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value, Branch)]
pub struct Blt<T = ()>(pub T);
