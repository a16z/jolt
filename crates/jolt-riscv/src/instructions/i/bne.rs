use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

/// RV64I BNE: branch if not equal. Returns 1 when `rs1 != rs2`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value, Branch)]
pub struct Bne<T = ()>(pub T);
