use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

/// RV64M REM: signed remainder. Returns `x` on division by zero,
/// returns 0 when `x == i64::MIN && y == -1`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct Rem<T = ()>(pub T);
