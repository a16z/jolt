use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

/// RV64M DIV: signed division with RISC-V overflow handling.
///
/// Special cases per the RISC-V spec:
/// - Division by zero returns `u64::MAX` (all bits set, i.e. -1 unsigned).
/// - `i64::MIN / -1` returns `i64::MIN` (overflow wraps).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct Div<T = ()>(pub T);
