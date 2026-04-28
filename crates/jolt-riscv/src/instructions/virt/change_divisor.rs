use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

/// Virtual CHANGE_DIVISOR: transforms divisor for signed division overflow.
/// Returns the divisor unchanged, unless dividend == MIN && divisor == -1,
/// in which case returns 1 to avoid overflow.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct VirtualChangeDivisor<T = ()>(pub T);
