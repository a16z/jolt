use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

/// Virtual CHANGE_DIVISOR_W: 32-bit version of change divisor.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct VirtualChangeDivisorW<T = ()>(pub T);
