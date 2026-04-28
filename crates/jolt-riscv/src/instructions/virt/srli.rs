use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

/// Virtual SRLI: logical right shift using a bitmask immediate.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsImm)]
pub struct VirtualSrli<T = ()>(pub T);
