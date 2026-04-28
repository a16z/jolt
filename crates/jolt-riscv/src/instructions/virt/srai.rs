use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

/// Virtual SRAI: arithmetic right shift using a bitmask immediate.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsImm)]
pub struct VirtualSrai<T = ()>(pub T);
