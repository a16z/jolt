use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

/// Virtual ROTRI: rotate right using a bitmask immediate.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsImm)]
pub struct VirtualRotri<T = ()>(pub T);
