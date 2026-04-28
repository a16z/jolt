use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

/// Virtual ROTRIW: 32-bit rotate right using a bitmask immediate, zero-extended to 64 bits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsImm)]
pub struct VirtualRotriw<T = ()>(pub T);
