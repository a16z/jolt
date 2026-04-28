use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

/// Virtual REV8W: byte-reverse within the lower 32 bits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(AddOperands, WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value)]
pub struct VirtualRev8W<T = ()>(pub T);
