use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

/// Virtual XOR then rotate right word (32-bit) by 16 bits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct VirtualXorRotW16<T = ()>(pub T);
