use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

/// Virtual SRA: arithmetic right shift using a bitmask-encoded shift amount.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct VirtualSra<T = ()>(pub T);
