use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

/// RV64I SRLI: shift right logical by immediate.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsImm)]
pub struct SrlI<T = ()>(pub T);
