use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

/// RV64I SLTIU: set if less than immediate (unsigned).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsImm)]
pub struct SltIU<T = ()>(pub T);
