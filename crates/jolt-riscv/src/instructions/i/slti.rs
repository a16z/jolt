use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

/// RV64I SLTI: set if less than immediate (signed).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsImm)]
pub struct SltI<T = ()>(pub T);
