use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

/// RV64I ORI: bitwise OR with sign-extended immediate.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsImm)]
pub struct OrI<T = ()>(pub T);
