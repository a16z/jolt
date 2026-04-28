use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

/// RV64I ANDI: bitwise AND with sign-extended immediate.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsImm)]
pub struct AndI<T = ()>(pub T);
