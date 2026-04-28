use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

/// Virtual MOVSIGN: returns all-ones if `x` is negative (signed), otherwise zero.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsImm)]
pub struct MovSign<T = ()>(pub T);
