use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

/// RV64I SRLIW: 32-bit shift right logical by immediate, sign-extended.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsImm)]
pub struct SrlIW<T = ()>(pub T);
