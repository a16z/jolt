use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

/// RV64I SRAIW: 32-bit shift right arithmetic by immediate, sign-extended.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsImm)]
pub struct SraIW<T = ()>(pub T);
