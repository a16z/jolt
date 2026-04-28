use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

/// RV64I SRLW: 32-bit shift right logical, sign-extended to 64 bits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct SrlW<T = ()>(pub T);
