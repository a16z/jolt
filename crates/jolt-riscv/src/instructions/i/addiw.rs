use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

/// RV64I ADDIW: 32-bit add immediate, sign-extended to 64 bits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(AddOperands, WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsImm)]
pub struct AddiW<T = ()>(pub T);
