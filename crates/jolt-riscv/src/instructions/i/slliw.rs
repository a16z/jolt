use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

/// RV64I SLLIW: 32-bit shift left logical by immediate, sign-extended.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsImm)]
pub struct SllIW<T = ()>(pub T);
