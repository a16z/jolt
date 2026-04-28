use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

/// RV64I SLLI: shift left logical by immediate. Immediate already masked.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsImm)]
pub struct SllI<T = ()>(pub T);
