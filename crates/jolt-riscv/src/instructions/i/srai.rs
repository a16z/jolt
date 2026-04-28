use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

/// RV64I SRAI: shift right arithmetic by immediate.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsImm)]
pub struct SraI<T = ()>(pub T);
