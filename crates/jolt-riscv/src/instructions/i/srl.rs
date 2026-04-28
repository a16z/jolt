use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

/// RV64I SRL: shift right logical. Shift amount from lower 6 bits of `y`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct Srl<T = ()>(pub T);
