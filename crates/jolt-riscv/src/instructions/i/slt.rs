use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

/// RV64I SLT: set if less than (signed). `rd = (rs1 < rs2) ? 1 : 0`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct Slt<T = ()>(pub T);
