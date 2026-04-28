use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

/// RV64I ADD: `rd = rs1 + rs2` (wrapping).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(AddOperands, WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct Add<T = ()>(pub T);
