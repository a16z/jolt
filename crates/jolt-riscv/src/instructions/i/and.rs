use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

/// RV64I AND: bitwise AND of two registers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct And<T = ()>(pub T);
