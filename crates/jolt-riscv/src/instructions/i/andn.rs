use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

/// Zbb ANDN: bitwise AND-NOT. `rd = rs1 & ~rs2`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct Andn<T = ()>(pub T);
