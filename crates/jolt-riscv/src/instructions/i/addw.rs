use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

/// RV64I ADDW: 32-bit add, sign-extended to 64 bits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(AddOperands, WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct AddW<T = ()>(pub T);
