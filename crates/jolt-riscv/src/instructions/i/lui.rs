use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

/// RV64I LUI: load upper immediate. Result is the immediate value itself.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(AddOperands, WriteLookupOutputToRD)]
#[instruction(RightOperandIsImm)]
pub struct Lui<T = ()>(pub T);
