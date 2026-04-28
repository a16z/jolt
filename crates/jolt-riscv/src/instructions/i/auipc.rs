use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

/// RV64I AUIPC: add upper immediate to PC. `rd = PC + imm`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(AddOperands, WriteLookupOutputToRD)]
#[instruction(LeftOperandIsPC, RightOperandIsImm)]
pub struct Auipc<T = ()>(pub T);
