use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

/// RV64I JAL: jump and link. `rd = PC + 4; PC = PC + imm`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(AddOperands, Jump)]
#[instruction(LeftOperandIsPC, RightOperandIsImm)]
pub struct Jal<T = ()>(pub T);
