use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

/// RV64I JALR: jump and link register. `rd = PC + 4; PC = (rs1 + imm) & !1`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(AddOperands, Jump)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsImm)]
pub struct Jalr<T = ()>(pub T);
