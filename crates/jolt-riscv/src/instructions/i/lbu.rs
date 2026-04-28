use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

/// RV64I LBU: load byte, zero-extended to 64 bits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(Load)]
pub struct Lbu<T = ()>(pub T);
