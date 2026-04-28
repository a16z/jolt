use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

/// RV64I LW: load word (32 bits), sign-extended to 64 bits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(Load)]
pub struct Lw<T = ()>(pub T);
