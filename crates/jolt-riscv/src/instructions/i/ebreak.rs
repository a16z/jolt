use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

/// RV64I EBREAK: breakpoint trap.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
pub struct Ebreak<T = ()>(pub T);
