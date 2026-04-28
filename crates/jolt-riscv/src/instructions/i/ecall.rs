use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

/// RV64I ECALL: environment call (syscall).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
pub struct Ecall<T = ()>(pub T);
