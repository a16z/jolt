use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

/// RV64I LD: load doubleword (64 bits). Identity operation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(Load)]
pub struct Ld<T = ()>(pub T);
