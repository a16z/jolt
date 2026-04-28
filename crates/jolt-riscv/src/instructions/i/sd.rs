use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

/// RV64I SD: store doubleword (full 64 bits). Identity operation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(Store)]
pub struct Sd<T = ()>(pub T);
