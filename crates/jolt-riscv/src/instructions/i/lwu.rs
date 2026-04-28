use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

/// RV64I LWU: load word, zero-extended to 64 bits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(Load)]
pub struct Lwu<T = ()>(pub T);
