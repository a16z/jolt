use serde::{Deserialize, Serialize};

/// RV32A AMOMAXU.W: atomic unsigned max word.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AmoMaxUW<T = ()>(pub T);
