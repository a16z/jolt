use serde::{Deserialize, Serialize};

/// RV32A AMOMINU.W: atomic unsigned min word.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AmoMinUW<T = ()>(pub T);
