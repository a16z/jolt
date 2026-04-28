use serde::{Deserialize, Serialize};

/// RV64A AMOMINU.D: atomic unsigned min doubleword.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AmoMinUD<T = ()>(pub T);
