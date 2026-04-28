use serde::{Deserialize, Serialize};

/// RV64A AMOMAXU.D: atomic unsigned max doubleword.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AmoMaxUD<T = ()>(pub T);
