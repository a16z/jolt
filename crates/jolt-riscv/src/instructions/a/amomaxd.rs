use serde::{Deserialize, Serialize};

/// RV64A AMOMAX.D: atomic signed max doubleword.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AmoMaxD<T = ()>(pub T);
