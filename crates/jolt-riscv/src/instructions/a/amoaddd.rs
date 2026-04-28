use serde::{Deserialize, Serialize};

/// RV64A AMOADD.D: atomic add doubleword.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AmoAddD<T = ()>(pub T);
