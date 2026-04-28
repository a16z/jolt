use serde::{Deserialize, Serialize};

/// RV64A AMOMIN.D: atomic signed min doubleword.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AmoMinD<T = ()>(pub T);
