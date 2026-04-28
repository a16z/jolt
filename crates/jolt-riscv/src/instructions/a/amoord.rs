use serde::{Deserialize, Serialize};

/// RV64A AMOOR.D: atomic OR doubleword.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AmoOrD<T = ()>(pub T);
