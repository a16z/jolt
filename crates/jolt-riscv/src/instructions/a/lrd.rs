use serde::{Deserialize, Serialize};

/// RV64A LR.D: load-reserved doubleword.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LrD<T = ()>(pub T);
