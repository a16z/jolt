use serde::{Deserialize, Serialize};

/// RV64A SC.D: store-conditional doubleword.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ScD<T = ()>(pub T);
