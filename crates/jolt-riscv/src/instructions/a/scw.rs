use serde::{Deserialize, Serialize};

/// RV32A SC.W: store-conditional word.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ScW<T = ()>(pub T);
