use serde::{Deserialize, Serialize};

/// Virtual ADVICE_LH: advice tape value for load halfword.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AdviceLh<T = ()>(pub T);
