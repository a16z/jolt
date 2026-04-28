use serde::{Deserialize, Serialize};

/// Virtual ADVICE_LB: advice tape value for load byte.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AdviceLb<T = ()>(pub T);
