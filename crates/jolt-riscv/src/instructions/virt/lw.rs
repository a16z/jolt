use serde::{Deserialize, Serialize};

/// Virtual LW: virtual load word used in tracer-emitted virtual sequences.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VirtualLw<T = ()>(pub T);
