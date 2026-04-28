use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

/// Virtual ADVICE: runtime-provided advice value.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(Advice, WriteLookupOutputToRD)]
pub struct VirtualAdvice<T = ()>(pub T);
