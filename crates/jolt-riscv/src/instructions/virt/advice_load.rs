use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

/// Virtual ADVICE_LOAD: advice-tape read.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(Advice, WriteLookupOutputToRD)]
pub struct VirtualAdviceLoad<T = ()>(pub T);
