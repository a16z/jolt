use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

/// Virtual ADVICE_LEN: advice-tape length query.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(Advice, WriteLookupOutputToRD)]
pub struct VirtualAdviceLen<T = ()>(pub T);
