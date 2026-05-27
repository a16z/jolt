//! Typed inputs for stage 6a.

use jolt_field::Field;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct Stage6AddressPhaseClaims<F: Field> {
    pub bytecode_read_raf: F,
    pub booleanity: F,
}
