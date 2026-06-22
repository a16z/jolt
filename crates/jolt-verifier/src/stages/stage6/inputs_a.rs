//! Typed inputs for stage 6a.

use jolt_field::Field;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: Deserialize<'de>"))]
pub struct Stage6AddressPhaseClaims<F: Field> {
    pub bytecode_read_raf: F,
    pub booleanity: F,
    /// `BytecodeValStage(s)` openings staged at the address-phase point;
    /// present only in committed program mode.
    pub bytecode_val_stages: Option<Vec<F>>,
}
