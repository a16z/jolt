//! Marker trait for concrete instruction payloads that can convert through
//! Jolt's canonical final bytecode row.

use crate::JoltRow;

pub trait JoltRowData: Copy + Into<JoltRow> + TryFrom<JoltRow> {
    fn jolt_row(&self) -> JoltRow {
        (*self).into()
    }

    fn is_virtual(&self) -> bool {
        self.jolt_row().virtual_sequence_remaining.is_some()
    }
}

impl JoltRowData for JoltRow {}
