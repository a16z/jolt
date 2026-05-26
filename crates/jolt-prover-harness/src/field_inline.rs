use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FieldInlineFixtureFacts {
    pub fr_rows: usize,
    pub bridge_rows: usize,
    pub fr_off: bool,
}

impl FieldInlineFixtureFacts {
    pub const fn fr_off() -> Self {
        Self {
            fr_rows: 0,
            bridge_rows: 0,
            fr_off: true,
        }
    }

    pub const fn fr_on(fr_rows: usize, bridge_rows: usize) -> Self {
        Self {
            fr_rows,
            bridge_rows,
            fr_off: false,
        }
    }

    pub const fn has_field_activity(&self) -> bool {
        !self.fr_off && (self.fr_rows > 0 || self.bridge_rows > 0)
    }
}
