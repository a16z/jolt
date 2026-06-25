use serde::{Deserialize, Serialize};

use super::geometry::dimensions::FieldRegistersReadWriteDimensions;

pub const FIELD_REGISTERS_LOG_K: usize = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FieldInlineRepresentation {
    NativeFieldElement,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FieldInlineConfig {
    pub enabled: bool,
    pub field_register_log_k: usize,
    pub representation: FieldInlineRepresentation,
}

impl FieldInlineConfig {
    pub const fn disabled() -> Self {
        Self {
            enabled: false,
            field_register_log_k: FIELD_REGISTERS_LOG_K,
            representation: FieldInlineRepresentation::NativeFieldElement,
        }
    }

    pub const fn read_write_dimensions(self, log_t: usize) -> FieldRegistersReadWriteDimensions {
        FieldRegistersReadWriteDimensions::new(
            log_t,
            self.field_register_log_k,
            log_t,
            self.field_register_log_k,
        )
    }
}
