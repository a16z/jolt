use serde::{Deserialize, Serialize};

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

    pub const fn native_v1() -> Self {
        Self {
            enabled: true,
            field_register_log_k: FIELD_REGISTERS_LOG_K,
            representation: FieldInlineRepresentation::NativeFieldElement,
        }
    }
}
