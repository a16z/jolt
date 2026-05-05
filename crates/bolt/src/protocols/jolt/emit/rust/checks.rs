use std::collections::BTreeSet;

use crate::emit::rust::EmitError;

pub(super) fn require_supported_symbol(
    kind: &str,
    actual: &str,
    expected: &str,
) -> Result<(), EmitError> {
    if actual == expected {
        Ok(())
    } else {
        Err(EmitError::new(format!(
            "unsupported {kind} @{actual}; expected @{expected}"
        )))
    }
}

pub(super) fn require_supported_symbol_for_emitter(
    emitter: &str,
    kind: &str,
    actual: &str,
    expected: &str,
) -> Result<(), EmitError> {
    if actual == expected {
        Ok(())
    } else {
        Err(EmitError::new(format!(
            "unsupported {kind} @{actual}; Rust {emitter} emitter currently supports @{expected}"
        )))
    }
}

pub(super) fn verify_count(
    kind: &str,
    symbol: &str,
    expected: usize,
    actual: usize,
) -> Result<(), EmitError> {
    if expected == actual {
        Ok(())
    } else {
        Err(EmitError::new(format!(
            "{kind} @{symbol} count mismatch: expected {expected}, got {actual}"
        )))
    }
}

pub(super) fn missing_role_binding(kind: &str, symbol: &str) -> EmitError {
    EmitError::new(format!("missing {kind} for `{symbol}`"))
}

pub(super) fn symbols<'a>(values: impl Iterator<Item = &'a String>) -> BTreeSet<String> {
    values.cloned().collect()
}
