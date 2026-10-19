//! Checked narrowing conversions from `usize` into the fixed-width integers
//! used by wire formats and plan encodings.
//!
//! These helpers give every module reporting "value does not fit" one message
//! shape and one error variant.

use crate::AkitaError;

/// Narrow to `u32`, naming the quantity in the error.
#[inline]
pub fn usize_to_u32(value: usize, name: &str) -> Result<u32, AkitaError> {
    u32::try_from(value).map_err(|_| AkitaError::InvalidInput(format!("{name} does not fit u32")))
}

/// Narrow to `u64`, naming the quantity in the error.
#[inline]
pub fn usize_to_u64(value: usize, name: &str) -> Result<u64, AkitaError> {
    u64::try_from(value).map_err(|_| AkitaError::InvalidInput(format!("{name} does not fit u64")))
}

/// Narrow to `u8`, naming the quantity in the error.
#[inline]
pub fn usize_to_u8(value: usize, name: &str) -> Result<u8, AkitaError> {
    u8::try_from(value).map_err(|_| AkitaError::InvalidInput(format!("{name} does not fit u8")))
}

#[cfg(test)]
mod tests {
    use super::{usize_to_u32, usize_to_u64, usize_to_u8};
    use crate::AkitaError;

    #[test]
    fn narrowing_accepts_boundary_and_names_rejected_quantity() {
        assert_eq!(usize_to_u8(u8::MAX as usize, "x"), Ok(u8::MAX));
        assert_eq!(
            usize_to_u8(u8::MAX as usize + 1, "digit count"),
            Err(AkitaError::InvalidInput(
                "digit count does not fit u8".into()
            ))
        );
        assert_eq!(usize_to_u32(u32::MAX as usize, "x"), Ok(u32::MAX));
        if let Some(too_large) = (u32::MAX as usize).checked_add(1) {
            assert_eq!(
                usize_to_u32(too_large, "entry count"),
                Err(AkitaError::InvalidInput(
                    "entry count does not fit u32".into()
                ))
            );
        }
        assert!(usize_to_u64(usize::MAX, "x").is_ok());
    }
}
