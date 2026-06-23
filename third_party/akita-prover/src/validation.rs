use akita_field::AkitaError;

pub(crate) const MAX_I8_LOG_BASIS: u32 = 6;

#[inline]
pub(crate) fn is_i8_log_basis(log_basis: u32) -> bool {
    (1..=MAX_I8_LOG_BASIS).contains(&log_basis)
}

#[inline]
pub(crate) fn validate_i8_setup_log_basis(log_basis: u32, context: &str) -> Result<(), AkitaError> {
    if is_i8_log_basis(log_basis) {
        Ok(())
    } else {
        Err(AkitaError::InvalidSetup(format!(
            "log_basis must be in 1..=6 {context}"
        )))
    }
}

#[inline]
pub(crate) fn validate_i8_input_log_basis(log_basis: u32, context: &str) -> Result<(), AkitaError> {
    if is_i8_log_basis(log_basis) {
        Ok(())
    } else {
        Err(AkitaError::InvalidInput(format!(
            "log_basis must be in 1..=6 {context}"
        )))
    }
}
