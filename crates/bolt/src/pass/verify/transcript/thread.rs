use melior::ir::operation::{OperationLike, OperationRef};

use crate::schema::operation_name;

use super::super::VerifyError;

#[derive(Default)]
pub(super) struct TranscriptThread {
    current_state: Option<String>,
}

impl TranscriptThread {
    pub(super) fn initialize(
        &mut self,
        operation: OperationRef<'_, '_>,
    ) -> Result<(), VerifyError> {
        if self.current_state.is_some() {
            return Err(VerifyError::new("multiple transcript.state ops"));
        }
        let result = operation
            .result(0)
            .map_err(|_| VerifyError::new("transcript.state requires one result"))?;
        self.current_state = Some(result.to_string());
        Ok(())
    }

    pub(super) fn require_state_input(
        &self,
        operation: OperationRef<'_, '_>,
        missing_state: impl Into<String>,
        missing_operand: impl Into<String>,
    ) -> Result<(), VerifyError> {
        let expected_input = self
            .current_state
            .as_deref()
            .ok_or_else(|| VerifyError::new(missing_state))?;
        let input = operation
            .operand(0)
            .map_err(|_| VerifyError::new(missing_operand))?
            .to_string();
        if input != expected_input {
            return Err(VerifyError::new(format!(
                "{} consumed transcript state {input}, expected {expected_input}",
                operation_name(operation)
            )));
        }
        Ok(())
    }

    pub(super) fn require_operand(
        &self,
        operation: OperationRef<'_, '_>,
        index: usize,
        message: impl Into<String>,
    ) -> Result<(), VerifyError> {
        operation
            .operand(index)
            .map(|_| ())
            .map_err(|_| VerifyError::new(message))
    }

    pub(super) fn advance_from_result(
        &mut self,
        operation: OperationRef<'_, '_>,
        missing_result: impl Into<String>,
    ) -> Result<(), VerifyError> {
        let result = operation
            .result(0)
            .map_err(|_| VerifyError::new(missing_result))?;
        self.current_state = Some(result.to_string());
        Ok(())
    }
}
