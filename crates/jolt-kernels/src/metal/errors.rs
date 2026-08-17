//! Shared error constructors for the Metal adapter layer.

use jolt_field::AkitaField;
use jolt_sumcheck::SumcheckError;

use crate::{KernelError, SumcheckKernelError};

pub(super) fn metal_error(message: impl ToString) -> SumcheckError<AkitaField> {
    SumcheckError::ComputeBackend {
        backend: "metal",
        message: message.to_string(),
    }
}

pub(super) fn metal_prepare_error(error: impl ToString) -> KernelError<AkitaField> {
    metal_error(error).into()
}

pub(super) fn metal_output_error(error: impl ToString) -> SumcheckKernelError<AkitaField> {
    SumcheckKernelError::ComputeBackend {
        backend: "metal",
        message: error.to_string(),
    }
}
