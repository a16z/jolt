use crate::error::MetalError;

pub(crate) enum RawBatchOutcome {
    CompletionConfirmed(Result<(), MetalError>),
    CompletionUncertain(MetalError),
}

#[cfg(target_os = "macos")]
mod macos;
#[cfg(target_os = "macos")]
pub(crate) use macos::*;

#[cfg(not(target_os = "macos"))]
mod unsupported;
#[cfg(not(target_os = "macos"))]
pub(crate) use unsupported::*;
