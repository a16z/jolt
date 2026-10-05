//! Platform backends. `macos` wraps Metal; `unsupported` has the same
//! crate-internal surface with uninhabited types, so every public type
//! compiles everywhere and only `Device::system_default` can be reached.

use std::time::Duration;

use crate::error::MetalError;

pub(crate) enum RawBatchOutcome {
    CompletionConfirmed(Result<Duration, MetalError>),
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
