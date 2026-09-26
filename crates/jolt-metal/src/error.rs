//! Typed runtime failures and the recovery class each one permits.

use thiserror::Error;

/// What a consumer may do after a [`MetalError`].
///
/// The runtime classifies; it never falls back or retries on its own. The
/// consumer chooses the recovery granularity (operation, stage, or proof).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ErrorClass {
    /// No usable Metal device. Use the CPU backend.
    Unavailable,
    /// A shader library or pipeline could not be built. Raised only while a
    /// [`ShaderLibrary`](crate::runtime::ShaderLibrary) is constructed, never
    /// mid-proof. Use the CPU backend and report the bug.
    Setup,
    /// A request exceeds a device limit. Detected before any GPU work is
    /// encoded. Re-plan the job or run it on the CPU.
    Capacity,
    /// The GPU rejected or abandoned submitted work for a reason unrelated to
    /// its contents (timeout, memory pressure, revoked access). The work may
    /// be retried or moved to the CPU.
    Transient,
    /// The work itself is wrong: a GPU memory fault, a caller contract
    /// violation, an Objective-C exception, or a result that fails validation.
    /// Never retry silently. Drop the device and fail loudly.
    Fault,
}

/// `MTLCommandBufferError` codes, from the macOS SDK header
/// `Metal.framework/Headers/MTLCommandBuffer.h`.
///
/// `DeviceRemoved` (11) is documented as impossible on Apple Silicon and
/// `Memoryless` (10) applies only to render targets; both arrive as
/// [`CommandBufferError::Unknown`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
pub enum CommandBufferError {
    /// `MTLCommandBufferErrorInternal` (1).
    #[error("Internal")]
    Internal,
    /// `MTLCommandBufferErrorTimeout` (2).
    #[error("Timeout")]
    Timeout,
    /// `MTLCommandBufferErrorPageFault` (3).
    #[error("PageFault")]
    PageFault,
    /// `MTLCommandBufferErrorAccessRevoked` (4).
    #[error("AccessRevoked")]
    AccessRevoked,
    /// `MTLCommandBufferErrorNotPermitted` (7).
    #[error("NotPermitted")]
    NotPermitted,
    /// `MTLCommandBufferErrorOutOfMemory` (8).
    #[error("OutOfMemory")]
    OutOfMemory,
    /// `MTLCommandBufferErrorInvalidResource` (9).
    #[error("InvalidResource")]
    InvalidResource,
    /// `MTLCommandBufferErrorStackOverflow` (12).
    #[error("StackOverflow")]
    StackOverflow,
    /// Any other code, or an error outside `MTLCommandBufferErrorDomain`.
    #[error("unknown error code {0}")]
    Unknown(isize),
    /// The command buffer reported an error status without an `NSError`.
    #[error("error status without an NSError")]
    Unreported,
}

impl CommandBufferError {
    /// Decodes an `NSError` code from `MTLCommandBufferErrorDomain`.
    pub(crate) fn from_code(code: isize) -> Self {
        match code {
            1 => Self::Internal,
            2 => Self::Timeout,
            3 => Self::PageFault,
            4 => Self::AccessRevoked,
            7 => Self::NotPermitted,
            8 => Self::OutOfMemory,
            9 => Self::InvalidResource,
            12 => Self::StackOverflow,
            other => Self::Unknown(other),
        }
    }

    pub fn class(self) -> ErrorClass {
        match self {
            Self::Timeout | Self::OutOfMemory | Self::AccessRevoked | Self::NotPermitted => {
                ErrorClass::Transient
            }
            Self::Internal
            | Self::PageFault
            | Self::InvalidResource
            | Self::StackOverflow
            | Self::Unknown(_)
            | Self::Unreported => ErrorClass::Fault,
        }
    }
}

/// The device limit a [`MetalError::CapacityExceeded`] request ran into.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CapacityLimit {
    /// The request's byte size does not fit in `usize`.
    AddressSpace,
    /// `MTLDevice.maxBufferLength`.
    MaxBufferLength,
    /// `MTLDevice.recommendedMaxWorkingSetSize`, counting memory the device
    /// has already allocated.
    WorkingSet,
}

/// A failure of the Metal runtime. Every variant has an [`ErrorClass`].
#[derive(Debug, Error)]
pub enum MetalError {
    #[error("Metal is unavailable: {reason}")]
    Unavailable { reason: String },

    #[error("invalid shader library declaration: {reason}")]
    InvalidLibrary { reason: String },

    #[error("shader compilation failed:\n{log}")]
    ShaderCompile { log: String },

    #[error("cannot build a pipeline for kernel `{kernel}`: {reason}")]
    Pipeline { kernel: String, reason: String },

    #[error("{requested} bytes exceeds the {limit:?} limit of {available} bytes")]
    CapacityExceeded {
        limit: CapacityLimit,
        requested: u128,
        available: u128,
    },

    #[error("Metal could not allocate a {bytes}-byte buffer")]
    AllocationFailed { bytes: usize },

    #[error("the library has no pipeline named `{name}`")]
    UnknownPipeline { name: String },

    #[error("invalid dispatch of `{pipeline}`: {reason}")]
    InvalidDispatch { pipeline: String, reason: String },

    #[error("command buffer failed ({code}) running [{}]: {description}", pipelines.join(", "))]
    CommandBuffer {
        code: CommandBufferError,
        description: String,
        /// Distinct pipelines encoded in the failed batch, in first-use order.
        pipelines: Vec<String>,
    },

    #[error("Metal returned no {object}")]
    NilObject { object: &'static str },

    #[error("Objective-C exception during {operation}: {description}")]
    ObjcException {
        operation: &'static str,
        description: String,
    },

    #[error("device buffer failed read-back validation: {reason}")]
    InvalidReadback { reason: String },
}

impl MetalError {
    pub fn class(&self) -> ErrorClass {
        match self {
            Self::Unavailable { .. } => ErrorClass::Unavailable,
            Self::InvalidLibrary { .. } | Self::ShaderCompile { .. } | Self::Pipeline { .. } => {
                ErrorClass::Setup
            }
            Self::CapacityExceeded { .. } | Self::AllocationFailed { .. } => ErrorClass::Capacity,
            Self::CommandBuffer { code, .. } => code.class(),
            Self::UnknownPipeline { .. }
            | Self::InvalidDispatch { .. }
            | Self::NilObject { .. }
            | Self::ObjcException { .. }
            | Self::InvalidReadback { .. } => ErrorClass::Fault,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Codes and names from `MTLCommandBuffer.h` (macOS 26 SDK).
    #[test]
    fn command_buffer_codes_match_sdk_header() {
        let header = [
            (1, CommandBufferError::Internal),
            (2, CommandBufferError::Timeout),
            (3, CommandBufferError::PageFault),
            (4, CommandBufferError::AccessRevoked),
            (7, CommandBufferError::NotPermitted),
            (8, CommandBufferError::OutOfMemory),
            (9, CommandBufferError::InvalidResource),
            (12, CommandBufferError::StackOverflow),
        ];
        for (code, expected) in header {
            assert_eq!(CommandBufferError::from_code(code), expected);
        }
        for unmapped in [0, 5, 6, 10, 11, 13, -1] {
            assert_eq!(
                CommandBufferError::from_code(unmapped),
                CommandBufferError::Unknown(unmapped)
            );
        }
    }

    #[test]
    fn only_environmental_failures_are_transient() {
        let transient: Vec<_> = (-1..=13)
            .map(CommandBufferError::from_code)
            .filter(|code| code.class() == ErrorClass::Transient)
            .collect();
        assert_eq!(
            transient,
            [
                CommandBufferError::Timeout,
                CommandBufferError::AccessRevoked,
                CommandBufferError::NotPermitted,
                CommandBufferError::OutOfMemory,
            ]
        );
        assert_eq!(CommandBufferError::Unreported.class(), ErrorClass::Fault);
    }
}
