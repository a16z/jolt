use thiserror::Error;

/// Failures from the Metal device tier. Construction-time errors
/// ([`JoltBackend::metal`](crate::JoltBackend::metal) is fail-closed: no
/// device, shader compilation, or pipeline failure surfaces there, never
/// mid-proof) and dispatch-time errors share this type.
#[derive(Debug, Clone, Error)]
pub enum MetalError {
    #[error("no Metal device available")]
    NoDevice,

    #[error("Metal command queue creation failed")]
    NoCommandQueue,

    #[error("Metal shader compilation failed: {0}")]
    Compile(String),

    /// The compiled library has no function with a registered kernel's name —
    /// a shader-source/`KernelId` mismatch, caught at prewarm.
    #[error("Metal kernel function not found: {0}")]
    MissingFunction(&'static str),

    #[error("Metal pipeline creation failed for {name}: {reason}")]
    Pipeline { name: &'static str, reason: String },

    /// A pipeline's per-threadgroup thread capacity came in under the fixed
    /// dispatch width (register pressure) — caught at prewarm.
    #[error("Metal pipeline {name} supports {max} threads/threadgroup, need {need}")]
    ThreadgroupTooSmall {
        name: &'static str,
        max: usize,
        need: usize,
    },

    #[error("Metal buffer allocation failed ({bytes} bytes)")]
    Alloc { bytes: usize },

    #[error("Metal command buffer creation failed")]
    NoCommandBuffer,

    #[error("Metal command buffer failed: {0}")]
    Execution(String),

    /// The instance's geometry is one a device kernel does not model (e.g. a
    /// degenerate table count) — the slot falls back to its optimized twin.
    #[error("shape unsupported by the device kernel: {0}")]
    UnsupportedShape(&'static str),
}
