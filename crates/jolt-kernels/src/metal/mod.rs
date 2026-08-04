//! Apple Metal compute kernels.
//!
//! Reusable field arithmetic and hybrid sumcheck kernels for Apple GPUs.

mod instruction_read_raf;
pub mod solinas;

#[cfg(test)]
pub(crate) use instruction_read_raf::MetalInstructionReadRafKernel;
pub use instruction_read_raf::{InstructionReadRafMetalConfig, MetalBackend, MetalConfig};
