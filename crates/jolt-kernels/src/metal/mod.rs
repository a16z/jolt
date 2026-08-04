//! Apple Metal compute kernels.
//!
//! Reusable field arithmetic and hybrid sumcheck kernels for Apple GPUs.

mod booleanity;
mod instruction_read_raf;
pub mod solinas;
mod spartan_outer;

pub use booleanity::BooleanityMetalConfig;
#[cfg(test)]
pub(crate) use instruction_read_raf::MetalInstructionReadRafKernel;
pub use instruction_read_raf::{InstructionReadRafMetalConfig, MetalBackend, MetalConfig};
pub use spartan_outer::SpartanOuterUniskipMetalConfig;
