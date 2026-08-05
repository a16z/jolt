//! Apple Metal compute kernels.
//!
//! Reusable field arithmetic and hybrid sumcheck kernels for Apple GPUs.

mod booleanity;
mod bytecode_read_raf;
mod instruction_input;
mod instruction_ra_virtualization;
mod instruction_read_raf;
pub mod solinas;
mod spartan_outer;

pub use booleanity::{BooleanityAddressMetalConfig, BooleanityMetalConfig};
pub use bytecode_read_raf::{BytecodeReadRafMetalConfig, BytecodeReadRafResidentRows};
pub use instruction_input::InstructionInputMetalConfig;
pub use instruction_ra_virtualization::InstructionRaVirtualizationMetalConfig;
#[cfg(test)]
pub(crate) use instruction_read_raf::MetalInstructionReadRafKernel;
pub use instruction_read_raf::{InstructionReadRafMetalConfig, MetalBackend, MetalConfig};
pub use spartan_outer::SpartanOuterUniskipMetalConfig;
