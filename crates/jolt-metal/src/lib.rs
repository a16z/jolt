#![cfg(target_os = "macos")]
#![allow(unused_results)]
#![allow(clippy::missing_safety_doc, dead_code)]

mod buffer;
pub mod compiler;
pub mod coop_field_gen;
mod device;
pub mod field;
pub mod field_config;
mod kernel;
pub mod metal_device_config;
pub mod msl_field_gen;
mod reduction;
pub mod shaders;

pub use buffer::MetalBuffer;
pub use compiler::CompileMode;
pub use device::MetalBackend;
pub use kernel::MetalKernel;
