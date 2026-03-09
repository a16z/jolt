#![cfg(target_os = "macos")]
#![allow(unused_results)]
#![allow(clippy::todo, clippy::missing_safety_doc, dead_code)]

mod buffer;
mod compiler;
mod device;
pub mod field;
mod kernel;
mod reduction;
pub(crate) mod shaders;

pub use buffer::MetalBuffer;
pub use device::MetalBackend;
pub use kernel::MetalKernel;
