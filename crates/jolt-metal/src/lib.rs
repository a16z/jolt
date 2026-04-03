#![cfg(target_os = "macos")]
#![allow(unused_results)]
#![allow(clippy::missing_safety_doc, dead_code)]

mod buffer;
pub mod config;
pub mod field;
pub mod field_params;
mod kernel;
pub mod msl_field;
pub mod msl_reduce;
pub mod pipeline;

mod backend;
pub use backend::MetalBackend;
pub use buffer::MetalBuffer;
pub use kernel::MetalKernel;
pub use msl_reduce::CompileMode;
