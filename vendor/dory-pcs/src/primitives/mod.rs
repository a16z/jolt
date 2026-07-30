//! # Primitives
//! This submodule defines the basic EC and FS related tools that Dory is built upon
pub mod arithmetic;
pub mod poly;
pub mod serialization;
pub mod transcript;
pub use serialization::*;
