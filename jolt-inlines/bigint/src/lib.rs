#![cfg_attr(not(feature = "host"), no_std)]

pub mod multiplication;
pub use multiplication::*;

#[cfg(feature = "host")]
mod host;
#[cfg(feature = "host")]
pub use host::*;
