#![cfg_attr(not(feature = "std"), no_std)]

extern crate jolt_sdk_macros;

pub use jolt_sdk_macros::main;
pub use postcard;

#[cfg(feature = "std")]
pub use liblasso::host;
