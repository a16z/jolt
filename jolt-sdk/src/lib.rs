#![cfg_attr(not(feature = "host"), no_std)]

extern crate jolt_sdk_macros;

pub use jolt_sdk_macros::provable;
pub use postcard;

#[cfg(feature = "host")]
pub mod host_utils;
#[cfg(feature = "host")]
pub use host_utils::*;

pub mod alloc;
pub use alloc::*;
