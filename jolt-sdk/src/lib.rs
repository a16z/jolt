#![cfg_attr(not(feature = "host"), no_std)]

extern crate jolt_sdk_macros;

pub use jolt_sdk_macros::provable;
pub use postcard;

#[cfg(feature = "host")]
pub mod host_utils;
#[cfg(feature = "host")]
pub use host_utils::*;

pub mod cycle_tracking;
pub use cycle_tracking::*;

pub mod alloc;
pub use alloc::*;

#[cfg(feature = "sha256")]
pub mod sha256;
#[cfg(feature = "sha256")]
pub use sha256::*;

#[cfg(feature = "keccak256")]
pub mod keccak256;
#[cfg(feature = "keccak256")]
pub use keccak256::*;

// This is a dummy _HEAP_PTR to keep the compiler happy.
// It should never be used when compiled as a guest or with
// our custom allocator
#[no_mangle]
#[cfg(feature = "host")]
pub static mut _HEAP_PTR: u8 = 0;
