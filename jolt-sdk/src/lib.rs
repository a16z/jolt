#![cfg_attr(not(any(feature = "host", feature = "guest-std")), no_std)]

extern crate jolt_sdk_macros;

pub use jolt_sdk_macros::provable;
pub use postcard;

#[cfg(feature = "host")]
pub mod host_utils;
#[cfg(feature = "host")]
pub use host_utils::*;

pub mod alloc;
pub use alloc::*;
#[cfg(any(feature = "host", feature = "guest-std"))]
pub mod random;
#[cfg(any(feature = "host", feature = "guest-std"))]
pub use random::*;
#[cfg(any(feature = "host", feature = "guest-std"))]
pub mod print;
#[cfg(any(feature = "host", feature = "guest-std"))]
pub use print::*;

pub mod cycle_tracking;
pub use cycle_tracking::*;

// This is a dummy _HEAP_PTR to keep the compiler happy.
// It should never be used when compiled as a guest or with
// our custom allocator
#[no_mangle]
#[cfg(feature = "host")]
pub static mut _HEAP_PTR: u8 = 0;
