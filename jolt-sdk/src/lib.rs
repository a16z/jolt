#![cfg_attr(not(any(feature = "host", feature = "guest-std")), no_std)]

extern crate jolt_sdk_macros;

// Link ZeroOS jolt-platform for guest builds on RISC-V.
// This provides __platform_bootstrap and (for guest-std) jolt_syscall.
// The `extern crate` ensures the linker includes it.
#[cfg(all(not(feature = "host"), target_arch = "riscv64"))]
extern crate zeroos_jolt_platform;

#[cfg(any(feature = "host", feature = "guest-verifier"))]
pub mod host_utils;
#[cfg(any(feature = "host", feature = "guest-verifier"))]
pub use host_utils::*;

// Guest std boot code - provides _start, kernel_main for std-mode guests
// Only compile this for the RISC-V target (not for host builds)
#[cfg(all(feature = "guest-std", target_arch = "riscv64"))]
mod guest_std_boot;

// Guest no-std boot code - provides _start, boot_main for no-std guests
// This initializes ZeroOS heap and provides clean exit via infinite loop
#[cfg(all(not(feature = "guest-std"), not(feature = "host"), target_arch = "riscv64"))]
mod guest_no_std_boot;

pub use jolt_platform::*;
pub use jolt_sdk_macros::provable;
pub use postcard;

use serde::{Deserialize, Serialize};

/// A wrapper type to mark guest program inputs as trusted_advice.
#[derive(Debug, Serialize, Deserialize)]
#[repr(transparent)]
pub struct TrustedAdvice<T> {
    value: T,
}

impl<T> TrustedAdvice<T> {
    pub fn new(value: T) -> Self {
        Self { value }
    }
}

impl<T> From<T> for TrustedAdvice<T> {
    fn from(value: T) -> Self {
        Self::new(value)
    }
}

impl<T> core::ops::Deref for TrustedAdvice<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

/// A wrapper type to mark guest program inputs as untrusted_advice.
#[derive(Debug, Serialize, Deserialize)]
#[repr(transparent)]
pub struct UntrustedAdvice<T> {
    value: T,
}

impl<T> UntrustedAdvice<T> {
    pub fn new(value: T) -> Self {
        Self { value }
    }
}

impl<T> From<T> for UntrustedAdvice<T> {
    fn from(value: T) -> Self {
        Self::new(value)
    }
}

impl<T> core::ops::Deref for UntrustedAdvice<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

// This is a dummy _HEAP_PTR to keep the compiler happy.
// It should never be used when compiled as a guest or with
// our custom allocator
#[no_mangle]
#[cfg(feature = "host")]
pub static mut _HEAP_PTR: u8 = 0;
