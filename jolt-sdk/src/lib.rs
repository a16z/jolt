#![cfg_attr(not(any(feature = "host", feature = "guest-std")), no_std)]

extern crate jolt_sdk_macros;

#[cfg(any(feature = "host", feature = "guest-verifier"))]
pub mod host_utils;
#[cfg(any(feature = "host", feature = "guest-verifier"))]
pub use host_utils::*;

pub use jolt_platform::*;
pub use jolt_sdk_macros::provable;
pub use postcard;

use serde::{Deserialize, Serialize};

/// A wrapper type to mark function inputs as private in zero-knowledge proofs.
/// 
/// When an input is wrapped with `Private<T>`, it indicates that this input
/// should be treated as private (witness) data in the proof system, meaning
/// its value will not be revealed in the generated proof.
/// 
/// # Example
/// ```rust
/// #[provable]
/// fn compute(public_input: u32, private_input: Private<u32>) -> u32 {
///     // In the function body, private_input is automatically unwrapped
///     public_input + private_input
/// }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(transparent)]
pub struct Private<T> {
    value: T,
}

impl<T> Private<T> {
    /// Create a new Private wrapper around a value
    pub fn new(value: T) -> Self {
        Self { value }
    }
    
    /// Extract the inner value
    pub fn into_inner(self) -> T {
        self.value
    }
    
    /// Get a reference to the inner value
    pub fn inner(&self) -> &T {
        &self.value
    }
    
    /// Get a mutable reference to the inner value
    pub fn inner_mut(&mut self) -> &mut T {
        &mut self.value
    }
}

impl<T> From<T> for Private<T> {
    fn from(value: T) -> Self {
        Self::new(value)
    }
}

// Implement Deref for ergonomic usage in guest code
impl<T> core::ops::Deref for Private<T> {
    type Target = T;
    
    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

impl<T> core::ops::DerefMut for Private<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.value
    }
}

// This is a dummy _HEAP_PTR to keep the compiler happy.
// It should never be used when compiled as a guest or with
// our custom allocator
#[no_mangle]
#[cfg(feature = "host")]
pub static mut _HEAP_PTR: u8 = 0;
