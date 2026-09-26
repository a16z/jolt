//! Prover-only Metal support for `jolt-field`.
//!
//! This crate owns the Metal runtime and the MSL field arithmetic shared by
//! Jolt's and Akita's GPU provers. It never enters a verifier dependency graph. On non-macOS targets
//! it compiles without any Objective-C dependency and
//! [`Device::system_default`](runtime::Device::system_default) returns
//! [`MetalError::Unavailable`].
//!
//! Every failure is a typed [`MetalError`] with an [`ErrorClass`] telling
//! the consumer whether it may fall back, retry, or must stop. Design:
//! `specs/jolt-metal-field.md`.
//!
//! Objective-C exceptions are caught with `objc2::exception::catch`, which
//! relies on unwinding: do not build consumers with `panic = "abort"`.

#![cfg_attr(not(target_os = "macos"), forbid(unsafe_code))]
// Off macOS every backend type is uninhabited: code past a backend call is
// unreachable and Metal-only helpers are unused. macOS builds lint both.
#![cfg_attr(
    not(target_os = "macos"),
    expect(
        unreachable_code,
        dead_code,
        reason = "the non-macOS backend is uninhabited"
    )
)]
#![deny(
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing,
    clippy::panic_in_result_fn,
    clippy::unreachable
)]

mod error;
pub mod field;
pub mod runtime;
pub mod shaders;

pub use error::{CapacityLimit, CommandBufferError, ErrorClass, MetalError};
pub use field::MetalField;
