//! Runtime support for Jolt zkVM guests
//!
//! This module provides the ZeroOS-based runtime for Jolt guests.
//! It includes boot, trap handling, and runtime functionality needed to run
//! guests on the Jolt zkVM.
//!
//! ## Architecture
//!
//! The runtime works as follows:
//! - Boot: `__platform_bootstrap` initializes ZeroOS subsystems (heap, VFS, scheduler)
//! - Trap: `trap_handler` routes syscalls through ZeroOS's Linux syscall layer
//! - Exit: `platform_exit` (in jolt-platform) uses infinite loop for clean termination

mod boot;
mod exit;

#[cfg(all(
    not(target_os = "none"),
    any(target_arch = "riscv32", target_arch = "riscv64")
))]
pub mod trap;

cfg_if::cfg_if! {
    if #[cfg(target_os = "linux")] {
        pub fn exit(code: i32) -> ! {
            std::process::exit(code)
        }
    } else if #[cfg(target_os = "none")] {
        pub use jolt_platform::putchar;

        pub fn exit(code: i32) -> ! {
            jolt_platform::platform_exit(code)
        }

        #[global_allocator]
        static ALLOCATOR: zeroos::alloc::System = zeroos::alloc::System;

        // Panic handler is provided by zeroos's "panic" feature (zeroos-runtime-nostd)
        // which calls __platform_abort from exit.rs
    }
}

/// Platform stdout write function - the fundamental output primitive.
///
/// This is the base write function that all other output functions can use.
/// It uses jolt-platform's VirtualHostIO instruction to emit output.
///
/// # Safety
/// - `msg` must be either null (in which case nothing is written) or a valid pointer to `len`
///   bytes of readable memory.
#[no_mangle]
pub unsafe extern "C" fn __platform_stdout_write(msg: *const u8, len: usize) {
    if !msg.is_null() && len > 0 {
        let slice = core::slice::from_raw_parts(msg, len);
        for &byte in slice {
            jolt_platform::putchar(byte);
        }
    }
}

/// Debug write function for ZeroOS debug crate integration.
///
/// This is an alias for `__platform_stdout_write` that is only compiled
/// when the `debug` feature is enabled. The zeroos-debug crate uses this
/// for conditional debug output.
///
/// # Safety
/// Same as `__platform_stdout_write`.
#[cfg(feature = "debug")]
#[no_mangle]
pub unsafe extern "C" fn __debug_write(msg: *const u8, len: usize) {
    __platform_stdout_write(msg, len);
}

/// Syscall handler for Jolt guests.
///
/// This function is called by jolt-sdk's trap_handler to route syscalls
/// through ZeroOS's Linux syscall infrastructure. The syscall dispatch
/// is proven as part of the guest execution.
///
/// # Arguments
/// * `a0`-`a5` - Syscall arguments
/// * `nr` - Syscall number
///
/// # Returns
/// The syscall return value (negative values indicate errors)
#[cfg(all(target_arch = "riscv64", target_os = "linux"))]
#[no_mangle]
pub extern "C" fn jolt_syscall(
    a0: usize,
    a1: usize,
    a2: usize,
    a3: usize,
    a4: usize,
    a5: usize,
    nr: usize,
) -> isize {
    zeroos::os::linux::linux_handle(a0, a1, a2, a3, a4, a5, nr)
}
