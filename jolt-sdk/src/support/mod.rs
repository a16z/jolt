//! ZeroOS integration for Jolt zkVM
//!
//! This module provides the canonical ZeroOS integration code for Jolt.
//! It includes boot, trap handling, and I/O functionality needed to run
//! ZeroOS-based guests on the Jolt zkVM.
//!
//! ## Architecture
//!
//! The ZeroOS integration works as follows:
//! - Boot: `__platform_bootstrap` initializes ZeroOS subsystems (heap, VFS, scheduler)
//! - Trap: `trap_handler` routes syscalls through ZeroOS's Linux syscall layer
//! - I/O: `ecall` provides ECALL-based print functionality
//! - Exit: `platform_exit` uses infinite loop for clean termination
//!
//! ## Feature Flags
//!
//! Enable the `zeroos` feature in `jolt-sdk` to use this module.

mod boot;

pub mod ecall;
#[cfg(all(
    not(target_os = "none"),
    any(target_arch = "riscv32", target_arch = "riscv64")
))]
pub mod trap;

// Note: jolt_print, jolt_println, println, eprintln macros are exported at
// crate root via #[macro_export] in ecall.rs


cfg_if::cfg_if! {
    if #[cfg(target_os = "linux")] {
        // println and eprintln macros are defined in ecall.rs via #[macro_export]
        // They work for both std and no-std modes.

        pub fn exit(code: i32) -> ! {
            std::process::exit(code)
        }
    } else if #[cfg(target_os = "none")] {
        use crate::eprintln;
        pub use ecall::putchar;

        pub fn exit(code: i32) -> ! {
            platform_exit(code)
        }

        #[global_allocator]
        static ALLOCATOR: zeroos::alloc::System = zeroos::alloc::System;

        #[panic_handler]
        fn panic(info: &core::panic::PanicInfo) -> ! {
            eprintln!("PANIC: {}", info);
            exit(1)
        }
    }
}

/// Exit the program.
///
/// Enters an infinite loop (`j .`) which the Jolt emulator detects via PC stall
/// (prev_pc == pc) and treats as clean termination.
#[no_mangle]
pub extern "C" fn platform_exit(_code: i32) -> ! {
    unsafe {
        core::arch::asm!("j .", options(noreturn));
    }
}

/// Debug write function for ZeroOS debug crate integration.
///
/// # Safety
/// - `msg` must be either null (in which case nothing is written) or a valid pointer to `len`
///   bytes of readable memory.
#[no_mangle]
pub unsafe extern "C" fn __debug_write(msg: *const u8, len: usize) {
    if !msg.is_null() && len > 0 {
        let slice = core::slice::from_raw_parts(msg, len);
        for &byte in slice {
            ecall::putchar(byte);
        }
    }
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
#[cfg(all(target_arch ="riscv64", target_os = "linux"))]
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
