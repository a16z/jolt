//! Jolt guest no-std boot implementation.
//!
//! Provides trap handler for Jolt guest programs compiled without std support.
//! This module is conditionally compiled when building for riscv64 without guest-std.
//!
//! The _start and boot sequence comes from ZeroOS arch-riscv.
//! This module provides the trap_handler that ZeroOS's _default_trap_handler calls.

/// Trap handler for no-std Jolt guests.
///
/// In no-std mode, traps are unexpected - just loop forever.
/// The emulator will detect this as termination via PC stall.
#[no_mangle]
pub extern "C" fn trap_handler(
    _a0: usize,
    _a1: usize,
    _a2: usize,
    _a3: usize,
    _a4: usize,
    _a5: usize,
    _a6: usize,
    _a7: usize,
) -> usize {
    // Unexpected trap in no-std mode - loop forever
    loop {
        unsafe {
            core::arch::asm!("j .", options(noreturn));
        }
    }
}
