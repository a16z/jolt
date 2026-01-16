//! Jolt guest no-std boot implementation.
//!
//! Provides boot code (_start) for Jolt guest programs compiled without std support.
//! This module is conditionally compiled when building for riscv64 without guest-std.
//!
//! The boot sequence:
//! 1. _start: Initialize global pointer and stack pointer
//! 2. boot_main: Initialize platform (heap), call user's main(), then loop forever
//!
//! Unlike guest_std_boot.rs, this does NOT set up a trap handler because:
//! - Exit uses `j .` (infinite loop) which the emulator detects via PC stall
//! - No syscalls are needed in no-std mode (no VFS, no stdio)

use core::arch::naked_asm;

extern "C" {
    static __heap_start: u8;
    static __heap_end: u8;

    /// Platform bootstrap function from ZeroOS jolt-platform.
    /// Initializes the heap allocator via foundation::kfn::memory::kinit().
    fn __platform_bootstrap();

    /// User's main function (provided by the jolt macro)
    fn main();
}

/// Entry point for Jolt no-std guest programs.
///
/// This is a simple boot that sets up gp and sp, then jumps to boot_main.
/// No trap handler is set up because exit uses `j .` (infinite loop detection).
#[unsafe(naked)]
#[no_mangle]
#[link_section = ".text.boot"]
pub unsafe extern "C" fn _start() -> ! {
    naked_asm!(
        // Initialize global pointer first (RISC-V ABI requirement)
        ".weak __global_pointer$",
        ".hidden __global_pointer$",
        ".option push",
        ".option norelax",
        "   lla     gp, __global_pointer$",
        ".option pop",

        // Initialize stack pointer
        ".weak __stack_top",
        ".hidden __stack_top",
        "   lla     sp, __stack_top",
        "   andi    sp, sp, -16",

        // Jump to boot_main
        "   tail    {boot_main}",

        boot_main = sym boot_main,
    )
}

/// Boot main: Initialize platform, then call user's main() and loop forever.
///
/// This function:
/// 1. Calls __platform_bootstrap() to initialize ZeroOS (heap allocator)
/// 2. Calls the user's main() function
/// 3. Enters an infinite loop for clean termination (emulator detects PC stall)
#[no_mangle]
extern "C" fn boot_main() -> ! {
    unsafe {
        // Initialize the platform - this sets up the heap allocator
        __platform_bootstrap();

        // Call the user's main function
        main();

        // Loop forever - emulator detects termination via prev_pc == pc
        core::arch::asm!(
            "j .",
            options(noreturn)
        );
    }
}
