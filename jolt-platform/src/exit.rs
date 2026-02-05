//! Exit functionality for Jolt guests
//!
//! Provides clean program termination that the Jolt emulator can detect.

/// Exit the program.
///
/// Enters an infinite loop (`j .`) which the Jolt emulator detects via PC stall
/// (prev_pc == pc) and treats as clean termination.
#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
#[no_mangle]
pub extern "C" fn platform_exit(_code: i32) -> ! {
    unsafe {
        core::arch::asm!("j .", options(noreturn));
    }
}
