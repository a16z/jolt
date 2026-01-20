//! Jolt guest std boot implementation.
//!
//! Provides Jolt-specific components for guest programs compiled with std support:
//! - `_trap_handler`: Jolt-specific trap entry using ECALL convention (t1 return address)
//! - `trap_handler`: High-level syscall router via `jolt_syscall`
//! - `__main_entry`: Entry point that calls the macro-generated `main()`
//!
//! Boot code (`_start`, `__runtime_bootstrap`, `_init`, `_fini`) is provided by ZeroOS
//! (arch-riscv and runtime-musl crates). This module only provides Jolt-specific overrides.

use core::arch::naked_asm;

extern "C" {
    // User's provable function (provided by the jolt macro as `main()` with no args)
    fn main();

    /// Syscall handler from jolt-platform.
    /// Routes syscalls through ZeroOS's Linux syscall infrastructure.
    fn jolt_syscall(
        a0: usize,
        a1: usize,
        a2: usize,
        a3: usize,
        a4: usize,
        a5: usize,
        nr: usize,
    ) -> isize;
}

/// Exit by entering an infinite loop.
///
/// The emulator detects termination when PC doesn't change between ticks.
/// This is the same mechanism used by no_std guests (`j .` after main returns).
fn jolt_exit(_code: u32) -> ! {
    unsafe {
        core::arch::asm!(
            "j .",  // infinite loop - emulator detects via prev_pc == pc
            options(noreturn)
        );
    }
}

/// Platform trap handler for Jolt guests.
///
/// Routes syscalls through ZeroOS's syscall infrastructure (jolt_syscall),
/// which is proven as part of the guest execution. Exit syscalls are handled
/// specially to terminate the emulator via infinite loop (prev_pc == pc detection).
#[no_mangle]
pub extern "C" fn trap_handler(
    arg0: usize,
    arg1: usize,
    arg2: usize,
    arg3: usize,
    arg4: usize,
    arg5: usize,
    _arg6: usize,
    syscall_nr: usize,
) -> isize {
    const SYS_EXIT: usize = 93;
    const SYS_EXIT_GROUP: usize = 94;

    match syscall_nr {
        SYS_EXIT | SYS_EXIT_GROUP => {
            // Exit syscalls need special handling to terminate the emulator
            jolt_exit(arg0 as u32);
        }
        _ => {
            // Route all other syscalls through ZeroOS's syscall infrastructure
            // This is proven as part of the guest execution
            unsafe { jolt_syscall(arg0, arg1, arg2, arg3, arg4, arg5, syscall_nr) }
        }
    }
}

/// Jolt-specific trap handler using ECALL for CSR operations.
///
/// This overrides the weak `_trap_handler` symbol from arch-riscv.
/// It uses ECALL convention instead of standard RISC-V trap mechanism:
/// - Return address is passed in t1 by ECALL inline sequence (no mepc needed)
/// - Returns via `jalr x0, t1, 0` instead of mret
/// - No CSR operations needed (mcause was unused, mepc replaced by t1)
#[unsafe(naked)]
#[no_mangle]
pub unsafe extern "C" fn _trap_handler() {
    naked_asm!(
        // Entry: t1 contains return address (ECALL_addr+4), passed by ECALL inline sequence

        // Allocate stack frame for: t1 + ra + a0-a7 = 80 bytes
        "   addi    sp, sp, -80",

        // Save t1 (return address) FIRST - it's caller-saved and will be clobbered
        "   sd      t1, 0(sp)",

        // Save ra and argument registers (a0-a7)
        "   sd      ra, 8(sp)",
        "   sd      a0, 16(sp)",
        "   sd      a1, 24(sp)",
        "   sd      a2, 32(sp)",
        "   sd      a3, 40(sp)",
        "   sd      a4, 48(sp)",
        "   sd      a5, 56(sp)",
        "   sd      a6, 64(sp)",
        "   sd      a7, 72(sp)",

        // Call trap_handler with a0-a7 as args
        // No mcause needed - it was never used
        "   call    {handler}",

        // a0 now contains syscall result - keep it there

        // Restore ra and t1 (return address)
        "   ld      ra, 8(sp)",
        "   ld      t1, 0(sp)",

        // Deallocate frame
        "   addi    sp, sp, 80",

        // Return to caller via JALR (no mret needed!)
        // a0 already contains the syscall result
        "   jalr    x0, t1, 0",

        handler = sym trap_handler,
    )
}

/// Entry point for the user's provable function.
///
/// This overrides the weak `__main_entry` from foundation. The jolt macro generates
/// `main()` with no arguments and void return. We wrap it to match the C-style
/// signature expected by `__libc_start_main`.
#[no_mangle]
pub extern "C" fn __main_entry(
    _argc: i32,
    _argv: *const *const u8,
    _envp: *const *const u8,
) -> i32 {
    unsafe { main() };
    0 // Jolt main() returns void; we return 0 as success
}
