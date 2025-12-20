//! Jolt guest std boot implementation.
//!
//! Provides boot code (_start, kernel_main) for Jolt guest programs compiled with std support.
//! This module is conditionally compiled when the `guest-std` feature is enabled.
//!
//! The boot sequence:
//! 1. _start: Initialize global pointer, stack pointer, and trap handler via ECALL
//! 2. kernel_main: Initialize heap, build musl-compatible stack, call __libc_start_main
//! 3. _main_c: Wrapper that calls the user's main() function
//!
//! This uses the Jolt-specific approach with ECALL for setting mtvec (not direct csrw),
//! which is required for Jolt's execution environment.

use core::arch::naked_asm;
use linked_list_allocator::LockedHeap;

#[global_allocator]
static ALLOCATOR: LockedHeap = LockedHeap::empty();

extern "C" {
    // User's main function (provided by the jolt macro - returns void, not i32)
    fn main();
}

extern "C" {
    /// Musl's __libc_start_main initializes libc/TLS/init arrays
    fn __libc_start_main(
        main_fn: extern "C" fn(i32, *mut *mut u8, *mut *mut u8) -> i32,
        argc: i32,
        argv: *mut *mut u8,
        init: extern "C" fn(),
        fini: extern "C" fn(),
        ldso_dummy: *const u8,
    ) -> !;

    static __stack_top: u8;
    static __stack_bottom: u8;
    static __ehdr_start: u8;
}

use zeroos_runtime_musl::build_musl_stack;

static PROGRAM_NAME: &[u8] = b"jolt-guest\0";

/// C entry point wrapper that calls Rust main and returns exit code 0
///
/// The jolt macro generates `fn main()` that returns void (writes to termination bit),
/// so we call it and return 0 as the exit code.
#[no_mangle]
extern "C" fn _main_c(_argc: i32, _argv: *mut *mut u8, _envp: *mut *mut u8) -> i32 {
    unsafe { main() };
    0  // Jolt main() returns void; we return 0 as success
}

#[no_mangle]
pub extern "C" fn _init() {}

#[no_mangle]
pub extern "C" fn _fini() {}

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
/// Handles syscalls that need special processing, particularly exit syscalls
/// which must use the JOLT_EXIT_ECALL mechanism to properly terminate the tracer.
#[no_mangle]
pub extern "C" fn trap_handler(
    arg0: usize,
    _arg1: usize,
    _arg2: usize,
    _arg3: usize,
    _arg4: usize,
    _arg5: usize,
    _arg6: usize,
    syscall_nr: usize,
    _mcause: usize,
) -> isize {
    const SYS_EXIT: usize = 93;
    const SYS_EXIT_GROUP: usize = 94;

    match syscall_nr {
        SYS_EXIT | SYS_EXIT_GROUP => {
            // Convert Linux exit syscall to Jolt's EXIT ECALL
            jolt_exit(arg0 as u32);
        }
        _ => {
            // Other syscalls are handled by the emulator directly
            0
        }
    }
}

/// Jolt-specific trap handler using ECALL for CSR operations.
///
/// This is equivalent to ZeroOS's _default_trap_handler but uses ECALL
/// instead of csrr/csrw instructions, which are not supported in Jolt's emulator.
///
/// Key differences from ZeroOS's version:
/// - ZeroOS uses csrr/csrw which don't clobber registers
/// - We use ECALL which clobbers a0-a3 (and returns in a0)
/// - So we must save/restore a0-a7 around the ECALLs
/// - Uses RET ECALL (0x524554) instead of mret to return from trap
#[unsafe(naked)]
#[no_mangle]
pub unsafe extern "C" fn _trap_handler() {
    naked_asm!(
        // Allocate stack frame for: mcause + ra + a0-a7 (8 slots) = 80 bytes
        "   addi    sp, sp, -80",

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

        // Read mcause via ECALL (clobbers a0-a3)
        "   li      a0, 0x435352",  // CSR_ECALL_NUM
        "   li      a1, 1",          // CSR_OP_READ
        "   li      a2, 0x342",      // CSR_MCAUSE
        "   li      a3, 0",
        "   ecall",
        "   sd      a0, 0(sp)",      // Save mcause at sp+0 (9th arg for trap_handler)

        // Read mepc via ECALL
        "   li      a0, 0x435352",  // CSR_ECALL_NUM
        "   li      a1, 1",          // CSR_OP_READ
        "   li      a2, 0x341",      // CSR_MEPC
        "   li      a3, 0",
        "   ecall",
        "   mv      t0, a0",         // Save mepc in t0

        // Increment mepc by 4 (ecall is always 32-bit)
        "   addi    t0, t0, 4",

        // Write mepc via ECALL
        "   li      a0, 0x435352",  // CSR_ECALL_NUM
        "   li      a1, 2",          // CSR_OP_WRITE
        "   li      a2, 0x341",      // CSR_MEPC
        "   mv      a3, t0",
        "   ecall",

        // Restore argument registers a0-a7
        "   ld      a0, 16(sp)",
        "   ld      a1, 24(sp)",
        "   ld      a2, 32(sp)",
        "   ld      a3, 40(sp)",
        "   ld      a4, 48(sp)",
        "   ld      a5, 56(sp)",
        "   ld      a6, 64(sp)",
        "   ld      a7, 72(sp)",

        // Call trap_handler with a0-a7 as args, mcause at sp+0 as 9th arg
        "   call    {handler}",

        // Save trap_handler return value (syscall result) in t0
        "   mv      t0, a0",

        // Restore ra, deallocate frame
        "   ld      ra, 8(sp)",
        "   addi    sp, sp, 80",

        // Return from trap using RET ECALL (instead of mret)
        // mepc was already updated by the trap handler
        "   li      a0, 0x524554",  // "RET" in ASCII - return from trap
        "   li      a7, 0",          // Clear a7 so this is treated as special ecall, not syscall
        "   ecall",

        // After RET ECALL returns, restore the syscall result to a0
        "   mv      a0, t0",

        handler = sym trap_handler,
    )
}

/// Jolt-specific _start that uses ECALL for setting mtvec
///
/// This is the entry point for Jolt guest programs with std support.
/// It initializes the global pointer, stack pointer, and trap handler,
/// then jumps to kernel_main for musl initialization.
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

        // Set up trap handler using CSR ECALL instead of csrw
        "   la      t0, {_trap_handler}",

        // Call CSR ECALL to set mtvec (Jolt-specific mechanism)
        "   lui      a0, 0x435",
        "   addi     a0, a0, 0x352",     // a0 = 0x435352 (CSR_ECALL_NUM)
        "   addi     a1, zero, 2",       // a1 = 2 (CSR_OP_WRITE)
        "   addi     a2, zero, 0x305",   // a2 = 0x305 (CSR_MTVEC)
        "   addi     a3, t0, 0",         // a3 = trap handler address
        "   ecall",

        "   tail    {kmain}",

        _trap_handler = sym _trap_handler,
        kmain = sym kernel_main,
    )
}

/// Initialize the global heap allocator
///
/// This must be called before any heap allocations are made.
/// The heap region is defined by the linker script symbols __heap_start and __heap_end.
fn init_heap() {
    extern "C" {
        static __heap_start: u8;
        static __heap_end: u8;
    }

    unsafe {
        let heap_start = core::ptr::addr_of!(__heap_start) as usize;
        let heap_end = core::ptr::addr_of!(__heap_end) as usize;
        let heap_size = heap_end - heap_start;

        ALLOCATOR.lock().init(heap_start as *mut u8, heap_size);
    }
}

/// Kernel main: Initialize heap and musl, then call __libc_start_main
///
/// This function:
/// 1. Initializes the heap allocator
/// 2. Builds a musl-compatible stack with argc/argv/envp/auxv
/// 3. Calls musl's __libc_start_main which will eventually call the user's main()
#[no_mangle]
extern "C" fn kernel_main() -> ! {
    unsafe {
        // Initialize the heap allocator before musl starts
        init_heap();

        // Get stack bounds from linker symbols
        let stack_top = core::ptr::addr_of!(__stack_top) as usize;
        let stack_bottom = core::ptr::addr_of!(__stack_bottom) as usize;
        let ehdr_start = core::ptr::addr_of!(__ehdr_start) as usize;

        // ZeroOS API: build_musl_stack returns size used, not new SP
        let size_used = build_musl_stack(stack_top, stack_bottom, ehdr_start, PROGRAM_NAME);
        let musl_sp = stack_top - size_used;

        core::arch::asm!(
            // Switch to new stack
            "   mv      sp, {new_sp}",

            // Read argc and argv from new stack (as build_musl_stack left them)
            "   lw      a1, 0(sp)",      // a1 = argc
            "   addi    a2, sp, 8",      // a2 = argv

            // Prepare arguments for __libc_start_main(main_fn, argc, argv, init, fini, ldso_dummy)
            "   la      a0, {main_c}",   // a0 = main_fn
            "   la      a3, {init}",     // a3 = _init
            "   la      a4, {fini}",     // a4 = _fini
            "   li      a5, 0",          // a5 = NULL (ldso_dummy)

            // Call __libc_start_main (should never return)
            "   call    {libc_start}",

            // Fallback infinite loop
            "   j       .",

            new_sp = in(reg) musl_sp,
            main_c = sym _main_c,
            init = sym _init,
            fini = sym _fini,
            libc_start = sym __libc_start_main,
            options(noreturn),
        );
    }
}
