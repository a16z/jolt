//! Jolt guest std boot implementation.
//!
//! Provides boot code (_start, kernel_main) for Jolt guest programs compiled with std support.
//! This module is conditionally compiled when the `guest-std` feature is enabled.
//!
//! The boot sequence:
//! 1. _start: Initialize global pointer, stack pointer, and trap handler via `csrw mtvec`
//! 2. kernel_main: Initialize heap, build musl-compatible stack, call __libc_start_main
//! 3. _main_c: Wrapper that calls the user's main() function
//!
//! The trap handler address is set using the proper RISC-V `csrw mtvec` instruction,
//! which Jolt's tracer decodes and maps to virtual register 33 for proof verification.

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

    /// Platform bootstrap function from jolt-platform.
    /// Initializes ZeroOS (registers syscall handlers, VFS, etc.)
    fn __platform_bootstrap();

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
/// This is equivalent to ZeroOS's _default_trap_handler but uses ECALL
/// instead of csrr/csrw instructions, which are not supported in Jolt's emulator.
///
/// Key differences from ZeroOS's version:
/// - ZeroOS uses csrr/csrw which don't clobber registers
/// - Return address is passed in t1 by ECALL inline sequence (no mepc needed)
/// - Returns via JALR t1 instead of RET ECALL
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

        // Return to caller via JALR (no RET ECALL needed!)
        // a0 already contains the syscall result
        "   jalr    x0, t1, 0",

        handler = sym trap_handler,
    )
}

/// Entry point for Jolt guest programs with std support.
///
/// Initializes the global pointer, stack pointer, and trap handler,
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

        // Set up trap handler using csrw
        "   la      t0, {_trap_handler}",
        "   csrw    mtvec, t0",

        // Initialize mscratch
        "   csrw    mscratch, x0",

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

/// Kernel main: Initialize platform, heap, and musl, then call __libc_start_main
///
/// This function:
/// 1. Initializes the platform (ZeroOS syscall handlers, VFS, etc.)
/// 2. Initializes the heap allocator
/// 3. Builds a musl-compatible stack with argc/argv/envp/auxv
/// 4. Calls musl's __libc_start_main which will eventually call the user's main()
#[no_mangle]
extern "C" fn kernel_main() -> ! {
    unsafe {
        // Initialize the platform first - this registers ZeroOS syscall handlers,
        // sets up VFS with console file descriptors, etc.
        __platform_bootstrap();

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
