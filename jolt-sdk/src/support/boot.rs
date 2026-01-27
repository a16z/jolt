//! Jolt platform boot code for ZeroOS integration
//!
//! Provides platform initialization for Jolt zkVM guests using ZeroOS.

// Debug macros from zeroos-debug crate - re-exported via zeroos
// These are no-ops when debug feature is disabled
macro_rules! debug_writeln {
    ($($arg:tt)*) => {
        // Debug output disabled in jolt-sdk to avoid external dependency
        // In production builds, this is a no-op
    };
}

extern "C" {
    static __heap_start: u8;
    static __heap_end: u8;
    static __stack_top: u8;
    static __stack_bottom: u8;
}

/// Install the trap vector (mtvec = _trap_handler).
/// This is called during platform bootstrap when os-linux feature is enabled.
#[inline(always)]
#[cfg(all(target_arch = "riscv64", target_os = "linux"))]
fn install_trap_vector() {
    #[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
    unsafe {
        core::arch::asm!("la t0, _trap_handler", "csrw mtvec, t0");
    }
}

#[no_mangle]
pub extern "C" fn __platform_bootstrap() {
    debug_writeln!("[BOOT] __platform_bootstrap (jolt-platform)");

    zeroos::initialize();

    {
        let heap_start = core::ptr::addr_of!(__heap_start) as usize;
        let heap_end = core::ptr::addr_of!(__heap_end) as usize;
        debug_writeln!("[BOOT] Heap start=0x{:x}, end=0x{:x}", heap_start, heap_end);
        let heap_size = heap_end - heap_start;
        zeroos::foundation::kfn::memory::kinit(heap_start, heap_size);

        let _stack_top = core::ptr::addr_of!(__stack_top) as usize;
        let _stack_bottom = core::ptr::addr_of!(__stack_bottom) as usize;
        debug_writeln!(
            "[BOOT] Stack top=0x{:x}, bottom=0x{:x}",
            _stack_top,
            _stack_bottom
        );
    }

    #[cfg(not(target_os = "none"))]
    {
        {
            install_trap_vector();
            debug_writeln!("[BOOT] Trap handler installed");
        }

        #[cfg(feature = "zeroos-thread")]
        let boot_thread_anchor: usize = {
            let anchor = zeroos::foundation::kfn::scheduler::kinit();

            // Prime the current tp with the returned anchor and set mscratch to 0.
            // In kernel mode, mscratch must be 0 to correctly identify traps from user mode.
            #[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
            unsafe {
                core::arch::asm!("mv tp, {0}", in(reg) anchor);
                core::arch::asm!("csrw mscratch, x0");
            }

            anchor
        };

        #[cfg(feature = "zeroos-vfs")]
        {
            zeroos::foundation::kfn::vfs::kinit();

            #[cfg(feature = "zeroos-vfs-device-console")]
            {
                debug_writeln!("[BOOT] Registering console file descriptors");
                register_console_fd(1, &STDOUT_FOPS);
                register_console_fd(2, &STDERR_FOPS);
            }
        }

        #[cfg(feature = "zeroos-random")]
        {
            // SECURITY: RNG seed is fixed (0) for deterministic ZK proofs.
            // This is intentional for zkVM - proofs must be reproducible.
            zeroos::foundation::kfn::random::kinit(0);
        }

        // Before entering libc: leave tp for TLS (musl owns it) and park anchor in mscratch,
        // so a user trap swaps the anchor into tp on entry.
        #[cfg(feature = "zeroos-thread")]
        {
            #[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
            unsafe {
                core::arch::asm!("csrw mscratch, {0}", in(reg) boot_thread_anchor);
                core::arch::asm!("mv tp, x0");
            }
        }
    }
}

#[cfg(feature = "zeroos-vfs-device-console")]
mod console {
    use super::super::ecall::putchar;
    use zeroos::vfs;

    pub fn jolt_console_write(_file: *mut u8, buf: *const u8, count: usize) -> isize {
        // Debug output disabled
        unsafe {
            let slice = core::slice::from_raw_parts(buf, count);
            for &byte in slice {
                putchar(byte);
            }
        }
        count as isize
    }

    pub fn register_console_fd(fd: i32, ops: &'static vfs::FileOps) {
        // Debug output disabled
        let _ = vfs::register_fd(
            fd,
            vfs::FdEntry {
                ops,
                private_data: core::ptr::null_mut(),
            },
        );
    }

    pub static STDOUT_FOPS: vfs::FileOps = vfs::devices::console::stdout_fops(jolt_console_write);
    pub static STDERR_FOPS: vfs::FileOps = vfs::devices::console::stderr_fops(jolt_console_write);
}

#[cfg(feature = "zeroos-vfs-device-console")]
use console::{register_console_fd, STDERR_FOPS, STDOUT_FOPS};
