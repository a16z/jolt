//! Print utilities for Jolt guests
//!
//! Provides print functionality that works across all build configurations:
//! - riscv: Uses VirtualHostIO instruction handled by the Jolt emulator
//! - non-riscv + std: Falls back to std::print!/println!
//! - non-riscv + no_std: No-op (no output mechanism available)

use core::fmt::{self, Write};

// ============================================================================
// Constants used by tracer to identify print operations
// ============================================================================

/// Identifier for Jolt print operations (placed in a0)
/// "PRI" in ASCII
pub const JOLT_PRINT_CALL_ID: u32 = 0x505249;

/// Print string without newline
pub const JOLT_PRINT_STRING: u32 = 1;

/// Print string with newline
pub const JOLT_PRINT_LINE: u32 = 2;

// ============================================================================
// VirtualHostIO-based printing for riscv targets
// ============================================================================

/// Write a buffer using Jolt's VirtualHostIO instruction.
///
/// This instruction is handled directly by the Jolt emulator as a no-op
/// virtual instruction, not routed through the guest's trap handler,
/// preventing infinite loops when syscall handling is done in the guest.
#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
#[inline(always)]
fn emit_jolt_print(buf: *const u8, len: usize) {
    unsafe {
        core::arch::asm!(
            ".insn i 0x5B, 2, x0, x0, 0", // VirtualHostIO (opcode=0x5B, funct3=2)
            in("a0") JOLT_PRINT_CALL_ID as usize,
            in("a1") buf,
            in("a2") len,
            in("a3") JOLT_PRINT_STRING as usize,
            options(nostack, preserves_flags)
        );
    }
}

/// Write a single character to stdout (riscv)
#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
pub fn putchar(c: u8) {
    let buf = [c];
    emit_jolt_print(buf.as_ptr(), 1);
}

/// Write a string to stdout (riscv)
#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
pub fn puts(s: &str) {
    emit_jolt_print(s.as_ptr(), s.len());
}

// ============================================================================
// Writers implementing core::fmt::Write
// ============================================================================

/// Writer for stdout
pub struct StdoutWriter;

impl Write for StdoutWriter {
    #[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
    fn write_str(&mut self, s: &str) -> fmt::Result {
        emit_jolt_print(s.as_ptr(), s.len());
        Ok(())
    }

    #[cfg(all(
        not(any(target_arch = "riscv32", target_arch = "riscv64")),
        feature = "std"
    ))]
    fn write_str(&mut self, s: &str) -> fmt::Result {
        std::print!("{s}");
        Ok(())
    }

    #[cfg(all(
        not(any(target_arch = "riscv32", target_arch = "riscv64")),
        not(feature = "std")
    ))]
    fn write_str(&mut self, _s: &str) -> fmt::Result {
        // No output mechanism available for non-riscv no_std
        Ok(())
    }
}

/// Writer for stderr
pub struct StderrWriter;

impl Write for StderrWriter {
    #[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
    fn write_str(&mut self, s: &str) -> fmt::Result {
        // Jolt uses same VirtualHostIO for stdout and stderr
        emit_jolt_print(s.as_ptr(), s.len());
        Ok(())
    }

    #[cfg(all(
        not(any(target_arch = "riscv32", target_arch = "riscv64")),
        feature = "std"
    ))]
    fn write_str(&mut self, s: &str) -> fmt::Result {
        std::eprint!("{s}");
        Ok(())
    }

    #[cfg(all(
        not(any(target_arch = "riscv32", target_arch = "riscv64")),
        not(feature = "std")
    ))]
    fn write_str(&mut self, _s: &str) -> fmt::Result {
        // No output mechanism available for non-riscv no_std
        Ok(())
    }
}

// ============================================================================
// Print macros
// ============================================================================

/// Print to stdout
#[macro_export]
macro_rules! print {
    ($($arg:tt)*) => {{
        use core::fmt::Write;
        let _ = write!($crate::StdoutWriter, $($arg)*);
    }};
}

/// Print to stdout with newline
#[macro_export]
macro_rules! println {
    () => {{
        $crate::print!("\n");
    }};
    ($($arg:tt)*) => {{
        use core::fmt::Write;
        let _ = writeln!($crate::StdoutWriter, $($arg)*);
    }};
}

/// Print to stderr
#[macro_export]
macro_rules! eprint {
    ($($arg:tt)*) => {{
        use core::fmt::Write;
        let _ = write!($crate::StderrWriter, $($arg)*);
    }};
}

/// Print to stderr with newline
#[macro_export]
macro_rules! eprintln {
    () => {{
        use core::fmt::Write;
        let _ = writeln!($crate::StderrWriter);
    }};
    ($($arg:tt)*) => {{
        use core::fmt::Write;
        let _ = writeln!($crate::StderrWriter, $($arg)*);
    }};
}
