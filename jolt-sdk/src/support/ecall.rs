//! ECALL-based I/O for Jolt zkVM
//!
//! Provides print/write functionality using Jolt's special ECALL mechanism.
//! These ECALLs are intercepted by the Jolt emulator (not routed through
//! the guest trap handler) to avoid infinite loops in syscall handling.
//!
//! The special ECALLs use a0 for the operation type (with a7=0), which
//! distinguishes them from Linux syscalls (which use a7 for syscall number).

use core::fmt::{self, Write};

/// Jolt special ECALL numbers (placed in a0, with a7=0)
const JOLT_PRINT_ECALL_NUM: usize = 0x505249; // "PRI" in ASCII
const JOLT_PRINT_STRING: usize = 1;

/// Write a buffer using Jolt's PRINT ECALL.
///
/// This ECALL is intercepted directly by the Jolt emulator and not
/// routed through the guest's trap handler, preventing infinite loops
/// when syscall handling is done in the guest.
#[inline(always)]
fn jolt_print_ecall(buf: *const u8, len: usize) {
    unsafe {
        core::arch::asm!(
            "ecall",
            in("a0") JOLT_PRINT_ECALL_NUM,
            in("a1") buf,
            in("a2") len,
            in("a3") JOLT_PRINT_STRING,
            in("a7") 0usize,  // a7=0 marks this as a special ECALL, not a syscall
            options(nostack)
        );
    }
}

/// Write a single character to stdout
pub fn putchar(c: u8) {
    let buf = [c];
    jolt_print_ecall(buf.as_ptr(), 1);
}

/// Write a string to stdout
pub fn puts(s: &str) {
    jolt_print_ecall(s.as_ptr(), s.len());
}

/// Writer for stdout
pub struct StdoutWriter;

impl Write for StdoutWriter {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        jolt_print_ecall(s.as_ptr(), s.len());
        Ok(())
    }
}

/// Writer for stderr (uses same ECALL as stdout for Jolt)
pub struct StderrWriter;

impl Write for StderrWriter {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        jolt_print_ecall(s.as_ptr(), s.len());
        Ok(())
    }
}

/// Print to stdout (Jolt version)
#[macro_export]
macro_rules! jolt_print {
    ($($arg:tt)*) => {{
        use core::fmt::Write;
        let _ = write!($crate::support::ecall::StdoutWriter, $($arg)*);
    }};
}

/// Print to stdout with newline (Jolt version)
#[macro_export]
macro_rules! jolt_println {
    () => {{
        $crate::jolt_print!("\n");
    }};
    ($($arg:tt)*) => {{
        use core::fmt::Write;
        let _ = writeln!($crate::support::ecall::StdoutWriter, $($arg)*);
    }};
}

/// Print to stdout (platform-generic name) - for zeroos feature
#[macro_export]
macro_rules! println {
    () => {{
        $crate::jolt_print!("\n");
    }};
    ($($arg:tt)*) => {{
        use core::fmt::Write;
        let _ = writeln!($crate::support::ecall::StdoutWriter, $($arg)*);
    }};
}

/// Print to stderr (platform-generic name) - for zeroos feature
#[macro_export]
macro_rules! eprintln {
    () => {{
        use core::fmt::Write;
        let _ = writeln!($crate::support::ecall::StderrWriter);
    }};
    ($($arg:tt)*) => {{
        use core::fmt::Write;
        let _ = writeln!($crate::support::ecall::StderrWriter, $($arg)*);
    }};
}
