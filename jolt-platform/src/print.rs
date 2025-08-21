//! Print utility for emulated riscv environment
//! Allows printing strings from within the emulated environment

// Constants to signal the emulator
// PRI in hex (ASCII)
pub const JOLT_PRINT_ECALL_NUM: u32 = 0x505249;
pub const JOLT_PRINT_STRING: u32 = 1;
pub const JOLT_PRINT_LINE: u32 = 2; // with newline

#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
mod riscv_specific {
    use super::{JOLT_PRINT_ECALL_NUM, JOLT_PRINT_LINE, JOLT_PRINT_STRING};

    pub fn print(text: &str) {
        let text_ptr = text.as_ptr() as usize;
        let text_len = text.len();
        emit_jolt_print_ecall(text_ptr as u32, text_len as u32, JOLT_PRINT_STRING);
    }

    pub fn println(text: &str) {
        let text_ptr = text.as_ptr() as usize;
        let text_len = text.len();
        emit_jolt_print_ecall(text_ptr as u32, text_len as u32, JOLT_PRINT_LINE);
    }

    // inserts an ECALL directly into the compiled code
    #[inline(always)]
    fn emit_jolt_print_ecall(text_ptr: u32, text_len: u32, print_type: u32) {
        #[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
        unsafe {
            core::arch::asm!(
            ".word 0x00000073", // ECALL opcode
            in("x10") JOLT_PRINT_ECALL_NUM,  // identifies this as a print syscall
            in("x11") text_ptr,  // pointer to the string in memory
            in("x12") text_len,  // length of the string
            in("x13") print_type, // whether to add newline or not
            options(nostack, nomem, preserves_flags)
            );
        }
    }
}

#[allow(unused_variables)]
pub fn print(text: &str) {
    #[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
    riscv_specific::print(text);
}

#[allow(unused_variables)]
pub fn println(text: &str) {
    #[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
    riscv_specific::println(text);
}

// You might also want formatting support
#[macro_export]
macro_rules! jolt_print {
    ($($arg:tt)*) => {{
        let s = format!($($arg)*);
        $crate::print(&s);
    }};
}

#[macro_export]
macro_rules! jolt_println {
    () => {
        $crate::println("");
    };
    ($($arg:tt)*) => {{
        let s = format!($($arg)*);
        $crate::println(&s);
    }};
}
