#![cfg_attr(not(any(feature = "host", feature = "guest-std")), no_std)]

extern crate jolt_sdk_macros;

// Instruction encoding constants for RISC-V custom instructions
// Note: These are used in inline assembly via `const` keyword, but the compiler
// doesn't recognize that usage, so we suppress the dead_code warning.
#[doc(hidden)]
pub const CUSTOM_OPCODE: u32 = 0x5B; // Custom instructions opcode
#[doc(hidden)]
pub const FUNCT3_VIRTUAL_ASSERT_EQ: u32 = 0b001; // VirtualAssertEQ funct3
#[doc(hidden)]
pub const FUNCT3_ADVICE_SB: u32 = 0b010; // Store byte from advice tape
#[doc(hidden)]
pub const FUNCT3_ADVICE_SH: u32 = 0b011; // Store halfword from advice tape
#[doc(hidden)]
pub const FUNCT3_ADVICE_SW: u32 = 0b100; // Store word from advice tape
#[doc(hidden)]
pub const FUNCT3_ADVICE_SD: u32 = 0b101; // Store doubleword from advice tape
#[doc(hidden)]
pub const FUNCT3_ADVICE_LEN: u32 = 0b110; // Get remaining bytes in advice tape

#[cfg(any(feature = "host", feature = "guest-verifier"))]
pub mod host_utils;
#[cfg(any(feature = "host", feature = "guest-verifier"))]
pub use host_utils::*;

pub use jolt_platform::*;
pub use jolt_sdk_macros::advice;
pub use jolt_sdk_macros::provable;
pub use postcard;

use serde::{Deserialize, Serialize};

/// A wrapper type to mark guest program inputs as trusted_advice.
#[derive(Debug, Serialize, Deserialize)]
#[repr(transparent)]
pub struct TrustedAdvice<T> {
    value: T,
}

impl<T> TrustedAdvice<T> {
    pub fn new(value: T) -> Self {
        Self { value }
    }
}

impl<T> From<T> for TrustedAdvice<T> {
    fn from(value: T) -> Self {
        Self::new(value)
    }
}

impl<T> core::ops::Deref for TrustedAdvice<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

/// A wrapper type to mark guest program inputs as untrusted_advice.
#[derive(Debug, Serialize, Deserialize)]
#[repr(transparent)]
pub struct UntrustedAdvice<T> {
    value: T,
}

impl<T> UntrustedAdvice<T> {
    pub fn new(value: T) -> Self {
        Self { value }
    }
}

impl<T> From<T> for UntrustedAdvice<T> {
    fn from(value: T) -> Self {
        Self::new(value)
    }
}

impl<T> core::ops::Deref for UntrustedAdvice<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

// This is a dummy _HEAP_PTR to keep the compiler happy.
// It should never be used when compiled as a guest or with
// our custom allocator
#[no_mangle]
#[cfg(feature = "host")]
pub static mut _HEAP_PTR: u8 = 0;

/// Runtime advice support
///
/// Provides mechanisms for guest programs to receive non-deterministic advice from the prover
/// and verify it within the zkVM.
///
/// Macro to assert that a condition holds, enforced by the prover.
///
/// This macro generates a VirtualAssertEQ instruction that ensures the given boolean expression
/// evaluates to true. If the condition is false, the proof will fail.
///
/// # Example
/// ```ignore
/// check_advice!(idx.len() == x.len());
/// check_advice!(x[idx[i]] < x[idx[i + 1]]);
/// ```
#[macro_export]
macro_rules! check_advice {
    ($cond:expr) => {{
        #[cfg(all(
            not(feature = "host"),
            any(target_arch = "riscv32", target_arch = "riscv64")
        ))]
        {
            let cond_value = if $cond { 1u64 } else { 0u64 };
            let expected_value = 1u64;
            unsafe {
                // VirtualAssertEQ: assert rs1 == rs2
                // Use B-format encoding with CUSTOM_OPCODE and FUNCT3_VIRTUAL_ASSERT_EQ
                core::arch::asm!(
                    ".insn b {opcode}, {funct3}, {rs1}, {rs2}, 0",
                    opcode = const $crate::CUSTOM_OPCODE,
                    funct3 = const $crate::FUNCT3_VIRTUAL_ASSERT_EQ,
                    rs1 = in(reg) cond_value,
                    rs2 = in(reg) expected_value,
                    options(nostack)
                );
            }
        }
        #[cfg(any(
            feature = "host",
            not(any(target_arch = "riscv32", target_arch = "riscv64"))
        ))]
        {
            assert!($cond, "Advice assertion failed");
        }
    }};
}

/// Writer for sending advice data to the host during the compute_advice phase.
///
/// Implements `embedded_io::Write` to allow serialization of advice data via postcard.
pub struct AdviceWriter;

impl AdviceWriter {
    /// Get a reference to the global advice writer.
    #[inline(always)]
    pub fn get() -> Self {
        AdviceWriter
    }
}

impl embedded_io::ErrorType for AdviceWriter {
    type Error = core::convert::Infallible;
}

#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
impl embedded_io::Write for AdviceWriter {
    fn write(&mut self, buf: &[u8]) -> Result<usize, Self::Error> {
        unsafe {
            let src_ptr = buf.as_ptr() as u64;
            let len = buf.len() as u64;
            core::arch::asm!(
                "ecall",
                in("a0") JOLT_ADVICE_WRITE_ECALL_NUM,
                in("a1") src_ptr,
                in("a2") len,
                options(nostack)
            );
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> Result<(), Self::Error> {
        // Nothing to flush, since write always processes the entire buffer
        Ok(())
    }
}

// Stub implementation for non-RISC-V targets (host builds)
#[cfg(not(any(target_arch = "riscv32", target_arch = "riscv64")))]
impl embedded_io::Write for AdviceWriter {
    fn write(&mut self, _buf: &[u8]) -> Result<usize, Self::Error> {
        // This should never be called on the host
        panic!("AdviceWriter::write() called on non-RISC-V target");
    }

    fn flush(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }
}

/// Reader for receiving advice data from the host during the proving phase.
///
/// Implements `embedded_io::Read` to allow deserialization of advice data via postcard.
pub struct AdviceReader;

impl AdviceReader {
    /// Get a reference to the global advice reader.
    #[inline(always)]
    pub fn get() -> Self {
        AdviceReader
    }
}

impl embedded_io::ErrorType for AdviceReader {
    type Error = core::convert::Infallible;
}

#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
impl embedded_io::Read for AdviceReader {
    fn read(&mut self, buf: &mut [u8]) -> Result<usize, Self::Error> {
        if buf.is_empty() {
            return Ok(0);
        }

        // First, query how many bytes remain in the advice tape
        let remaining: u64;
        unsafe {
            // VirtualAdviceLen uses custom opcode with funct3 encoding
            // Encode as I-format: opcode | rd | funct3 | rs1=x0 | imm=0
            core::arch::asm!(
                ".insn i {opcode}, {funct3}, {rd}, x0, 0",
                opcode = const CUSTOM_OPCODE,
                funct3 = const FUNCT3_ADVICE_LEN,
                rd = out(reg) remaining,
                options(nostack)
            );
        }

        // Read up to min(buf.len(), remaining) bytes
        let bytes_to_read = core::cmp::min(buf.len(), remaining as usize);
        let dst_ptr = buf.as_mut_ptr();
        if bytes_to_read >= 8 {
            unsafe {
                core::arch::asm!(
                    ".insn s {opcode}, {funct3}, x0, 0({rs1})",
                    opcode = const CUSTOM_OPCODE,
                    funct3 = const FUNCT3_ADVICE_SD,
                    rs1 = in(reg) dst_ptr,
                    options(nostack)
                );
            }
            Ok(8)
        } else if bytes_to_read >= 4 {
            unsafe {
                core::arch::asm!(
                    ".insn s {opcode}, {funct3}, x0, 0({rs1})",
                    opcode = const CUSTOM_OPCODE,
                    funct3 = const FUNCT3_ADVICE_SW,
                    rs1 = in(reg) dst_ptr,
                    options(nostack)
                );
            }
            Ok(4)
        } else if bytes_to_read >= 2 {
            unsafe {
                core::arch::asm!(
                    ".insn s {opcode}, {funct3}, x0, 0({rs1})",
                    opcode = const CUSTOM_OPCODE,
                    funct3 = const FUNCT3_ADVICE_SH,
                    rs1 = in(reg) dst_ptr,
                    options(nostack)
                );
            }
            Ok(2)
        } else if bytes_to_read == 1 {
            unsafe {
                core::arch::asm!(
                    ".insn s {opcode}, {funct3}, x0, 0({rs1})",
                    opcode = const CUSTOM_OPCODE,
                    funct3 = const FUNCT3_ADVICE_SB,
                    rs1 = in(reg) dst_ptr,
                    options(nostack)
                );
            }
            Ok(1)
        } else {
            Ok(0)
        }
    }
}

// Stub implementation for non-RISC-V targets (host builds)
#[cfg(not(any(target_arch = "riscv32", target_arch = "riscv64")))]
impl embedded_io::Read for AdviceReader {
    fn read(&mut self, _buf: &mut [u8]) -> Result<usize, Self::Error> {
        // This should never be called on the host
        panic!("AdviceReader::read() called on non-RISC-V target");
    }
}
