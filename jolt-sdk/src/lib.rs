#![cfg_attr(not(any(feature = "host", feature = "guest-std")), no_std)]

extern crate jolt_sdk_macros;

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
                // Use B-format encoding: opcode=0x5B, funct3=4
                core::arch::asm!(
                    ".insn b 0x5B, 4, {rs1}, {rs2}, 0",
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
#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
pub struct AdviceWriter;

#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
impl AdviceWriter {
    /// Get a reference to the global advice writer.
    #[inline(always)]
    pub fn get() -> Self {
        AdviceWriter
    }
}

#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
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
                in("a0") $crate::JOLT_ADVICE_WRITE_ECALL_NUM,
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

/// Reader for receiving advice data from the host during the proving phase.
///
/// Implements `embedded_io::Read` to allow deserialization of advice data via postcard.
#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
pub struct AdviceReader;

#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
impl AdviceReader {
    /// Get a reference to the global advice reader.
    #[inline(always)]
    pub fn get() -> Self {
        AdviceReader
    }
}

#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
impl embedded_io::ErrorType for AdviceReader {
    type Error = core::convert::Infallible;
}

#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
impl embedded_io::Read for AdviceReader {
    fn read(&mut self, buf: &mut [u8]) -> Result<usize, Self::Error> {
        let len = buf.len();
        let dst_ptr = buf.as_mut_ptr();

        unsafe {
            if len >= 8 {
                // Use ADVICE_SD (store doubleword) - reads 8 bytes from advice tape
                // Format: .insn s opcode, funct3, rs2, imm(rs1)
                // Since rs2 is unused (advice comes from tape), we use x0
                // funct3=3 for SD (doubleword)
                core::arch::asm!(
                    ".insn s 0x5B, 3, x0, 0({rs1})",
                    rs1 = in(reg) dst_ptr,
                    options(nostack)
                );
                Ok(8)
            } else if len >= 4 {
                // Use ADVICE_SW (store word) - reads 4 bytes from advice tape
                // funct3=2 for SW (word)
                core::arch::asm!(
                    ".insn s 0x5B, 2, x0, 0({rs1})",
                    rs1 = in(reg) dst_ptr,
                    options(nostack)
                );
                Ok(4)
            } else if len >= 2 {
                // Use ADVICE_SH (store halfword) - reads 2 bytes from advice tape
                // funct3=1 for SH (halfword)
                core::arch::asm!(
                    ".insn s 0x5B, 1, x0, 0({rs1})",
                    rs1 = in(reg) dst_ptr,
                    options(nostack)
                );
                Ok(2)
            } else if len == 1 {
                // Use ADVICE_SB (store byte) - reads 1 byte from advice tape
                // funct3=0 for SB (byte)
                core::arch::asm!(
                    ".insn s 0x5B, 0, x0, 0({rs1})",
                    rs1 = in(reg) dst_ptr,
                    options(nostack)
                );
                Ok(1)
            } else {
                Ok(0)
            }
        }
    }
}
