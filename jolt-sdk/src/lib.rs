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
pub const FUNCT3_ADVICE_LB: u32 = 0b010; // Load byte from advice tape
#[doc(hidden)]
pub const FUNCT3_ADVICE_LH: u32 = 0b011; // Load halfword from advice tape
#[doc(hidden)]
pub const FUNCT3_ADVICE_LW: u32 = 0b100; // Load word from advice tape
#[doc(hidden)]
pub const FUNCT3_ADVICE_LD: u32 = 0b101; // Load doubleword from advice tape
#[doc(hidden)]
pub const FUNCT3_ADVICE_LEN: u32 = 0b110; // Get number of remaining bytes in advice tape

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
        #[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
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
        #[cfg(not(any(target_arch = "riscv32", target_arch = "riscv64")))]
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

impl AdviceReader {
    // Load a single byte from the advice tape and return it
    #[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
    pub unsafe fn read_lb(&mut self) -> u8 {
        let x;
        core::arch::asm!(
            ".insn i {opcode}, {funct3}, {rd}, x0, 8",
            opcode = const CUSTOM_OPCODE,
            funct3 = const FUNCT3_ADVICE_LB,
            rd = out(reg) x,
            options(nostack)
        );
        x
    }
    // Load a halfword (2 bytes) from the advice tape and return it
    #[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
    pub unsafe fn read_lh(&mut self) -> u16 {
        let x;
        core::arch::asm!(
            ".insn i {opcode}, {funct3}, {rd}, x0, 8",
            opcode = const CUSTOM_OPCODE,
            funct3 = const FUNCT3_ADVICE_LH,
            rd = out(reg) x,
            options(nostack)
        );
        x
    }
    // Load a word (4 bytes) from the advice tape and return it
    #[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
    pub unsafe fn read_lw(&mut self) -> u32 {
        let x;
        core::arch::asm!(
            ".insn i {opcode}, {funct3}, {rd}, x0, 8",
            opcode = const CUSTOM_OPCODE,
            funct3 = const FUNCT3_ADVICE_LW,
            rd = out(reg) x,
            options(nostack)
        );
        x
    }
    // Load a doubleword (8 bytes) from the advice tape and return it
    // on 32-bit targets, this is performed via two 4-byte reads
    #[cfg(any(target_arch = "riscv32"))]
    pub unsafe fn read_ld(&mut self) -> u64 {
        let low = self.read_lw() as u64;
        let high = self.read_lw() as u64;
        (high << 32) | low
    }
    // Load a doubleword (8 bytes) from the advice tape and return it
    // on 64-bit targets, this is a single 8-byte read
    #[cfg(any(target_arch = "riscv64"))]
    pub unsafe fn read_ld(&mut self) -> u64 {
        let x;
        core::arch::asm!(
            ".insn i {opcode}, {funct3}, {rd}, x0, 8",
            opcode = const CUSTOM_OPCODE,
            funct3 = const FUNCT3_ADVICE_LD,
            rd = out(reg) x,
            options(nostack)
        );
        x
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
        // Read as many bytes at once as possible
        if bytes_to_read >= 8 {
            unsafe {
                // On 32-bit architectures this is effectively two 4-byte reads
                // On 64-bit architectures this is a single 8-byte read
                let value = self.read_ld();
                core::ptr::write_unaligned(dst_ptr as *mut u64, value);
            }
            Ok(8)
        } else if bytes_to_read >= 4 {
            unsafe {
                let value = self.read_lw();
                core::ptr::write_unaligned(dst_ptr as *mut u32, value);
            }
            Ok(4)
        } else if bytes_to_read >= 2 {
            unsafe {
                let value = self.read_lh();
                core::ptr::write_unaligned(dst_ptr as *mut u16, value);
            }
            Ok(2)
        } else if bytes_to_read == 1 {
            unsafe {
                let value = self.read_lb();
                core::ptr::write_unaligned(dst_ptr as *mut u8, value);
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

// Trait for writing to and reading from the advice tape
// with default implementation going through AdviceWriter/AdviceReader
pub trait AdviceTapeIO: Sized + Serialize + for<'a> Deserialize<'a> {
    fn write_to_advice_tape(&self) {
        let mut writer = AdviceWriter::get();
        postcard::to_eio(self, &mut writer).expect("Failed to write advice to tape");
    }
    fn new_from_advice_tape() -> Self {
        // Create a scratch buffer for postcard deserialization
        let mut buffer: [u8; 8] = [0u8; 8];
        // create advice reader and read from tape
        let mut reader = AdviceReader::get();
        let (result, _) = postcard::from_eio((&mut reader, &mut buffer))
            .expect("Failed to read advice from tape");
        // return deserialized result
        result
    }
}

#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
use embedded_io::Write;

// Implement AdviceTapeIO for primitive types

impl AdviceTapeIO for u8 {
    #[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
    fn write_to_advice_tape(&self) {
        // write u8 to advice tape directly via ECALL
        let mut writer = AdviceWriter::get();
        AdviceWriter::write(&mut writer, &self.to_le_bytes()).expect("Failed to write advice");
    }
    #[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
    fn new_from_advice_tape() -> Self {
        // read u8 from advice tape directly
        let mut reader = AdviceReader::get();
        unsafe { reader.read_lb() }
    }
}

impl AdviceTapeIO for u16 {
    #[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
    fn write_to_advice_tape(&self) {
        // write u16 to advice tape directly via ECALL
        let mut writer = AdviceWriter::get();
        AdviceWriter::write(&mut writer, &self.to_le_bytes()).expect("Failed to write advice");
    }
    #[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
    fn new_from_advice_tape() -> Self {
        // read u16 from advice tape directly
        let mut reader = AdviceReader::get();
        unsafe { reader.read_lh() }
    }
}

impl AdviceTapeIO for u32 {
    #[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
    fn write_to_advice_tape(&self) {
        // write u32 to advice tape directly via ECALL
        let mut writer = AdviceWriter::get();
        AdviceWriter::write(&mut writer, &self.to_le_bytes()).expect("Failed to write advice");
    }
    #[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
    fn new_from_advice_tape() -> Self {
        // read u32 from advice tape directly
        let mut reader = AdviceReader::get();
        unsafe { reader.read_lw() }
    }
}

impl AdviceTapeIO for u64 {
    #[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
    fn write_to_advice_tape(&self) {
        // write u64 to advice tape directly via ECALL
        let mut writer = AdviceWriter::get();
        AdviceWriter::write(&mut writer, &self.to_le_bytes()).expect("Failed to write advice");
    }
    #[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
    fn new_from_advice_tape() -> Self {
        // read u64 from advice tape directly
        let mut reader = AdviceReader::get();
        unsafe { reader.read_ld() }
    }
}

// Implement AdviceTapeIO for tuples and small arrays of AdviceTapeIO types
// The goal is to avoid Postcard's overhead for these common cases
// TODO, potentially replace these with a macro

impl<S, T> AdviceTapeIO for (S, T)
where
    S: AdviceTapeIO,
    T: AdviceTapeIO,
{
    fn write_to_advice_tape(&self) {
        self.0.write_to_advice_tape();
        self.1.write_to_advice_tape();
    }
    fn new_from_advice_tape() -> Self {
        (S::new_from_advice_tape(), T::new_from_advice_tape())
    }
}

impl<S, T, U> AdviceTapeIO for (S, T, U)
where
    S: AdviceTapeIO,
    T: AdviceTapeIO,
    U: AdviceTapeIO,
{
    fn write_to_advice_tape(&self) {
        self.0.write_to_advice_tape();
        self.1.write_to_advice_tape();
        self.2.write_to_advice_tape();
    }
    fn new_from_advice_tape() -> Self {
        (
            S::new_from_advice_tape(),
            T::new_from_advice_tape(),
            U::new_from_advice_tape(),
        )
    }
}

impl<S> AdviceTapeIO for [S; 2]
where
    S: AdviceTapeIO,
{
    fn write_to_advice_tape(&self) {
        self[0].write_to_advice_tape();
        self[1].write_to_advice_tape();
    }
    fn new_from_advice_tape() -> Self {
        [S::new_from_advice_tape(), S::new_from_advice_tape()]
    }
}

impl<S> AdviceTapeIO for [S; 3]
where
    S: AdviceTapeIO,
{
    fn write_to_advice_tape(&self) {
        self[0].write_to_advice_tape();
        self[1].write_to_advice_tape();
        self[2].write_to_advice_tape();
    }
    fn new_from_advice_tape() -> Self {
        [
            S::new_from_advice_tape(),
            S::new_from_advice_tape(),
            S::new_from_advice_tape(),
        ]
    }
}

impl<S> AdviceTapeIO for [S; 4]
where
    S: AdviceTapeIO,
{
    fn write_to_advice_tape(&self) {
        self[0].write_to_advice_tape();
        self[1].write_to_advice_tape();
        self[2].write_to_advice_tape();
        self[3].write_to_advice_tape();
    }
    fn new_from_advice_tape() -> Self {
        [
            S::new_from_advice_tape(),
            S::new_from_advice_tape(),
            S::new_from_advice_tape(),
            S::new_from_advice_tape(),
        ]
    }
}
