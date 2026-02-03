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
pub struct AdviceWriter;

impl AdviceWriter {
    /// Get a reference to the global advice writer.
    #[inline(always)]
    pub fn get() -> Self {
        AdviceWriter
    }
    /// Write a slice of bytes to the advice tape.
    #[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
    fn write_bytes(&mut self, buf: &[u8]) -> usize {
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
        buf.len()
    }
    #[cfg(not(any(target_arch = "riscv32", target_arch = "riscv64")))]
    fn write_bytes(&mut self, _buf: &[u8]) -> usize {
        panic!("Advice tape IO is not supported on non-RISC-V targets");
    }
    // Write a single byte to the advice tape
    pub fn write_u8(&mut self, value: u8) {
        self.write_bytes(&value.to_le_bytes());
    }
    // Write a halfword (2 bytes) to the advice tape
    pub fn write_u16(&mut self, value: u16) {
        self.write_bytes(&value.to_le_bytes());
    }
    // Write a word (4 bytes) to the advice tape
    pub fn write_u32(&mut self, value: u32) {
        self.write_bytes(&value.to_le_bytes());
    }
    // Write a doubleword (8 bytes) to the advice tape
    pub fn write_u64(&mut self, value: u64) {
        self.write_bytes(&value.to_le_bytes());
    }
}

/// Reader for receiving advice data from the host during the proving phase.
pub struct AdviceReader;

impl AdviceReader {
    /// Get a reference to the global advice reader.
    #[inline(always)]
    pub fn get() -> Self {
        AdviceReader
    }
    // Load a single byte from the advice tape and return it
    #[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
    pub fn read_u8(&mut self) -> u8 {
        let x;
        unsafe {
            core::arch::asm!(
                ".insn i {opcode}, {funct3}, {rd}, x0, 0",
                opcode = const CUSTOM_OPCODE,
                funct3 = const FUNCT3_ADVICE_LB,
                rd = out(reg) x,
                options(nostack)
            );
        }
        x
    }
    #[cfg(not(any(target_arch = "riscv32", target_arch = "riscv64")))]
    pub fn read_u8(&mut self) -> u8 {
        panic!("Advice tape IO is not supported on non-RISC-V targets");
    }
    // Load a halfword (2 bytes) from the advice tape and return it
    #[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
    pub fn read_u16(&mut self) -> u16 {
        let x;
        unsafe {
            core::arch::asm!(
                ".insn i {opcode}, {funct3}, {rd}, x0, 0",
                opcode = const CUSTOM_OPCODE,
                funct3 = const FUNCT3_ADVICE_LH,
                rd = out(reg) x,
                options(nostack)
            );
        }
        x
    }
    #[cfg(not(any(target_arch = "riscv32", target_arch = "riscv64")))]
    pub fn read_u16(&mut self) -> u16 {
        panic!("Advice tape IO is not supported on non-RISC-V targets");
    }
    // Load a word (4 bytes) from the advice tape and return it
    #[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
    pub fn read_u32(&mut self) -> u32 {
        let x;
        unsafe {
            core::arch::asm!(
                ".insn i {opcode}, {funct3}, {rd}, x0, 0",
                opcode = const CUSTOM_OPCODE,
                funct3 = const FUNCT3_ADVICE_LW,
                rd = out(reg) x,
                options(nostack)
            );
        }
        x
    }
    #[cfg(not(any(target_arch = "riscv32", target_arch = "riscv64")))]
    pub fn read_u32(&mut self) -> u32 {
        panic!("Advice tape IO is not supported on non-RISC-V targets");
    }
    // Load a doubleword (8 bytes) from the advice tape and return it
    // on 32-bit targets, this is performed via two 4-byte reads
    #[cfg(target_arch = "riscv32")]
    pub fn read_u64(&mut self) -> u64 {
        let low = self.read_u32() as u64;
        let high = self.read_u32() as u64;
        (high << 32) | low
    }
    // Load a doubleword (8 bytes) from the advice tape and return it
    // on 64-bit targets, this is a single 8-byte read
    #[cfg(target_arch = "riscv64")]
    pub fn read_u64(&mut self) -> u64 {
        let x;
        unsafe {
            core::arch::asm!(
                ".insn i {opcode}, {funct3}, {rd}, x0, 8",
                opcode = const CUSTOM_OPCODE,
                funct3 = const FUNCT3_ADVICE_LD,
                rd = out(reg) x,
                options(nostack)
            );
        }
        x
    }
    #[cfg(not(any(target_arch = "riscv32", target_arch = "riscv64")))]
    pub fn read_u64(&mut self) -> u64 {
        panic!("Advice tape IO is not supported on non-RISC-V targets");
    }
    // Get the number of remaining bytes in the advice tape
    #[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
    pub fn bytes_remaining(&mut self) -> u64 {
        let remaining: u64;
        // VirtualAdviceLen uses custom opcode with funct3 encoding
        // Encode as I-format: opcode | rd | funct3 | rs1=x0 | imm=0
        unsafe {
            core::arch::asm!(
                ".insn i {opcode}, {funct3}, {rd}, x0, 0",
                opcode = const CUSTOM_OPCODE,
                funct3 = const FUNCT3_ADVICE_LEN,
                rd = out(reg) remaining,
                options(nostack)
            );
        }
        remaining
    }
    #[cfg(not(any(target_arch = "riscv32", target_arch = "riscv64")))]
    pub fn bytes_remaining(&mut self) -> u64 {
        panic!("Advice tape IO is not supported on non-RISC-V targets");
    }
    // Fill the provided buffer with advice data read from the advice tape
    // Attempts to read as much data as possible per instruction
    // As with the instructions above, reading beyond the end of the advice tape
    // will result in a runtime error during proof generation
    #[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
    fn read_slice(&mut self, buf: &mut [u8]) {
        let mut ptr = buf.as_mut_ptr();
        let mut remaining = buf.len();
        unsafe {
            while remaining >= 8 {
                core::ptr::write_unaligned(ptr as *mut u64, self.read_u64());
                ptr = ptr.add(8);
                remaining -= 8;
            }

            if remaining >= 4 {
                core::ptr::write_unaligned(ptr as *mut u32, self.read_u32());
                ptr = ptr.add(4);
                remaining -= 4;
            }

            if remaining >= 2 {
                core::ptr::write_unaligned(ptr as *mut u16, self.read_u16());
                ptr = ptr.add(2);
                remaining -= 2;
            }

            if remaining == 1 {
                core::ptr::write_unaligned(ptr, self.read_u8());
            }
        }
    }
    #[cfg(not(any(target_arch = "riscv32", target_arch = "riscv64")))]
    fn read_slice(&mut self, _buf: &mut [u8]) {
        panic!("Advice tape IO is not supported on non-RISC-V targets");
    }
}

// Trait for writing to and reading from the advice tape
pub trait AdviceTapeIO: Sized {
    fn write_to_advice_tape(&self) {
        panic!("Advice tape IO not implemented for this type/target");
    }
    fn new_from_advice_tape() -> Self {
        panic!("Advice tape IO not implemented for this type/target");
    }
}

// Implement AdviceTapeIO for primitive types
impl AdviceTapeIO for u8 {
    fn write_to_advice_tape(&self) {
        // write u8 to advice tape directly via ECALL
        let mut writer = AdviceWriter::get();
        writer.write_u8(*self);
    }
    fn new_from_advice_tape() -> Self {
        // read u8 from advice tape directly
        let mut reader = AdviceReader::get();
        reader.read_u8()
    }
}

impl AdviceTapeIO for u16 {
    fn write_to_advice_tape(&self) {
        // write u16 to advice tape directly via ECALL
        let mut writer = AdviceWriter::get();
        writer.write_u16(*self);
    }
    fn new_from_advice_tape() -> Self {
        // read u16 from advice tape directly
        let mut reader = AdviceReader::get();
        reader.read_u16()
    }
}

impl AdviceTapeIO for u32 {
    fn write_to_advice_tape(&self) {
        // write u32 to advice tape directly via ECALL
        let mut writer = AdviceWriter::get();
        writer.write_u32(*self);
    }
    fn new_from_advice_tape() -> Self {
        // read u32 from advice tape directly
        let mut reader = AdviceReader::get();
        reader.read_u32()
    }
}

impl AdviceTapeIO for u64 {
    fn write_to_advice_tape(&self) {
        // write u64 to advice tape directly via ECALL
        let mut writer = AdviceWriter::get();
        writer.write_u64(*self);
    }
    fn new_from_advice_tape() -> Self {
        // read u64 from advice tape directly
        let mut reader = AdviceReader::get();
        reader.read_u64()
    }
}

impl AdviceTapeIO for usize {
    fn write_to_advice_tape(&self) {
        // write usize to advice tape directly via ECALL
        let mut writer = AdviceWriter::get();
        // to_le_bytes implicitly handles 32 vs 64 bit usize
        AdviceWriter::write_bytes(&mut writer, &self.to_le_bytes());
    }
    fn new_from_advice_tape() -> Self {
        // if usize is 32 bits, read u32; if 64 bits, read u64
        let mut reader = AdviceReader::get();
        #[cfg(target_pointer_width = "32")]
        {
            reader.read_u32() as usize
        }
        #[cfg(target_pointer_width = "64")]
        {
            reader.read_u64() as usize
        }
    }
}

// Implement AdviceTapeIO for tuples and small arrays of AdviceTapeIO types

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

// Implement AdviceTapeIO for Vecs of u8s, u16s, u32s, u64s, and usizes
// Optimized to read/write in bulk rather than element-by-element
#[cfg(any(feature = "host", feature = "guest-std"))]
impl AdviceTapeIO for Vec<u8> {
    fn write_to_advice_tape(&self) {
        // Write the length of the Vec<u8> first
        self.len().write_to_advice_tape();
        // Then write the contents of the Vec<u8> to the advice tape
        let mut writer = AdviceWriter::get();
        AdviceWriter::write_bytes(&mut writer, self);
    }
    fn new_from_advice_tape() -> Self {
        // First read the length of the advice data
        let len = usize::new_from_advice_tape();
        // Create a vec of u8s with length len
        let mut buf = Vec::with_capacity(len);
        // Read the contents into the vec directly
        let mut reader = AdviceReader::get();
        AdviceReader::read_slice(&mut reader, &mut buf);
        // Adjust the length of the Vec<u8> after reading
        unsafe {
            buf.set_len(len);
        }
        // Return the filled Vec<u8>
        buf
    }
}

#[cfg(any(feature = "host", feature = "guest-std"))]
impl AdviceTapeIO for Vec<u16> {
    fn write_to_advice_tape(&self) {
        // Write the length of the Vec<u16> first
        self.len().write_to_advice_tape();
        // Then write the contents of the Vec<u16> to the advice tape
        let mut writer = AdviceWriter::get();
        let bytes =
            unsafe { core::slice::from_raw_parts(self.as_ptr() as *const u8, self.len() * 2) };
        AdviceWriter::write_bytes(&mut writer, bytes);
    }
    fn new_from_advice_tape() -> Self {
        // First read the length of the Vec<u16>
        let len = usize::new_from_advice_tape();
        // Create a vec of u16s with length len
        let mut buf = Vec::<u16>::with_capacity(len);
        // Cast the Vec<u16> to a byte slice of twice the length
        let bytes =
            unsafe { core::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut u8, 2 * len) };
        // Read the contents into the byte slice
        let mut reader = AdviceReader::get();
        AdviceReader::read_slice(&mut reader, bytes);
        // Adjust the length of the Vec<u16> after reading
        unsafe {
            buf.set_len(len);
        }
        // Return the filled Vec<u16>
        buf
    }
}

#[cfg(any(feature = "host", feature = "guest-std"))]
impl AdviceTapeIO for Vec<u32> {
    fn write_to_advice_tape(&self) {
        // Write the length of the Vec<u32> first
        self.len().write_to_advice_tape();
        // Then write the contents of the Vec<u32> to the advice tape
        let mut writer = AdviceWriter::get();
        let bytes =
            unsafe { core::slice::from_raw_parts(self.as_ptr() as *const u8, self.len() * 4) };
        AdviceWriter::write_bytes(&mut writer, bytes);
    }
    fn new_from_advice_tape() -> Self {
        // First read the length of the Vec<u32>
        let len = usize::new_from_advice_tape();
        // Create a vec of u32s with length len
        let mut buf = Vec::<u32>::with_capacity(len);
        // Cast the Vec<u32> to a byte slice of 4x the length
        let bytes =
            unsafe { core::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut u8, 4 * len) };
        // Read the contents into the byte slice
        let mut reader = AdviceReader::get();
        AdviceReader::read_slice(&mut reader, bytes);
        // Adjust the length of the Vec<u32> after reading
        unsafe {
            buf.set_len(len);
        }
        // Return the filled Vec<u32>
        buf
    }
}

#[cfg(any(feature = "host", feature = "guest-std"))]
impl AdviceTapeIO for Vec<u64> {
    fn write_to_advice_tape(&self) {
        // Write the length of the Vec<u64> first
        self.len().write_to_advice_tape();
        // Then write the contents of the Vec<u64> to the advice tape
        let mut writer = AdviceWriter::get();
        let bytes =
            unsafe { core::slice::from_raw_parts(self.as_ptr() as *const u8, self.len() * 8) };
        AdviceWriter::write_bytes(&mut writer, bytes);
    }
    fn new_from_advice_tape() -> Self {
        // First read the length of the Vec<u64>
        let len = usize::new_from_advice_tape();
        // Create a vec of u64s with length len
        let mut buf = Vec::<u64>::with_capacity(len);
        // Cast the Vec<u64> to a byte slice of 8x the length
        let bytes =
            unsafe { core::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut u8, 8 * len) };
        // Read the contents into the byte slice
        let mut reader = AdviceReader::get();
        AdviceReader::read_slice(&mut reader, bytes);
        // Adjust the length of the Vec<u64> after reading
        unsafe {
            buf.set_len(len);
        }
        // Return the filled Vec<u64>
        buf
    }
}

#[cfg(any(feature = "host", feature = "guest-std"))]
impl AdviceTapeIO for Vec<usize> {
    fn write_to_advice_tape(&self) {
        // Write the length of the Vec<usize> first
        self.len().write_to_advice_tape();
        // Then write the contents of the Vec<usize> to the advice tape
        for &item in self.iter() {
            item.write_to_advice_tape();
        }
    }
    fn new_from_advice_tape() -> Self {
        // First read the length of the Vec<usize>
        let len = usize::new_from_advice_tape();
        // Create a vec of usizes with length len
        let mut buf = Vec::<usize>::with_capacity(len);
        // Cast the Vec<usize> to a byte slice of either 4x or 8x the length
        #[cfg(target_pointer_width = "32")]
        let bytes =
            unsafe { core::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut u8, 4 * len) };
        #[cfg(target_pointer_width = "64")]
        let bytes =
            unsafe { core::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut u8, 8 * len) };
        // Read the contents into the byte slice
        let mut reader = AdviceReader::get();
        AdviceReader::read_slice(&mut reader, bytes);
        // Adjust the length of the Vec<usize> after reading
        unsafe {
            buf.set_len(len);
        }
        // Return the filled Vec<usize>
        buf
    }
}
