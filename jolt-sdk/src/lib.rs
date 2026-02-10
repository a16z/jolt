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
pub const FUNCT3_ADVICE_LB: u32 = 0b011; // Load byte from advice tape
#[doc(hidden)]
pub const FUNCT3_ADVICE_LH: u32 = 0b100; // Load halfword from advice tape
#[doc(hidden)]
pub const FUNCT3_ADVICE_LW: u32 = 0b101; // Load word from advice tape
#[doc(hidden)]
pub const FUNCT3_ADVICE_LD: u32 = 0b110; // Load doubleword from advice tape
#[doc(hidden)]
pub const FUNCT3_ADVICE_LEN: u32 = 0b111; // Get number of remaining bytes in advice tape

#[cfg(any(feature = "host", feature = "guest-verifier"))]
pub mod host_utils;

#[cfg(any(feature = "host", feature = "guest-verifier"))]
pub use host_utils::*;

pub use jolt_platform::*;
pub use jolt_sdk_macros::advice;
pub use jolt_sdk_macros::provable;
pub use postcard;

use bytemuck::Pod;
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
    ($cond:expr) => {
        $crate::check_advice!($cond, "Advice assertion failed")
    };
    ($cond:expr, $err:expr) => {{
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
            assert!($cond, $err);
        }
    }};
}

/// Macro to assert that two values are equal, enforced by the prover.
/// This is a specialization of check_advice! for equality checks.
/// Rather than evaluating a boolean condition and then calling VirtualAssertEQ,
/// this calls VirtualAssertEQ directly on the provided LHS and RHS
/// Requires that both values fit in registers (fails to compile otherwise)
/// This is similar to the distinction between assert! and assert_eq!
#[macro_export]
macro_rules! check_advice_eq {
    ($left:expr, $right:expr) => {
        $crate::check_advice_eq!($left, $right, "Advice equality assertion failed")
    };
    ($left:expr, $right:expr, $err:expr) => {{
        #[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
        {
            let left = $left;
            let right = $right;
            unsafe {
                core::arch::asm!(
                    ".insn b {opcode}, {funct3}, {rs1}, {rs2}, 0",
                    opcode = const $crate::CUSTOM_OPCODE,
                    funct3 = const $crate::FUNCT3_VIRTUAL_ASSERT_EQ,
                    rs1 = in(reg) left,
                    rs2 = in(reg) right,
                    options(nostack)
                );
            }
        }
        #[cfg(not(any(target_arch = "riscv32", target_arch = "riscv64")))]
        {
            assert_eq!($left, $right, $err);
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
                ".insn i 0x5B, 2, x0, x0, 0", // VirtualHostIO (opcode=0x5B, funct3=2)
                in("a0") JOLT_ADVICE_WRITE_CALL_ID,
                in("a1") src_ptr,
                in("a2") len,
                options(nostack, preserves_flags)
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
        panic!("Advice tape I/O is not supported on non-RISC-V targets");
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
        panic!("Advice tape I/O is not supported on non-RISC-V targets");
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
        panic!("Advice tape I/O is not supported on non-RISC-V targets");
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
                ".insn i {opcode}, {funct3}, {rd}, x0, 0",
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
        panic!("Advice tape I/O is not supported on non-RISC-V targets");
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
        panic!("Advice tape I/O is not supported on non-RISC-V targets");
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
            // get misalignment of ptr to 8-byte boundary
            let mut to_align = core::cmp::min((8 - (ptr as usize & 7)) & 7, remaining);
            // Perform largest aligned writes possible until aligned to 8-byte boundary
            while to_align > 0 {
                let addr = ptr as usize;
                if to_align >= 4 && addr & 3 == 0 {
                    core::ptr::write(ptr as *mut u32, self.read_u32());
                    ptr = ptr.add(4);
                    remaining -= 4;
                    to_align -= 4;
                } else if to_align >= 2 && addr & 1 == 0 {
                    core::ptr::write(ptr as *mut u16, self.read_u16());
                    ptr = ptr.add(2);
                    remaining -= 2;
                    to_align -= 2;
                } else {
                    core::ptr::write(ptr, self.read_u8());
                    ptr = ptr.add(1);
                    remaining -= 1;
                    to_align -= 1;
                }
            }
            // Read and write in aligned 8-byte chunks
            while remaining >= 8 {
                core::ptr::write(ptr as *mut u64, self.read_u64());
                ptr = ptr.add(8);
                remaining -= 8;
            }
            // Handle any remaining bytes greedily with aligned reads/writes
            if remaining >= 4 {
                core::ptr::write(ptr as *mut u32, self.read_u32());
                ptr = ptr.add(4);
                remaining -= 4;
            }
            if remaining >= 2 {
                core::ptr::write(ptr as *mut u16, self.read_u16());
                ptr = ptr.add(2);
                remaining -= 2;
            }
            if remaining == 1 {
                core::ptr::write(ptr, self.read_u8());
            }
        }
    }
    #[cfg(not(any(target_arch = "riscv32", target_arch = "riscv64")))]
    fn read_slice(&mut self, _buf: &mut [u8]) {
        panic!("Advice tape I/O is not supported on non-RISC-V targets");
    }
}

/// Trait for writing to and reading from the advice tape
pub trait AdviceTapeIO: Sized {
    fn write_to_advice_tape(&self) {
        panic!("AdviceTapeIO not implemented for this type/target");
    }
    fn new_from_advice_tape() -> Self {
        panic!("AdviceTapeIO not implemented for this type/target");
    }
}

/// Empty marker trait for types that are Pod (Plain Old Data)
/// This trait excludes Vec<_> explicitly to avoid conflicts with the Vec<T> implementation below
pub trait JoltPod: Pod {}

macro_rules! impl_joltpod {
    ($($t:ty),*) => {
        $(
            impl JoltPod for $t {}
        )*
    };
}

impl_joltpod!(u8, u16, u32, u64, usize, i8, i16, i32, i64);

/// implement AdviceTapeIO for all Pod types using bytemuck
impl<T: JoltPod> AdviceTapeIO for T {
    fn write_to_advice_tape(&self) {
        let bytes = bytemuck::bytes_of(self);
        let mut writer = AdviceWriter::get();
        AdviceWriter::write_bytes(&mut writer, bytes);
    }
    fn new_from_advice_tape() -> Self {
        let mut value = core::mem::MaybeUninit::<T>::uninit();
        let bytes = unsafe {
            core::slice::from_raw_parts_mut(
                value.as_mut_ptr() as *mut u8,
                core::mem::size_of::<T>(),
            )
        };
        let mut reader = AdviceReader::get();
        AdviceReader::read_slice(&mut reader, bytes);
        unsafe { value.assume_init() }
    }
}

/// implement AdviceTapeIO for tuples via a macro
macro_rules! impl_tuple_adviceio {
    ( $( $name:ident ),+ ) => {
        #[allow(non_snake_case)]
        impl<$( $name ),+> AdviceTapeIO for ( $( $name ),+ )
        where
            $( $name: AdviceTapeIO ),+
        {
            fn write_to_advice_tape(&self) {
                let ( $( $name ),+ ) = self;
                $( $name.write_to_advice_tape(); )+
            }

            fn new_from_advice_tape() -> Self {
                (
                    $( <$name as AdviceTapeIO>::new_from_advice_tape(), )+
                )
            }
        }
    };
}

// implement AdviceTapeIO for tuples up to size 7
impl_tuple_adviceio!(A, B);
impl_tuple_adviceio!(A, B, C);
impl_tuple_adviceio!(A, B, C, D);
impl_tuple_adviceio!(A, B, C, D, E);
impl_tuple_adviceio!(A, B, C, D, E, F);
impl_tuple_adviceio!(A, B, C, D, E, F, G);

/// implement AdviceTapeIO for arrays of Pod types
impl<T: Pod, const N: usize> AdviceTapeIO for [T; N] {
    fn write_to_advice_tape(&self) {
        let bytes = bytemuck::cast_slice(self);
        let mut writer = AdviceWriter::get();
        AdviceWriter::write_bytes(&mut writer, bytes);
    }
    fn new_from_advice_tape() -> Self {
        let mut value = core::mem::MaybeUninit::<[T; N]>::uninit();
        let bytes = unsafe {
            core::slice::from_raw_parts_mut(
                value.as_mut_ptr() as *mut u8,
                N * core::mem::size_of::<T>(),
            )
        };
        let mut reader = AdviceReader::get();
        AdviceReader::read_slice(&mut reader, bytes);
        unsafe { value.assume_init() }
    }
}

/// implement AdviceTapeIO for Vec<T> where T: Pod
#[cfg(any(feature = "host", feature = "guest-std"))]
impl<T: Pod> AdviceTapeIO for Vec<T> {
    fn write_to_advice_tape(&self) {
        // Write the length and capacity of the Vec<T> first
        self.len().write_to_advice_tape();
        self.capacity().write_to_advice_tape();
        // Then write the contents of the Vec<T> to the advice tape as bytes
        let bytes = bytemuck::cast_slice(self.as_slice());
        let mut writer = AdviceWriter::get();
        AdviceWriter::write_bytes(&mut writer, bytes);
    }
    fn new_from_advice_tape() -> Self {
        // First read the length and capacity of the Vec<T>
        let len = usize::new_from_advice_tape();
        let capacity = usize::new_from_advice_tape();
        // panic and spoil the proof if capacity < len
        check_advice!(capacity >= len);
        // Create a vec of T with length len
        let mut buf = Vec::<T>::with_capacity(capacity);
        // Cast the Vec<T> to a byte slice of len * size_of::<T>()
        let bytes = unsafe {
            core::slice::from_raw_parts_mut(
                buf.as_mut_ptr() as *mut u8,
                len * core::mem::size_of::<T>(),
            )
        };
        // Read the contents into the byte slice
        let mut reader = AdviceReader::get();
        AdviceReader::read_slice(&mut reader, bytes);
        // Adjust the length of the Vec<T> after reading
        unsafe {
            buf.set_len(len);
        }
        // Return the filled Vec<T>
        buf
    }
}

#[cfg(target_arch = "riscv64")]
pub mod runtime;
