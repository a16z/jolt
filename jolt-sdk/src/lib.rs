#![cfg_attr(not(any(feature = "host", feature = "guest-std")), no_std)]

extern crate jolt_sdk_macros;

// Instruction encoding constants for RISC-V custom instructions
// Note: These are used in inline assembly via `const` keyword, but the compiler
// doesn't recognize that usage, so we suppress the dead_code warning.
#[doc(hidden)]
pub const CUSTOM_OPCODE: u32 = 0x5B; // Custom instructions opcode
#[doc(hidden)]
pub const FUNCT3_VIRTUAL_R: u32 = 0b000; // Virtual R-type instructions funct3
#[doc(hidden)]
pub const FUNCT3_VIRTUAL_ASSERT_EQ: u32 = 0b001; // VirtualAssertEQ funct3
#[doc(hidden)]
pub const FUNCT7_ADVICE_LB: u32 = 0x00; // Load byte from advice tape
#[doc(hidden)]
pub const FUNCT7_ADVICE_LH: u32 = 0x01; // Load halfword from advice tape
#[doc(hidden)]
pub const FUNCT7_ADVICE_LW: u32 = 0x02; // Load word from advice tape
#[doc(hidden)]
pub const FUNCT7_ADVICE_LD: u32 = 0x03; // Load doubleword from advice tape
#[doc(hidden)]
pub const FUNCT7_ADVICE_LEN: u32 = 0x04; // Get number of remaining bytes in advice tape

#[doc(hidden)]
pub const FIELD_INLINE_OPCODE: u32 = 0x7b;
#[doc(hidden)]
pub const FIELD_INLINE_R_TYPE_FUNCT7: u32 = 0;
#[doc(hidden)]
pub const FIELD_INLINE_ADD_FUNCT3: u32 = 0;
#[doc(hidden)]
pub const FIELD_INLINE_SUB_FUNCT3: u32 = 1;
#[doc(hidden)]
pub const FIELD_INLINE_MUL_FUNCT3: u32 = 2;
#[doc(hidden)]
pub const FIELD_INLINE_INV_FUNCT3: u32 = 3;
#[doc(hidden)]
pub const FIELD_INLINE_ASSERT_EQ_FUNCT3: u32 = 4;
#[doc(hidden)]
pub const FIELD_INLINE_LOAD_ACCUMULATE_FROM_REGISTER_FUNCT3: u32 = 5;
#[doc(hidden)]
pub const FIELD_INLINE_ASSERT_ZERO_FUNCT3: u32 = 6;
#[doc(hidden)]
pub const FIELD_INLINE_ASSERT_ZERO_FUNCT7: u32 = 2;
#[doc(hidden)]
pub const FIELD_INLINE_LOAD_IMM_FUNCT3: u32 = 7;
#[doc(hidden)]
pub const FIELD_INLINE_ADVICE_LIMB_FUNCT3: u32 = 6;
#[doc(hidden)]
pub const FIELD_INLINE_ADVICE_LIMB_FUNCT7: u32 = 1;
/// Memory-sourced accumulation shares the register form's funct3; funct7
/// carries this family tag with the word offset in bits 4..0.
#[doc(hidden)]
pub const FIELD_INLINE_LOAD_ACCUMULATE_FROM_MEMORY_FUNCT7: u32 = 0x60;

/// Number of field registers the field-inline extension addresses.
pub const FIELD_REGISTER_COUNT: u32 = 16;
/// The x-register field-inline ingress and limb readout move values through
/// (`a0`), pinned by the asm operand constraints of the blocks that encode it.
#[doc(hidden)]
pub const FIELD_INLINE_BRIDGE_X_REGISTER: u32 = 10;

/// A field-register operand; out-of-range literals fail at compile time
/// instead of wrapping into the encoding of a different register.
#[doc(hidden)]
pub const fn field_register(index: u32) -> u32 {
    assert!(
        index < FIELD_REGISTER_COUNT,
        "field-inline field register index must be below 16"
    );
    index
}

/// A 12-bit LoadImm immediate; wider literals fail at compile time.
#[doc(hidden)]
pub const fn field_inline_imm12(imm: u32) -> u32 {
    assert!(imm < 1 << 12, "field-inline immediates are 12 bits");
    imm
}

#[doc(hidden)]
pub const fn field_inline_r_word(funct7: u32, funct3: u32, rd: u32, rs1: u32, rs2: u32) -> u32 {
    assert!(
        funct7 < 1 << 7 && funct3 < 1 << 3 && rd < 32 && rs1 < 32 && rs2 < 32,
        "field-inline instruction word field out of range"
    );
    FIELD_INLINE_OPCODE | (rd << 7) | (funct3 << 12) | (rs1 << 15) | (rs2 << 20) | (funct7 << 25)
}

#[doc(hidden)]
pub const fn field_inline_i_word(funct3: u32, rd: u32, imm: u32) -> u32 {
    assert!(
        funct3 < 1 << 3 && rd < 32 && imm < 1 << 12,
        "field-inline instruction word field out of range"
    );
    FIELD_INLINE_OPCODE | (rd << 7) | (funct3 << 12) | (imm << 20)
}

#[doc(hidden)]
#[macro_export]
macro_rules! __field_inline_word {
    ($word:expr) => {{
        #[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
        {
            const WORD: u32 = $word;
            // SAFETY: this emits a fixed custom Jolt instruction word for the
            // tracer; operands are encoded constants, and no Rust memory is touched.
            unsafe {
                core::arch::asm!(".word {word}", word = const WORD, options(nostack));
            }
        }
        #[cfg(not(any(target_arch = "riscv32", target_arch = "riscv64")))]
        {
            let _ = $word;
        }
    }};
}

#[macro_export]
macro_rules! field_load_imm {
    ($rd:literal, $imm:literal) => {
        $crate::__field_inline_word!($crate::field_inline_i_word(
            $crate::FIELD_INLINE_LOAD_IMM_FUNCT3,
            $crate::field_register($rd),
            $crate::field_inline_imm12($imm)
        ))
    };
}

#[macro_export]
macro_rules! field_add {
    ($rd:literal, $rs1:literal, $rs2:literal) => {
        $crate::__field_inline_word!($crate::field_inline_r_word(
            $crate::FIELD_INLINE_R_TYPE_FUNCT7,
            $crate::FIELD_INLINE_ADD_FUNCT3,
            $crate::field_register($rd),
            $crate::field_register($rs1),
            $crate::field_register($rs2)
        ))
    };
}

#[macro_export]
macro_rules! field_sub {
    ($rd:literal, $rs1:literal, $rs2:literal) => {
        $crate::__field_inline_word!($crate::field_inline_r_word(
            $crate::FIELD_INLINE_R_TYPE_FUNCT7,
            $crate::FIELD_INLINE_SUB_FUNCT3,
            $crate::field_register($rd),
            $crate::field_register($rs1),
            $crate::field_register($rs2)
        ))
    };
}

#[macro_export]
macro_rules! field_mul {
    ($rd:literal, $rs1:literal, $rs2:literal) => {
        $crate::__field_inline_word!($crate::field_inline_r_word(
            $crate::FIELD_INLINE_R_TYPE_FUNCT7,
            $crate::FIELD_INLINE_MUL_FUNCT3,
            $crate::field_register($rd),
            $crate::field_register($rs1),
            $crate::field_register($rs2)
        ))
    };
}

#[macro_export]
macro_rules! field_inv {
    ($rd:literal, $rs1:literal) => {
        $crate::__field_inline_word!($crate::field_inline_r_word(
            $crate::FIELD_INLINE_R_TYPE_FUNCT7,
            $crate::FIELD_INLINE_INV_FUNCT3,
            $crate::field_register($rd),
            $crate::field_register($rs1),
            0
        ))
    };
}

#[macro_export]
macro_rules! field_assert_eq {
    ($rs1:literal, $rs2:literal) => {
        $crate::__field_inline_word!($crate::field_inline_r_word(
            $crate::FIELD_INLINE_R_TYPE_FUNCT7,
            $crate::FIELD_INLINE_ASSERT_EQ_FUNCT3,
            0,
            $crate::field_register($rs1),
            $crate::field_register($rs2)
        ))
    };
}

/// Asserts that field register `$rs1` is zero without changing it.
#[macro_export]
macro_rules! field_assert_zero {
    ($rs1:literal) => {
        $crate::__field_inline_word!($crate::field_inline_r_word(
            $crate::FIELD_INLINE_ASSERT_ZERO_FUNCT7,
            $crate::FIELD_INLINE_ASSERT_ZERO_FUNCT3,
            0,
            $crate::field_register($rs1),
            0
        ))
    };
}

/// Updates field register `$rd` to `old_rd * 2^64 + value` modulo the proof field,
/// appending one `u64` limb through the LoadAccumulateFromRegister bridge. Initialize
/// `$rd` to zero with [`field_load_imm!`] before starting a new value; append
/// limbs from most significant to least significant. [`field_from_limbs!`] owns
/// that sequence for canonical values.
///
/// The bridge names an x-register in the instruction word, so the value must
/// live in that register when the word executes; the only placement the
/// compiler guarantees is an operand bound in the same asm block, which pins
/// it to `a0` here.
#[macro_export]
macro_rules! field_load_accumulate_from_register {
    ($rd:literal, $value:expr) => {
        $crate::field_inline::load_accumulate_from_register::<{ $rd }>($value)
    };
}

/// Reads field register `$register` out as the little-endian `u64` limbs of its
/// canonical integer representative (`limbs[0]` is least significant), leaving
/// every field register unchanged.
///
/// ```ignore
/// let limbs: [u64; 4] = jolt::field_to_limbs!(3); // limb count from the type
/// let [low, high] = jolt::field_to_limbs!(3, 2); // explicit limb count
/// ```
///
/// `$register` is a constant `u32` expression below 16, and the limb count is
/// at least one; both are checked at compile time.
///
/// # Guarantees
///
/// The readout is proven, not trusted. With `K` the limb width of the
/// [modulus](field_inline::MODULUS) (4 for BN254, 2 for fp128), the macro:
///
/// 1. splits the register in place with `K` FIELD_ADVICE_LIMB instructions,
///    each bounding its limb below 2^64, and asserts the remaining quotient
///    is zero (FIELD_ASSERT_ZERO), proving `value = Σ limb_i·2^(64·i) (mod p)`;
/// 2. proves the guest was built for the executing proof field (see
///    [`field_inline::MODULUS`]);
/// 3. checks `Σ limb_i·2^(64·i) < p` in guest code, which makes the limbs the
///    unique canonical representative; limbs failing it can only come from a
///    dishonest prover, so the guest spoils the proof
///    ([`spoil_proof`](crate::spoil_proof));
/// 4. restores the consumed register by accumulating the limbs back, most
///    significant first.
///
/// # Width
///
/// A limb count above `K` zero-extends. Below `K`, values that do not fit
/// panic the guest instead of truncating; the panic is a provable outcome,
/// because the canonical limbs are already fixed when the guest checks them.
///
/// # Cost and clobbers
///
/// `3K + 2` field-inline rows (14 for BN254, 8 for fp128) plus the limb
/// comparison, whatever the limb count. The asm blocks declare their use of
/// `a0` and `a1`; no field register beyond `$register` is touched, so all 16
/// remain available to the caller.
///
/// # Failures
///
/// A guest built for a different proof field fails the step-2 FIELD_ASSERT_ZERO
/// at trace time ("FIELD_ASSERT_ZERO of nonzero field register"), and no proof
/// exists. Only the riscv64 Jolt guest target has field registers; elsewhere
/// the macro panics rather than return placeholder limbs.
#[macro_export]
macro_rules! field_to_limbs {
    ($register:expr $(,)?) => {
        $crate::field_inline::to_limbs::<{ $register }, _>()
    };
    ($register:expr, $limbs:expr $(,)?) => {
        $crate::field_inline::to_limbs::<{ $register }, { $limbs }>()
    };
}

/// Replaces field register `$register` with the field element whose canonical
/// little-endian `u64` limbs are `$limbs` (`limbs[0]` is least significant).
///
/// ```ignore
/// jolt::field_from_limbs!(3, [low, high]);
/// ```
///
/// `$register` is a constant `u32` expression below 16, and `$limbs` is an
/// array of at least one `u64`, evaluated once.
///
/// # Canonical import
///
/// The limbs must encode an integer below the [modulus](field_inline::MODULUS),
/// so the conversion is lossless and inverts [`field_to_limbs!`]: limbs past the
/// modulus width must be zero, and an integer of `p` or more panics the guest.
/// For explicit reduction modulo `p`, reset the register with
/// [`field_load_imm!`] and accumulate with
/// [`field_load_accumulate_from_register!`].
///
/// The macro resets the register to zero, proves the guest was built for the
/// executing proof field (see [`field_inline::MODULUS`]), and accumulates the
/// limbs most significant first, so repeated imports replace the value.
///
/// # Cost and clobbers
///
/// `K + 2 + min(N, K)` field-inline rows for `N` limbs and modulus width `K`
/// (10 for four BN254 limbs). The asm blocks declare their use of `a0` and
/// `a1`; no other field register is touched. Only the riscv64 Jolt guest
/// target has field registers; elsewhere the macro panics.
#[macro_export]
macro_rules! field_from_limbs {
    ($register:expr, $limbs:expr $(,)?) => {
        $crate::field_inline::from_limbs::<{ $register }, _>($limbs)
    };
}

pub mod field_inline;

#[cfg(any(feature = "host", feature = "guest-verifier"))]
pub mod host_utils;

#[cfg(any(feature = "host", feature = "guest-verifier"))]
pub use jolt_verifier;

#[cfg(feature = "host")]
pub use jolt_prover;

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

impl<T: Clone> Clone for UntrustedAdvice<T> {
    fn clone(&self) -> Self {
        Self {
            value: self.value.clone(),
        }
    }
}

impl<T: Copy> Copy for UntrustedAdvice<T> {}

/// Alias for `UntrustedAdvice<T>` — marks a guest function parameter as private
/// (committed by the prover, cryptographically hidden from the verifier via BlindFold).
///
/// Using `PrivateInput<T>` in a guest function requires the `zk` feature on `jolt-sdk`
/// in the host crate. The `#[jolt::provable]` macro enforces this at compile time.
pub type PrivateInput<T> = UntrustedAdvice<T>;

#[doc(hidden)]
pub const _ZK_FEATURE_ENABLED: bool = cfg!(feature = "zk");

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
                core::arch::asm!(
                    ".insn b {opcode}, {funct3}, {rs1}, {rs2}, .",
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
                    ".insn b {opcode}, {funct3}, {rs1}, {rs2}, .",
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
                ".insn r {opcode}, {funct3}, {funct7}, {rd}, x0, x0",
                opcode = const CUSTOM_OPCODE,
                funct3 = const FUNCT3_VIRTUAL_R,
                funct7 = const FUNCT7_ADVICE_LB,
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
                ".insn r {opcode}, {funct3}, {funct7}, {rd}, x0, x0",
                opcode = const CUSTOM_OPCODE,
                funct3 = const FUNCT3_VIRTUAL_R,
                funct7 = const FUNCT7_ADVICE_LH,
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
                ".insn r {opcode}, {funct3}, {funct7}, {rd}, x0, x0",
                opcode = const CUSTOM_OPCODE,
                funct3 = const FUNCT3_VIRTUAL_R,
                funct7 = const FUNCT7_ADVICE_LW,
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
                ".insn r {opcode}, {funct3}, {funct7}, {rd}, x0, x0",
                opcode = const CUSTOM_OPCODE,
                funct3 = const FUNCT3_VIRTUAL_R,
                funct7 = const FUNCT7_ADVICE_LD,
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
                ".insn r {opcode}, {funct3}, {funct7}, {rd}, x0, x0",
                opcode = const CUSTOM_OPCODE,
                funct3 = const FUNCT3_VIRTUAL_R,
                funct7 = const FUNCT7_ADVICE_LEN,
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
/// This trait excludes `Vec<_>` explicitly to avoid conflicts with the `Vec<T>` implementation below
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

/// implement AdviceTapeIO for `Vec<T>` where `T: Pod`
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
