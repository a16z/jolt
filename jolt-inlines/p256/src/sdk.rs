//! P-256 (NIST P-256 / secp256r1) operations optimized for Jolt zkVM.
//!
//! Provides `P256Fq` (base field) and `P256Fr` (scalar field) types that wrap
//! `[u64; 4]` limbs in standard (non-Montgomery) form.  Multiplication, squaring,
//! and division are dispatched to custom RISC-V inline instructions on guest builds,
//! and to `ark_secp256r1` arithmetic on host builds.  Addition, subtraction,
//! negation, doubling, and tripling are implemented as pure integer arithmetic.

#[cfg(feature = "host")]
use ark_ff::{BigInt, Field, PrimeField};
#[cfg(feature = "host")]
use ark_secp256r1::{Fq as ArkFq, Fr as ArkFr};

use serde::{Deserialize, Serialize};

use crate::{P256_MODULUS, P256_ORDER};

/// Returns `true` iff `x >= p` (base field modulus), i.e., `x` is non-canonical.
///
/// P-256 modulus limbs (little-endian):
///   [0] = 0xFFFFFFFFFFFFFFFF
///   [1] = 0x00000000FFFFFFFF
///   [2] = 0x0000000000000000
///   [3] = 0xFFFFFFFF00000001
///
/// Because the limbs have mixed values, we need a full top-down comparison.
#[inline(always)]
fn is_fq_non_canonical(x: &[u64; 4]) -> bool {
    if x[3] < P256_MODULUS[3] {
        return false;
    } else if x[3] > P256_MODULUS[3] {
        return true;
    }
    // x[3] == P256_MODULUS[3]
    if x[2] < P256_MODULUS[2] {
        return false;
    } else if x[2] > P256_MODULUS[2] {
        return true;
    }
    // x[2] == P256_MODULUS[2] == 0
    if x[1] < P256_MODULUS[1] {
        return false;
    } else if x[1] > P256_MODULUS[1] {
        return true;
    }
    // x[1] == P256_MODULUS[1]
    x[0] >= P256_MODULUS[0]
}

/// Returns `true` iff `x >= n` (scalar field order), i.e., `x` is non-canonical.
///
/// P-256 order limbs (little-endian):
///   [0] = 0xF3B9CAC2FC632551
///   [1] = 0xBCE6FAADA7179E84
///   [2] = 0xFFFFFFFFFFFFFFFF
///   [3] = 0xFFFFFFFF00000000
///
/// Full top-down comparison since limbs have mixed values.
#[inline(always)]
fn is_fr_non_canonical(x: &[u64; 4]) -> bool {
    if x[3] < P256_ORDER[3] {
        return false;
    } else if x[3] > P256_ORDER[3] {
        return true;
    }
    // x[3] == P256_ORDER[3]
    if x[2] < P256_ORDER[2] {
        return false;
    } else if x[2] > P256_ORDER[2] {
        return true;
    }
    // x[2] == P256_ORDER[2] == 0xFFFFFFFFFFFFFFFF
    if x[1] < P256_ORDER[1] {
        return false;
    } else if x[1] > P256_ORDER[1] {
        return true;
    }
    // x[1] == P256_ORDER[1]
    x[0] >= P256_ORDER[0]
}

/// Add with carry: a + b + carry_in -> (sum, carry_out)
#[inline(always)]
const fn adc(a: u64, b: u64, carry: u64) -> (u64, u64) {
    let wide = a as u128 + b as u128 + carry as u128;
    (wide as u64, (wide >> 64) as u64)
}

/// Subtract with borrow: a - b - borrow_in -> (diff, borrow_out)
#[inline(always)]
const fn sbb(a: u64, b: u64, borrow: u64) -> (u64, u64) {
    let wide = (a as u128)
        .wrapping_sub(b as u128)
        .wrapping_sub(borrow as u128);
    (wide as u64, ((wide >> 64) & 1) as u64)
}

/// r = a + b mod modulus.  Both a and b must be < modulus.
#[inline(always)]
fn add_mod(a: &[u64; 4], b: &[u64; 4], modulus: &[u64; 4]) -> [u64; 4] {
    let (r0, c) = adc(a[0], b[0], 0);
    let (r1, c) = adc(a[1], b[1], c);
    let (r2, c) = adc(a[2], b[2], c);
    let (r3, c) = adc(a[3], b[3], c);

    // Try subtracting modulus; if underflow we keep the original sum
    let (s0, bw) = sbb(r0, modulus[0], 0);
    let (s1, bw) = sbb(r1, modulus[1], bw);
    let (s2, bw) = sbb(r2, modulus[2], bw);
    let (s3, bw) = sbb(r3, modulus[3], bw);

    // If there was a carry from the addition (c != 0) then sum >= 2^256 > modulus,
    // so the subtraction is valid.  If c == 0 but no borrow from subtraction (bw == 0),
    // the subtraction is also valid.  Otherwise keep the un-subtracted value.
    let use_sub = c != 0 || bw == 0;
    if use_sub {
        [s0, s1, s2, s3]
    } else {
        [r0, r1, r2, r3]
    }
}

/// r = a - b mod modulus.  Both a and b must be < modulus.
#[inline(always)]
fn sub_mod(a: &[u64; 4], b: &[u64; 4], modulus: &[u64; 4]) -> [u64; 4] {
    let (r0, bw) = sbb(a[0], b[0], 0);
    let (r1, bw) = sbb(a[1], b[1], bw);
    let (r2, bw) = sbb(a[2], b[2], bw);
    let (r3, bw) = sbb(a[3], b[3], bw);

    // If there was a borrow, add modulus back
    if bw != 0 {
        let (s0, c) = adc(r0, modulus[0], 0);
        let (s1, c) = adc(r1, modulus[1], c);
        let (s2, c) = adc(r2, modulus[2], c);
        let (s3, _) = adc(r3, modulus[3], c);
        [s0, s1, s2, s3]
    } else {
        [r0, r1, r2, r3]
    }
}

/// r = -a mod modulus.  a must be < modulus.
#[inline(always)]
fn neg_mod(a: &[u64; 4], modulus: &[u64; 4]) -> [u64; 4] {
    if a[0] == 0 && a[1] == 0 && a[2] == 0 && a[3] == 0 {
        [0u64; 4]
    } else {
        sub_mod(modulus, a, modulus)
    }
}

#[inline(always)]
fn limbs_to_bytes(limbs: &[u64; 4]) -> [u8; 32] {
    let mut out = [0u8; 32];
    let mut i = 0;
    while i < 4 {
        let bytes = limbs[i].to_le_bytes();
        let base = i * 8;
        let mut j = 0;
        while j < 8 {
            out[base + j] = bytes[j];
            j += 1;
        }
        i += 1;
    }
    out
}

#[inline(always)]
fn bytes_to_limbs(bytes: &[u8; 32]) -> [u64; 4] {
    let mut limbs = [0u64; 4];
    let mut i = 0;
    while i < 4 {
        let base = i * 8;
        let mut buf = [0u8; 8];
        let mut j = 0;
        while j < 8 {
            buf[j] = bytes[base + j];
            j += 1;
        }
        limbs[i] = u64::from_le_bytes(buf);
        i += 1;
    }
    limbs
}

pub use jolt_inlines_sdk::{spoil_proof, UnwrapOrSpoilProof};

/// Error types for P-256 operations.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum P256Error {
    InvalidFqElement,        // input array does not correspond to a valid Fq element
    InvalidFrElement,        // input array does not correspond to a valid Fr element
    NotOnCurve,              // point is not on the P-256 curve
    QAtInfinity,             // public key is point at infinity
    ROrSZero,                // one of the signature components is zero
    RxMismatch,              // computed R.x does not match r
    InvalidGlvSignWord(u64), // GLV sign word is not 0 or 1
}

/// Decode a GLV sign word: must be exactly 0 or 1.
#[cfg(all(
    not(feature = "host"),
    any(target_arch = "riscv32", target_arch = "riscv64")
))]
#[inline(always)]
fn decode_glv_sign_word(w: u64) -> Result<bool, P256Error> {
    match w {
        0 => Ok(false),
        1 => Ok(true),
        _ => Err(P256Error::InvalidGlvSignWord(w)),
    }
}

/// P-256 base field element, `[u64; 4]` in standard (non-Montgomery) form.
#[derive(Clone, PartialEq, Debug)]
pub struct P256Fq {
    e: [u64; 4],
}

impl P256Fq {
    /// Creates a new P256Fq element from a `[u64; 4]` array.
    /// Returns `Err(P256Error::InvalidFqElement)` if the value >= p.
    #[inline(always)]
    pub fn from_u64_arr(arr: &[u64; 4]) -> Result<Self, P256Error> {
        if is_fq_non_canonical(arr) {
            return Err(P256Error::InvalidFqElement);
        }
        Ok(P256Fq { e: *arr })
    }

    /// Creates a new P256Fq element from a `[u64; 4]` array (unchecked).
    /// The array is assumed to contain a value in the range `[0, p)`.
    #[inline(always)]
    pub fn from_u64_arr_unchecked(arr: &[u64; 4]) -> Self {
        P256Fq { e: *arr }
    }

    /// Returns the four u64 limbs (little-endian).
    #[inline(always)]
    pub fn e(&self) -> [u64; 4] {
        self.e
    }

    /// Returns the element as 32 little-endian bytes.
    #[inline(always)]
    pub fn to_bytes(&self) -> [u8; 32] {
        limbs_to_bytes(&self.e)
    }

    /// Creates a P256Fq from 32 little-endian bytes.
    /// Returns error if the value >= p.
    #[inline(always)]
    pub fn from_bytes(bytes: &[u8; 32]) -> Result<Self, P256Error> {
        let limbs = bytes_to_limbs(bytes);
        Self::from_u64_arr(&limbs)
    }

    /// Returns the additive identity element (0).
    #[inline(always)]
    pub fn zero() -> Self {
        P256Fq { e: [0u64; 4] }
    }

    /// Returns true if the element is zero.
    #[inline(always)]
    pub fn is_zero(&self) -> bool {
        self.e == [0u64; 4]
    }

    /// Returns `-self mod p`.
    #[inline(always)]
    pub fn neg(&self) -> Self {
        P256Fq {
            e: neg_mod(&self.e, &P256_MODULUS),
        }
    }

    /// Returns `self + other mod p`.
    #[inline(always)]
    pub fn add(&self, other: &P256Fq) -> Self {
        P256Fq {
            e: add_mod(&self.e, &other.e, &P256_MODULUS),
        }
    }

    /// Returns `self - other mod p`.
    #[inline(always)]
    pub fn sub(&self, other: &P256Fq) -> Self {
        P256Fq {
            e: sub_mod(&self.e, &other.e, &P256_MODULUS),
        }
    }

    /// Returns `2 * self mod p`.
    #[inline(always)]
    pub fn dbl(&self) -> Self {
        self.add(self)
    }

    /// Returns `3 * self mod p`.
    #[inline(always)]
    pub fn tpl(&self) -> Self {
        self.dbl().add(self)
    }

    // mul — RISC-V guest (inline instruction)

    /// Returns `self * other mod p`.
    /// Uses a custom RISC-V inline instruction for performance.
    #[cfg(all(
        not(feature = "host"),
        any(target_arch = "riscv32", target_arch = "riscv64")
    ))]
    #[inline(always)]
    pub fn mul(&self, other: &P256Fq) -> Self {
        let mut e = [0u64; 4];
        // SAFETY: e is a stack-local array with a valid pointer. The custom
        // instruction is intercepted by the Jolt tracer; nostack is valid.
        unsafe {
            use crate::{INLINE_OPCODE, P256_FUNCT7, P256_MULQ_FUNCT3};
            core::arch::asm!(
                ".insn r {opcode}, {funct3}, {funct7}, {rd}, {rs1}, {rs2}",
                opcode = const INLINE_OPCODE,
                funct3 = const P256_MULQ_FUNCT3,
                funct7 = const P256_FUNCT7,
                rd = in(reg) e.as_mut_ptr(),
                rs1 = in(reg) self.e.as_ptr(),
                rs2 = in(reg) other.e.as_ptr(),
                options(nostack)
            );
        }
        if is_fq_non_canonical(&e) {
            spoil_proof();
        }
        P256Fq::from_u64_arr_unchecked(&e[0..4].try_into().unwrap())
    }

    /// Panics on non-RISC-V guest (no inline available).
    #[cfg(all(
        not(feature = "host"),
        not(any(target_arch = "riscv32", target_arch = "riscv64"))
    ))]
    pub fn mul(&self, _other: &P256Fq) -> Self {
        panic!("P256Fq::mul called on non-RISC-V target without host feature");
    }

    #[cfg(feature = "host")]
    #[inline(always)]
    pub fn mul(&self, other: &P256Fq) -> Self {
        P256Fq {
            e: (ArkFq::new(BigInt(self.e)) * ArkFq::new(BigInt(other.e)))
                .into_bigint()
                .0,
        }
    }

    // square — RISC-V guest (inline instruction)

    /// Returns `self^2 mod p`.
    /// Uses a custom RISC-V inline instruction for performance.
    #[cfg(all(
        not(feature = "host"),
        any(target_arch = "riscv32", target_arch = "riscv64")
    ))]
    #[inline(always)]
    pub fn square(&self) -> Self {
        let mut e = [0u64; 4];
        // SAFETY: e is a stack-local array with a valid pointer. The custom
        // instruction is intercepted by the Jolt tracer; nostack is valid.
        unsafe {
            use crate::{INLINE_OPCODE, P256_FUNCT7, P256_SQUAREQ_FUNCT3};
            core::arch::asm!(
                ".insn r {opcode}, {funct3}, {funct7}, {rd}, {rs1}, x0",
                opcode = const INLINE_OPCODE,
                funct3 = const P256_SQUAREQ_FUNCT3,
                funct7 = const P256_FUNCT7,
                rd = in(reg) e.as_mut_ptr(),
                rs1 = in(reg) self.e.as_ptr(),
                options(nostack)
            );
        }
        if is_fq_non_canonical(&e) {
            spoil_proof();
        }
        P256Fq::from_u64_arr_unchecked(&e[0..4].try_into().unwrap())
    }

    /// Panics on non-RISC-V guest (no inline available).
    #[cfg(all(
        not(feature = "host"),
        not(any(target_arch = "riscv32", target_arch = "riscv64"))
    ))]
    pub fn square(&self) -> Self {
        panic!("P256Fq::square called on non-RISC-V target without host feature");
    }

    #[cfg(feature = "host")]
    #[inline(always)]
    pub fn square(&self) -> Self {
        P256Fq {
            e: ArkFq::new(BigInt(self.e)).square().into_bigint().0,
        }
    }

    // div / div_assume_nonzero — RISC-V guest (inline instruction)

    /// Returns `self / other mod p`.  Assumes `other != 0`.
    /// Uses a custom RISC-V inline instruction for performance.
    #[cfg(all(
        not(feature = "host"),
        any(target_arch = "riscv32", target_arch = "riscv64")
    ))]
    #[inline(always)]
    fn div_assume_nonzero(&self, other: &P256Fq) -> Self {
        let mut e = [0u64; 4];
        // SAFETY: e is a stack-local array with a valid pointer. The custom
        // instruction is intercepted by the Jolt tracer; nostack is valid.
        unsafe {
            use crate::{INLINE_OPCODE, P256_DIVQ_FUNCT3, P256_FUNCT7};
            core::arch::asm!(
                ".insn r {opcode}, {funct3}, {funct7}, {rd}, {rs1}, {rs2}",
                opcode = const INLINE_OPCODE,
                funct3 = const P256_DIVQ_FUNCT3,
                funct7 = const P256_FUNCT7,
                rd = in(reg) e.as_mut_ptr(),
                rs1 = in(reg) self.e.as_ptr(),
                rs2 = in(reg) other.e.as_ptr(),
                options(nostack)
            );
        }
        if is_fq_non_canonical(&e) {
            spoil_proof();
        }
        P256Fq::from_u64_arr_unchecked(&e[0..4].try_into().unwrap())
    }

    /// Returns `self / other mod p`.
    /// Spoils the proof if `other == 0`.
    #[cfg(all(
        not(feature = "host"),
        any(target_arch = "riscv32", target_arch = "riscv64")
    ))]
    #[inline(always)]
    pub fn div(&self, other: &P256Fq) -> Self {
        // spoil proof if other == 0
        if other.is_zero() {
            spoil_proof();
        }
        self.div_assume_nonzero(other)
    }

    /// Panics on non-RISC-V guest (no inline available).
    #[cfg(all(
        not(feature = "host"),
        not(any(target_arch = "riscv32", target_arch = "riscv64"))
    ))]
    pub fn div_assume_nonzero(&self, _other: &P256Fq) -> Self {
        panic!("P256Fq::div_assume_nonzero called on non-RISC-V target without host feature");
    }

    /// Panics on non-RISC-V guest (no inline available).
    #[cfg(all(
        not(feature = "host"),
        not(any(target_arch = "riscv32", target_arch = "riscv64"))
    ))]
    pub fn div(&self, _other: &P256Fq) -> Self {
        panic!("P256Fq::div called on non-RISC-V target without host feature");
    }

    #[cfg(feature = "host")]
    #[inline(always)]
    pub fn div_assume_nonzero(&self, other: &P256Fq) -> Self {
        P256Fq {
            e: (ArkFq::new(BigInt(self.e)) / ArkFq::new(BigInt(other.e)))
                .into_bigint()
                .0,
        }
    }

    /// Host implementation: checks `other != 0` then delegates.
    #[cfg(feature = "host")]
    #[inline(always)]
    pub fn div(&self, other: &P256Fq) -> Self {
        if other.is_zero() {
            panic!("division by zero in P256Fq::div");
        }
        self.div_assume_nonzero(other)
    }
}

/// P-256 scalar field element, `[u64; 4]` in standard (non-Montgomery) form.
#[derive(Clone, PartialEq, Debug)]
pub struct P256Fr {
    e: [u64; 4],
}

impl P256Fr {
    /// Creates a new P256Fr element from a `[u64; 4]` array.
    /// Returns `Err(P256Error::InvalidFrElement)` if the value >= n.
    #[inline(always)]
    pub fn from_u64_arr(arr: &[u64; 4]) -> Result<Self, P256Error> {
        if is_fr_non_canonical(arr) {
            return Err(P256Error::InvalidFrElement);
        }
        Ok(P256Fr { e: *arr })
    }

    /// Creates a new P256Fr element from a `[u64; 4]` array (unchecked).
    /// The array is assumed to contain a value in the range `[0, n)`.
    #[inline(always)]
    pub fn from_u64_arr_unchecked(arr: &[u64; 4]) -> Self {
        P256Fr { e: *arr }
    }

    /// Returns the four u64 limbs (little-endian).
    #[inline(always)]
    pub fn e(&self) -> [u64; 4] {
        self.e
    }

    /// Returns the element as 32 little-endian bytes.
    #[inline(always)]
    pub fn to_bytes(&self) -> [u8; 32] {
        limbs_to_bytes(&self.e)
    }

    /// Creates a P256Fr from 32 little-endian bytes.
    /// Returns error if the value >= n.
    #[inline(always)]
    pub fn from_bytes(bytes: &[u8; 32]) -> Result<Self, P256Error> {
        let limbs = bytes_to_limbs(bytes);
        Self::from_u64_arr(&limbs)
    }

    /// Returns the additive identity element (0).
    #[inline(always)]
    pub fn zero() -> Self {
        P256Fr { e: [0u64; 4] }
    }

    /// Returns true if the element is zero.
    #[inline(always)]
    pub fn is_zero(&self) -> bool {
        self.e == [0u64; 4]
    }

    /// Returns `-self mod n`.
    #[inline(always)]
    pub fn neg(&self) -> Self {
        P256Fr {
            e: neg_mod(&self.e, &P256_ORDER),
        }
    }

    /// Returns `self + other mod n`.
    #[inline(always)]
    pub fn add(&self, other: &P256Fr) -> Self {
        P256Fr {
            e: add_mod(&self.e, &other.e, &P256_ORDER),
        }
    }

    /// Returns `self - other mod n`.
    #[inline(always)]
    pub fn sub(&self, other: &P256Fr) -> Self {
        P256Fr {
            e: sub_mod(&self.e, &other.e, &P256_ORDER),
        }
    }

    /// Returns `2 * self mod n`.
    #[inline(always)]
    pub fn dbl(&self) -> Self {
        self.add(self)
    }

    /// Returns `3 * self mod n`.
    #[inline(always)]
    pub fn tpl(&self) -> Self {
        self.dbl().add(self)
    }

    // mul — RISC-V guest (inline instruction)

    /// Returns `self * other mod n`.
    /// Uses a custom RISC-V inline instruction for performance.
    #[cfg(all(
        not(feature = "host"),
        any(target_arch = "riscv32", target_arch = "riscv64")
    ))]
    #[inline(always)]
    pub fn mul(&self, other: &P256Fr) -> Self {
        let mut e = [0u64; 4];
        // SAFETY: e is a stack-local array with a valid pointer. The custom
        // instruction is intercepted by the Jolt tracer; nostack is valid.
        unsafe {
            use crate::{INLINE_OPCODE, P256_FUNCT7, P256_MULR_FUNCT3};
            core::arch::asm!(
                ".insn r {opcode}, {funct3}, {funct7}, {rd}, {rs1}, {rs2}",
                opcode = const INLINE_OPCODE,
                funct3 = const P256_MULR_FUNCT3,
                funct7 = const P256_FUNCT7,
                rd = in(reg) e.as_mut_ptr(),
                rs1 = in(reg) self.e.as_ptr(),
                rs2 = in(reg) other.e.as_ptr(),
                options(nostack)
            );
        }
        if is_fr_non_canonical(&e) {
            spoil_proof();
        }
        P256Fr::from_u64_arr_unchecked(&e[0..4].try_into().unwrap())
    }

    /// Panics on non-RISC-V guest (no inline available).
    #[cfg(all(
        not(feature = "host"),
        not(any(target_arch = "riscv32", target_arch = "riscv64"))
    ))]
    pub fn mul(&self, _other: &P256Fr) -> Self {
        panic!("P256Fr::mul called on non-RISC-V target without host feature");
    }

    #[cfg(feature = "host")]
    #[inline(always)]
    pub fn mul(&self, other: &P256Fr) -> Self {
        P256Fr {
            e: (ArkFr::new(BigInt(self.e)) * ArkFr::new(BigInt(other.e)))
                .into_bigint()
                .0,
        }
    }

    // square — RISC-V guest (inline instruction)

    /// Returns `self^2 mod n`.
    /// Uses a custom RISC-V inline instruction for performance.
    #[cfg(all(
        not(feature = "host"),
        any(target_arch = "riscv32", target_arch = "riscv64")
    ))]
    #[inline(always)]
    pub fn square(&self) -> Self {
        let mut e = [0u64; 4];
        // SAFETY: e is a stack-local array with a valid pointer. The custom
        // instruction is intercepted by the Jolt tracer; nostack is valid.
        unsafe {
            use crate::{INLINE_OPCODE, P256_FUNCT7, P256_SQUARER_FUNCT3};
            core::arch::asm!(
                ".insn r {opcode}, {funct3}, {funct7}, {rd}, {rs1}, x0",
                opcode = const INLINE_OPCODE,
                funct3 = const P256_SQUARER_FUNCT3,
                funct7 = const P256_FUNCT7,
                rd = in(reg) e.as_mut_ptr(),
                rs1 = in(reg) self.e.as_ptr(),
                options(nostack)
            );
        }
        if is_fr_non_canonical(&e) {
            spoil_proof();
        }
        P256Fr::from_u64_arr_unchecked(&e[0..4].try_into().unwrap())
    }

    /// Panics on non-RISC-V guest (no inline available).
    #[cfg(all(
        not(feature = "host"),
        not(any(target_arch = "riscv32", target_arch = "riscv64"))
    ))]
    pub fn square(&self) -> Self {
        panic!("P256Fr::square called on non-RISC-V target without host feature");
    }

    #[cfg(feature = "host")]
    #[inline(always)]
    pub fn square(&self) -> Self {
        P256Fr {
            e: ArkFr::new(BigInt(self.e)).square().into_bigint().0,
        }
    }

    // div / div_assume_nonzero — RISC-V guest (inline instruction)

    /// Returns `self / other mod n`.  Assumes `other != 0`.
    /// Uses a custom RISC-V inline instruction for performance.
    #[cfg(all(
        not(feature = "host"),
        any(target_arch = "riscv32", target_arch = "riscv64")
    ))]
    #[inline(always)]
    fn div_assume_nonzero(&self, other: &P256Fr) -> Self {
        let mut e = [0u64; 4];
        // SAFETY: e is a stack-local array with a valid pointer. The custom
        // instruction is intercepted by the Jolt tracer; nostack is valid.
        unsafe {
            use crate::{INLINE_OPCODE, P256_DIVR_FUNCT3, P256_FUNCT7};
            core::arch::asm!(
                ".insn r {opcode}, {funct3}, {funct7}, {rd}, {rs1}, {rs2}",
                opcode = const INLINE_OPCODE,
                funct3 = const P256_DIVR_FUNCT3,
                funct7 = const P256_FUNCT7,
                rd = in(reg) e.as_mut_ptr(),
                rs1 = in(reg) self.e.as_ptr(),
                rs2 = in(reg) other.e.as_ptr(),
                options(nostack)
            );
        }
        if is_fr_non_canonical(&e) {
            spoil_proof();
        }
        P256Fr::from_u64_arr_unchecked(&e[0..4].try_into().unwrap())
    }

    /// Returns `self / other mod n`.
    /// Spoils the proof if `other == 0`.
    #[cfg(all(
        not(feature = "host"),
        any(target_arch = "riscv32", target_arch = "riscv64")
    ))]
    #[inline(always)]
    pub fn div(&self, other: &P256Fr) -> Self {
        // spoil proof if other == 0
        if other.is_zero() {
            spoil_proof();
        }
        self.div_assume_nonzero(other)
    }

    /// Panics on non-RISC-V guest (no inline available).
    #[cfg(all(
        not(feature = "host"),
        not(any(target_arch = "riscv32", target_arch = "riscv64"))
    ))]
    pub fn div_assume_nonzero(&self, _other: &P256Fr) -> Self {
        panic!("P256Fr::div_assume_nonzero called on non-RISC-V target without host feature");
    }

    /// Panics on non-RISC-V guest (no inline available).
    #[cfg(all(
        not(feature = "host"),
        not(any(target_arch = "riscv32", target_arch = "riscv64"))
    ))]
    pub fn div(&self, _other: &P256Fr) -> Self {
        panic!("P256Fr::div called on non-RISC-V target without host feature");
    }

    #[cfg(feature = "host")]
    #[inline(always)]
    pub fn div_assume_nonzero(&self, other: &P256Fr) -> Self {
        P256Fr {
            e: (ArkFr::new(BigInt(self.e)) / ArkFr::new(BigInt(other.e)))
                .into_bigint()
                .0,
        }
    }

    /// Host implementation: checks `other != 0` then delegates.
    #[cfg(feature = "host")]
    #[inline(always)]
    pub fn div(&self, other: &P256Fr) -> Self {
        if other.is_zero() {
            panic!("division by zero in P256Fr::div");
        }
        self.div_assume_nonzero(other)
    }
}

use jolt_inlines_sdk::ec::{AffinePoint, CurveParams, ECField};

impl ECField for P256Fq {
    type Error = P256Error;
    #[inline(always)]
    fn zero() -> Self {
        Self::zero()
    }
    #[inline(always)]
    fn is_zero(&self) -> bool {
        self.is_zero()
    }
    #[inline(always)]
    fn neg(&self) -> Self {
        self.neg()
    }
    #[inline(always)]
    fn add(&self, other: &Self) -> Self {
        self.add(other)
    }
    #[inline(always)]
    fn sub(&self, other: &Self) -> Self {
        self.sub(other)
    }
    #[inline(always)]
    fn dbl(&self) -> Self {
        self.dbl()
    }
    #[inline(always)]
    fn tpl(&self) -> Self {
        self.tpl()
    }
    #[inline(always)]
    fn mul(&self, other: &Self) -> Self {
        self.mul(other)
    }
    #[inline(always)]
    fn square(&self) -> Self {
        self.square()
    }
    #[inline(always)]
    fn div(&self, other: &Self) -> Self {
        self.div(other)
    }
    #[inline(always)]
    fn div_assume_nonzero(&self, other: &Self) -> Self {
        self.div_assume_nonzero(other)
    }
    #[inline(always)]
    fn to_u64_arr(&self) -> [u64; 4] {
        self.e()
    }
    #[inline(always)]
    fn from_u64_arr(arr: &[u64; 4]) -> Result<Self, Self::Error> {
        Self::from_u64_arr(arr)
    }
    #[inline(always)]
    fn from_u64_arr_unchecked(arr: &[u64; 4]) -> Self {
        Self::from_u64_arr_unchecked(arr)
    }
}

/// P-256 curve: y² = x³ + ax + b where a = p-3
#[derive(Clone)]
pub struct P256Curve;

impl CurveParams<P256Fq> for P256Curve {
    type Error = P256Error;

    fn curve_a() -> Option<P256Fq> {
        // a = p - 3
        Some(P256Fq::from_u64_arr_unchecked(&[
            0xFFFF_FFFF_FFFF_FFFC,
            0x0000_0000_FFFF_FFFF,
            0x0000_0000_0000_0000,
            0xFFFF_FFFF_0000_0001,
        ]))
    }

    fn curve_b() -> P256Fq {
        P256Fq::from_u64_arr_unchecked(&crate::P256_CURVE_B)
    }

    // Fake GLV Shamir verification produces infinity — needs this check.
    const DOUBLE_AND_ADD_DIVISOR_CHECK: bool = true;

    fn not_on_curve_error() -> Self::Error {
        P256Error::NotOnCurve
    }
}

/// P-256 affine point. All point arithmetic (add, double, double_and_add, neg,
/// is_on_curve) is provided by `AffinePoint` in `jolt-inlines-sdk`.
pub type P256Point = AffinePoint<P256Fq, P256Curve>;

/// Extension trait for P-256-specific point operations.
pub trait P256PointExt {
    fn generator() -> P256Point;
}

impl P256PointExt for P256Point {
    #[inline(always)]
    fn generator() -> P256Point {
        P256Point::new_unchecked(
            P256Fq::from_u64_arr_unchecked(&crate::P256_GENERATOR_X),
            P256Fq::from_u64_arr_unchecked(&crate::P256_GENERATOR_Y),
        )
    }
}

// ECDSA P-256 verification using Fake GLV
//
// Reference: "Fake GLV: You don't need an efficient endomorphism to implement
// GLV-like scalar multiplication in SNARK circuits" (Latincrypt 2025)

/// Call the Fake GLV advice inline to get R = s*P and half-GCD decomposition.
/// Returns (R, a_lo, a_hi, a_sign, b_lo, b_hi, b_sign).
#[cfg(all(
    not(feature = "host"),
    any(target_arch = "riscv32", target_arch = "riscv64")
))]
fn fake_glv_scalar_mul(s: &P256Fr, p: &P256Point) -> (P256Point, u128, bool, u128, bool) {
    // Output buffer: 14 u64 values
    let mut out = [0u64; 14];
    let s_arr = s.e();
    let p_arr = p.to_u64_arr();
    // SAFETY: s_arr and p_arr are stack-local arrays with valid pointers.
    // The custom instruction is intercepted by the Jolt tracer and expanded
    // into a virtual instruction sequence; it never executes on real hardware.
    // nostack is correct because the instruction does not modify the stack.
    unsafe {
        core::arch::asm!(
            ".insn r {opcode}, {funct3}, {funct7}, {rd}, {rs1}, {rs2}",
            opcode = const crate::INLINE_OPCODE,
            funct3 = const crate::P256_FAKE_GLV_ADV_FUNCT3,
            funct7 = const crate::P256_FUNCT7,
            rd = in(reg) out.as_mut_ptr(),
            rs1 = in(reg) s_arr.as_ptr(),
            rs2 = in(reg) p_arr.as_ptr(),
            options(nostack)
        );
    }
    let r_point = P256Point::from_u64_arr_unchecked(&[
        out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7],
    ]);
    let a_val = (out[8] as u128) | ((out[9] as u128) << 64);
    let a_sign = decode_glv_sign_word(out[10]).unwrap_or_spoil_proof();
    let b_val = (out[11] as u128) | ((out[12] as u128) << 64);
    let b_sign = decode_glv_sign_word(out[13]).unwrap_or_spoil_proof();
    (r_point, a_val, a_sign, b_val, b_sign)
}

#[cfg(feature = "host")]
fn fake_glv_scalar_mul(s: &P256Fr, p: &P256Point) -> (P256Point, u128, bool, u128, bool) {
    use ark_ff::{BigInt, PrimeField};
    use num_bigint::BigInt as NBigInt;

    // Compute R = s * P using arkworks
    use ark_ec::CurveGroup;
    use ark_secp256r1::{Affine, Fq, Fr, Projective};
    let s_fr = Fr::new(BigInt(s.e()));
    let p_affine = Affine::new(Fq::new(BigInt(p.x().e())), Fq::new(BigInt(p.y().e())));
    let r_result = (Projective::from(p_affine) * s_fr).into_affine();
    let rx: [u64; 4] = r_result.x.into_bigint().0;
    let ry: [u64; 4] = r_result.y.into_bigint().0;
    let r_point = P256Point::from_u64_arr_unchecked(&[
        rx[0], rx[1], rx[2], rx[3], ry[0], ry[1], ry[2], ry[3],
    ]);

    // Half-GCD decomposition via shared module
    let s_big: NBigInt = Fr::new(BigInt(s.e())).into_bigint().into();
    let (a_val, a_sign, b_val, b_sign) = crate::fake_glv::decompose_to_u128s(&s_big);
    (r_point, a_val, a_sign, b_val, b_sign)
}

#[cfg(all(
    not(feature = "host"),
    not(any(target_arch = "riscv32", target_arch = "riscv64"))
))]
fn fake_glv_scalar_mul(_s: &P256Fr, _p: &P256Point) -> (P256Point, u128, bool, u128, bool) {
    panic!("fake_glv_scalar_mul not available on this target");
}

/// 4-scalar 128-bit Shamir's trick.
#[inline(always)]
fn shamir_4x128(scalars: [u128; 4], points: [P256Point; 4]) -> P256Point {
    // Build 16-entry lookup table from 4 base points
    let p01 = points[0].add(&points[1]);
    let p02 = points[0].add(&points[2]);
    let p03 = points[0].add(&points[3]);
    let p12 = points[1].add(&points[2]);
    let p13 = points[1].add(&points[3]);
    let p23 = points[2].add(&points[3]);
    let p012 = p01.add(&points[2]);
    let p013 = p01.add(&points[3]);
    let p023 = p02.add(&points[3]);
    let p123 = p12.add(&points[3]);
    let p0123 = p012.add(&points[3]);

    let table = [
        P256Point::infinity(),
        points[0].clone(),
        points[1].clone(),
        p01,
        points[2].clone(),
        p02,
        p12,
        p012,
        points[3].clone(),
        p03,
        p13,
        p013,
        p23,
        p023,
        p123,
        p0123,
    ];

    let mut res = P256Point::infinity();
    for bit in (0..128).rev() {
        let mut idx = 0usize;
        for (j, scalar) in scalars.iter().enumerate() {
            if (scalar >> bit) & 1 == 1 {
                idx |= 1 << j;
            }
        }
        if res.is_infinity() {
            if idx != 0 {
                res = table[idx].clone();
            }
        } else if idx != 0 {
            res = res.double_and_add(&table[idx]);
        } else {
            res = res.double();
        }
    }
    res
}

/// Verify an ECDSA P-256 signature using Fake GLV.
///
/// The prover computes R1 = u1*G and R2 = u2*Q off-circuit via the Fake GLV
/// advice inline, which also provides half-GCD decompositions (a1,b1) and (a2,b2)
/// with b_i * u_i ≡ a_i (mod n) and |a_i|, |b_i| ≤ √n ≈ 2^128.
///
/// The guest verifies:
///   1. b1*u1 ≡ a1 (mod n) and b2*u2 ≡ a2 (mod n) — scalar field checks
///   2. R1, R2 on curve — point validity
///   3. a1*G - b1*R1 + a2*Q - b2*R2 = O — 4-scalar 128-bit Shamir (the main savings)
///   4. (R1 + R2).x mod n == r — ECDSA final check
///
/// This halves the doublings: 128 instead of 256, achieving ~1.7x speedup.
#[inline(always)]
pub fn ecdsa_verify(z: P256Fr, r: P256Fr, s: P256Fr, q: P256Point) -> Result<(), P256Error> {
    if q.is_infinity() {
        return Err(P256Error::QAtInfinity);
    }
    if r.is_zero() || s.is_zero() {
        return Err(P256Error::ROrSZero);
    }

    // Step 1: Compute u1 = z/s, u2 = r/s
    let u1 = z.div_assume_nonzero(&s);
    let u2 = r.div_assume_nonzero(&s);

    // Step 2: Get R1 = u1*G and decomposition via Fake GLV advice
    let g = P256Point::generator();
    let (r1, a1_val, a1_sign, b1_val, b1_sign) = fake_glv_scalar_mul(&u1, &g);

    // Step 3: Get R2 = u2*Q and decomposition via Fake GLV advice
    let (r2, a2_val, a2_sign, b2_val, b2_sign) = fake_glv_scalar_mul(&u2, &q);

    // Step 4: Verify R1, R2 are on curve
    if !r1.is_on_curve() {
        spoil_proof();
    }
    if !r2.is_on_curve() {
        spoil_proof();
    }

    // Step 5: Verify decompositions: b_i * u_i ≡ a_i (mod n)
    // Construct a_i and b_i as P256Fr elements
    let make_fr = |val: u128, sign: bool| -> P256Fr {
        let lo = val as u64;
        let hi = (val >> 64) as u64;
        let fr = P256Fr::from_u64_arr_unchecked(&[lo, hi, 0, 0]);
        if sign {
            fr.neg()
        } else {
            fr
        }
    };
    let a1_fr = make_fr(a1_val, a1_sign);
    let b1_fr = make_fr(b1_val, b1_sign);
    let a2_fr = make_fr(a2_val, a2_sign);
    let b2_fr = make_fr(b2_val, b2_sign);

    // Check b1*u1 ≡ a1 (mod n)
    let check1 = b1_fr.mul(&u1);
    if check1.e() != a1_fr.e() {
        spoil_proof();
    }
    // Check b2*u2 ≡ a2 (mod n)
    let check2 = b2_fr.mul(&u2);
    if check2.e() != a2_fr.e() {
        spoil_proof();
    }

    // Step 6: Prepare points for 4-scalar 128-bit Shamir
    // We verify: a1*G + a2*Q - b1*R1 - b2*R2 = O
    // Which is: a1*G + a2*Q + (-b1)*R1 + (-b2)*R2 = O
    // Negate the sign of b1, b2 by negating the points R1, R2 if b is positive,
    // or keeping them if b is negative (since -b * R = |b| * (-R))
    let r1_adj = if b1_sign { r1.clone() } else { r1.neg() };
    let r2_adj = if b2_sign { r2.clone() } else { r2.neg() };

    // Scalars for the 4-scalar Shamir: a1, a2, |b1|, |b2| (all ≤ 128 bits)
    let scalars = [a1_val, a2_val, b1_val, b2_val];
    let points_arr = [
        if a1_sign { g.neg() } else { g },
        if a2_sign { q.neg() } else { q },
        r1_adj,
        r2_adj,
    ];

    // Step 7: 4-scalar 128-bit Shamir — should equal O (infinity)
    let check_point = shamir_4x128(scalars, points_arr);
    if !check_point.is_infinity() {
        spoil_proof();
    }

    // Step 8: Check (R1 + R2).x mod n == r
    let r_sum = r1.add(&r2);
    if r_sum.is_infinity() {
        return Err(P256Error::RxMismatch);
    }

    let mut rx = r_sum.x();
    if is_fr_non_canonical(&rx.e()) {
        rx = rx.sub(&P256Fq::from_u64_arr_unchecked(&crate::P256_ORDER));
    }
    if rx.e() != r.e() {
        return Err(P256Error::RxMismatch);
    }
    Ok(())
}
