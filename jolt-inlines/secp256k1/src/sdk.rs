//! secp256k1 operations optimized for Jolt zkVM.

use ark_ff::AdditiveGroup;
#[cfg(feature = "host")]
use ark_ff::Field;
use ark_ff::{BigInt, PrimeField};
use ark_secp256k1::{Fq, Fr};

#[cfg(feature = "host")]
use num_bigint::BigInt as NBigInt;
#[cfg(feature = "host")]
use num_bigint::Sign;
#[cfg(feature = "host")]
use num_integer::Integer;

extern crate alloc;
use alloc::vec::Vec;
use serde::{Deserialize, Serialize};

/// Returns `true` iff `x >= p` (Fq modulus), i.e., `x` is non-canonical.
/// Specialized: since p's upper 3 limbs are all u64::MAX, x >= p iff
/// all upper 3 limbs are MAX and limb[0] >= Fq::MODULUS.0[0].
#[inline(always)]
fn is_fq_non_canonical(x: &[u64; 4]) -> bool {
    x[3] == u64::MAX && x[2] == u64::MAX && x[1] == u64::MAX && x[0] >= Fq::MODULUS.0[0]
}

/// Returns `true` iff `x >= n` (Fr modulus), i.e., `x` is non-canonical.
/// Specialized: since n's limb[3] is u64::MAX, we short-circuit if x[3] < MAX.
#[inline(always)]
fn is_fr_non_canonical(x: &[u64; 4]) -> bool {
    if x[3] != u64::MAX {
        return false;
    }
    if x[2] > Fr::MODULUS.0[2] {
        return true;
    }
    if x[2] < Fr::MODULUS.0[2] {
        return false;
    }
    if x[1] > Fr::MODULUS.0[1] {
        return true;
    }
    if x[1] < Fr::MODULUS.0[1] {
        return false;
    }
    x[0] >= Fr::MODULUS.0[0]
}

/// panic instruction
/// spoils the proof
/// used for inline checks
#[cfg(all(
    not(feature = "host"),
    any(target_arch = "riscv32", target_arch = "riscv64")
))]
#[inline(always)]
pub fn hcf() {
    unsafe {
        let u = 0u64;
        let v = 1u64;
        core::arch::asm!(
            ".insn b {opcode}, {funct3}, {rs1}, {rs2}, 0",
            opcode = const 0x5B, // virtual instruction opcode
            funct3 = const 0b001, // VirtualAssertEQ funct3
            rs1 = in(reg) u,
            rs2 = in(reg) v,
            options(nostack)
        );
    }
}
#[cfg(all(
    not(feature = "host"),
    not(any(target_arch = "riscv32", target_arch = "riscv64"))
))]
pub fn hcf() {
    panic!("hcf called on non-RISC-V target without host feature");
}
#[cfg(feature = "host")]
pub fn hcf() {
    panic!("explicit host code panic function called");
}

/// A trait for unwrapping Results in a way that spoils the proof on error.
///
/// # When to Use
///
/// Use `.unwrap_or_spoil_proof()` when you want to **assert** that a condition holds,
/// and if it doesn't, **no valid proof should exist**. This is appropriate when:
///
/// - You want to prove "X is valid" (not "I checked X")
/// - A malicious prover should not be able to produce any proof if the condition fails
/// - The error case represents something that should be cryptographically impossible
///
/// # When NOT to Use
///
/// Do NOT use `.unwrap_or_spoil_proof()` for:
///
/// - Input validation (use `.unwrap()` or return `Result` instead)
/// - Expected error cases that should be handled gracefully
/// - Cases where you want a valid proof showing the error occurred
///
/// # Example
///
/// ```ignore
/// // Soft verification - returns Result, proof is valid either way
/// let result = ecdsa_verify(z, r, s, q);
///
/// // Normal panic - proof is valid, shows program panicked
/// ecdsa_verify(z, r, s, q).unwrap();
///
/// // Spoil proof - NO valid proof can exist if signature is invalid
/// ecdsa_verify(z, r, s, q).unwrap_or_spoil_proof();
/// ```
pub trait UnwrapOrSpoilProof<T> {
    /// Unwraps the Result, returning the success value.
    ///
    /// If the Result is `Err`, this function triggers a halt-and-catch-fire (HCF)
    /// instruction that makes the proof unsatisfiable. No valid proof can be
    /// generated for an execution that reaches this error path.
    ///
    /// # Returns
    /// The unwrapped `Ok` value if successful.
    ///
    /// # Proof Implications
    /// - `Ok(v)` → Returns `v`, proof proceeds normally
    /// - `Err(_)` → Proof becomes unsatisfiable (cannot be verified)
    fn unwrap_or_spoil_proof(self) -> T;
}

impl<T> UnwrapOrSpoilProof<T> for Result<T, Secp256k1Error> {
    #[inline(always)]
    fn unwrap_or_spoil_proof(self) -> T {
        match self {
            Ok(v) => v,
            Err(_) => {
                hcf();
                unreachable!()
            }
        }
    }
}

/// Error types for secp256k1 operations
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Secp256k1Error {
    InvalidFqElement, // input array does not correspond to a valid Fq element
    InvalidFrElement, // input array does not correspond to a valid Fr element
    NotOnCurve,       // point is not on the secp256k1 curve
    QAtInfinity,      // public key is point at infinity
    ROrSZero,         // one of the signature components is zero
    RxMismatch,       // computed R.x does not match r
}

/// secp256k1 base field element
/// not in montgomery form
/// as a wrapper around 4 u64 limbs
/// so that various operations can be replaced with inlines
/// uses arkworks Fq for addition and subtraction even though
/// arkworks Fq is in montgomery form. This doesn't affect correctness
/// since addition and subtraction are the same in montgomery and
/// non-montgomery form.
/// uses arkworks Fq for host multiplication and division with appropriate conversions
/// defers to inlines for multiplication and division in guest builds
#[derive(Clone, PartialEq, Debug)]
pub struct Secp256k1Fq {
    e: [u64; 4],
}

impl Secp256k1Fq {
    /// creates a new Secp256k1Fq element from a [u64; 4] array
    /// returns Err(Secp256k1Error) if the array does not correspond to a valid Fq element
    #[inline(always)]
    pub fn from_u64_arr(arr: &[u64; 4]) -> Result<Self, Secp256k1Error> {
        if is_fq_non_canonical(arr) {
            return Err(Secp256k1Error::InvalidFqElement);
        }
        Ok(Secp256k1Fq { e: *arr })
    }
    /// creates a new Secp256k1Fq element from a [u64; 4] array (unchecked)
    /// the array is assumed to contain a value in the range [0, p)
    #[inline(always)]
    pub fn from_u64_arr_unchecked(arr: &[u64; 4]) -> Self {
        Secp256k1Fq { e: *arr }
    }
    /// get limbs
    #[inline(always)]
    pub fn e(&self) -> [u64; 4] {
        self.e
    }
    /// returns the additive identity element (0)
    #[inline(always)]
    pub fn zero() -> Self {
        Secp256k1Fq { e: [0u64; 4] }
    }
    /// returns seven
    #[inline(always)]
    pub fn seven() -> Self {
        Secp256k1Fq {
            e: [7u64, 0u64, 0u64, 0u64],
        }
    }
    /// returns true if the element is zero
    #[inline(always)]
    pub fn is_zero(&self) -> bool {
        self.e == [0u64; 4]
    }
    /// returns -self
    #[inline(always)]
    pub fn neg(&self) -> Self {
        Secp256k1Fq {
            e: (-Fq::new_unchecked(BigInt(self.e))).0 .0,
        }
    }
    /// returns self + other
    #[inline(always)]
    pub fn add(&self, other: &Secp256k1Fq) -> Self {
        Secp256k1Fq {
            e: (Fq::new_unchecked(BigInt(self.e)) + Fq::new_unchecked(BigInt(other.e)))
                .0
                 .0,
        }
    }
    /// returns self - other
    #[inline(always)]
    pub fn sub(&self, other: &Secp256k1Fq) -> Self {
        Secp256k1Fq {
            e: (Fq::new_unchecked(BigInt(self.e)) - Fq::new_unchecked(BigInt(other.e)))
                .0
                 .0,
        }
    }
    /// returns 2*self
    #[inline(always)]
    pub fn dbl(&self) -> Self {
        Secp256k1Fq {
            e: (Fq::new_unchecked(BigInt(self.e)).double()).0 .0,
        }
    }
    /// returns 3*self
    #[inline(always)]
    pub fn tpl(&self) -> Self {
        self.dbl().add(self)
    }
    /// returns self * other
    /// uses custom inline for performance
    #[cfg(all(
        not(feature = "host"),
        any(target_arch = "riscv32", target_arch = "riscv64")
    ))]
    #[inline(always)]
    pub fn mul(&self, other: &Secp256k1Fq) -> Self {
        let mut e = [0u64; 4];
        unsafe {
            use crate::{INLINE_OPCODE, SECP256K1_FUNCT7, SECP256K1_MULQ_FUNCT3};
            core::arch::asm!(
                ".insn r {opcode}, {funct3}, {funct7}, {rd}, {rs1}, {rs2}",
                opcode = const INLINE_OPCODE,
                funct3 = const SECP256K1_MULQ_FUNCT3,
                funct7 = const SECP256K1_FUNCT7,
                rd = in(reg) e.as_mut_ptr(),
                rs1 = in(reg) self.e.as_ptr(),
                rs2 = in(reg) other.e.as_ptr(),
                options(nostack)
            );
        }
        if is_fq_non_canonical(&e) {
            hcf();
        }
        Secp256k1Fq::from_u64_arr_unchecked(&e[0..4].try_into().unwrap())
    }
    #[cfg(all(
        not(feature = "host"),
        not(any(target_arch = "riscv32", target_arch = "riscv64"))
    ))]
    pub fn mul(&self, _other: &Secp256k1Fq) -> Self {
        panic!("Secp256k1Fq::mul called on non-RISC-V target without host feature");
    }
    #[cfg(feature = "host")]
    #[inline(always)]
    pub fn mul(&self, other: &Secp256k1Fq) -> Self {
        Secp256k1Fq {
            e: (Fq::new(BigInt(self.e)) * Fq::new(BigInt(other.e)))
                .into_bigint()
                .0,
        }
    }
    /// returns self^2
    /// uses custom inline for performance
    #[cfg(all(
        not(feature = "host"),
        any(target_arch = "riscv32", target_arch = "riscv64")
    ))]
    #[inline(always)]
    pub fn square(&self) -> Self {
        let mut e = [0u64; 4];
        unsafe {
            use crate::{INLINE_OPCODE, SECP256K1_FUNCT7, SECP256K1_SQUAREQ_FUNCT3};
            core::arch::asm!(
                ".insn r {opcode}, {funct3}, {funct7}, {rd}, {rs1}, x0",
                opcode = const INLINE_OPCODE,
                funct3 = const SECP256K1_SQUAREQ_FUNCT3,
                funct7 = const SECP256K1_FUNCT7,
                rd = in(reg) e.as_mut_ptr(),
                rs1 = in(reg) self.e.as_ptr(),
                options(nostack)
            );
        }
        if is_fq_non_canonical(&e) {
            hcf();
        }
        Secp256k1Fq::from_u64_arr_unchecked(&e[0..4].try_into().unwrap())
    }
    #[cfg(all(
        not(feature = "host"),
        not(any(target_arch = "riscv32", target_arch = "riscv64"))
    ))]
    pub fn square(&self) -> Self {
        panic!("Secp256k1Fq::square called on non-RISC-V target without host feature");
    }
    #[cfg(feature = "host")]
    #[inline(always)]
    pub fn square(&self) -> Self {
        Secp256k1Fq {
            e: Fq::new(BigInt(self.e)).square().into_bigint().0,
        }
    }
    /// returns self / other
    /// uses custom inline for performance
    /// assumes that other is non-zero
    #[cfg(all(
        not(feature = "host"),
        any(target_arch = "riscv32", target_arch = "riscv64")
    ))]
    #[inline(always)]
    fn div_assume_nonzero(&self, other: &Secp256k1Fq) -> Self {
        let mut e = [0u64; 4];
        unsafe {
            use crate::{INLINE_OPCODE, SECP256K1_DIVQ_FUNCT3, SECP256K1_FUNCT7};
            core::arch::asm!(
                ".insn r {opcode}, {funct3}, {funct7}, {rd}, {rs1}, {rs2}",
                opcode = const INLINE_OPCODE,
                funct3 = const SECP256K1_DIVQ_FUNCT3,
                funct7 = const SECP256K1_FUNCT7,
                rd = in(reg) e.as_mut_ptr(),
                rs1 = in(reg) self.e.as_ptr(),
                rs2 = in(reg) other.e.as_ptr(),
                options(nostack)
            );
        }
        if is_fq_non_canonical(&e) {
            hcf();
        }
        Secp256k1Fq::from_u64_arr_unchecked(&e[0..4].try_into().unwrap())
    }
    /// panics and spoils the proof if other is zero
    /// returns self / other
    /// uses custom inline for performance
    #[cfg(all(
        not(feature = "host"),
        any(target_arch = "riscv32", target_arch = "riscv64")
    ))]
    #[inline(always)]
    pub fn div(&self, other: &Secp256k1Fq) -> Self {
        // spoil proof if other == 0
        if other.is_zero() {
            hcf();
        }
        self.div_assume_nonzero(other)
    }
    #[cfg(all(
        not(feature = "host"),
        not(any(target_arch = "riscv32", target_arch = "riscv64"))
    ))]
    pub fn div_assume_nonzero(&self, _other: &Secp256k1Fq) -> Self {
        panic!("Secp256k1Fq::div_assume_nonzero called on non-RISC-V target without host feature");
    }
    #[cfg(all(
        not(feature = "host"),
        not(any(target_arch = "riscv32", target_arch = "riscv64"))
    ))]
    pub fn div(&self, _other: &Secp256k1Fq) -> Self {
        panic!("Secp256k1Fq::div called on non-RISC-V target without host feature");
    }
    /// assumes other != 0
    #[cfg(feature = "host")]
    #[inline(always)]
    pub fn div_assume_nonzero(&self, other: &Secp256k1Fq) -> Self {
        Secp256k1Fq {
            e: (Fq::new(BigInt(self.e)) / Fq::new(BigInt(other.e)))
                .into_bigint()
                .0,
        }
    }
    /// checks other != 0 then calls div_assume_nonzero
    #[cfg(feature = "host")]
    #[inline(always)]
    pub fn div(&self, other: &Secp256k1Fq) -> Self {
        if other.is_zero() {
            panic!("division by zero in Secp256k1Fq::div");
        }
        self.div_assume_nonzero(other)
    }
}

/// secp256k1 scalar field element
/// not in montgomery form
/// as a wrapper around 4 u64 limbs
/// so that various operations can be replaced with inlines
/// uses arkworks Fr for addition and subtraction even though
/// arkworks Fr is in montgomery form. This doesn't affect correctness
/// since addition and subtraction are the same in montgomery and
/// non-montgomery form.
/// uses arkworks Fr for host multiplication and division with appropriate conversions
/// defers to inlines for multiplication and division in guest builds
#[derive(Clone, PartialEq, Debug)]
pub struct Secp256k1Fr {
    e: [u64; 4],
}

impl Secp256k1Fr {
    /// creates a new Secp256k1Fr element from a [u64; 4] array
    /// returns Err(Secp256k1Error) if the array does not correspond to a valid Fr element
    #[inline(always)]
    pub fn from_u64_arr(arr: &[u64; 4]) -> Result<Self, Secp256k1Error> {
        if is_fr_non_canonical(arr) {
            return Err(Secp256k1Error::InvalidFrElement);
        }
        Ok(Secp256k1Fr { e: *arr })
    }
    /// creates a new Secp256k1Fr element from a [u64; 4] array (unchecked)
    /// the array is assumed to contain a value in the range [0, p)
    #[inline(always)]
    pub fn from_u64_arr_unchecked(arr: &[u64; 4]) -> Self {
        Secp256k1Fr { e: *arr }
    }
    /// get limbs
    #[inline(always)]
    pub fn e(&self) -> [u64; 4] {
        self.e
    }
    /// as a pair of u128s (little-endian)
    #[inline(always)]
    pub fn as_u128_pair(&self) -> (u128, u128) {
        let low = self.e[0] as u128 + ((self.e[1] as u128) << 64);
        let high = self.e[2] as u128 + ((self.e[3] as u128) << 64);
        (low, high)
    }
    /// returns the additive identity element (0)
    #[inline(always)]
    pub fn zero() -> Self {
        Secp256k1Fr { e: [0u64; 4] }
    }
    /// returns true if the element is zero
    #[inline(always)]
    pub fn is_zero(&self) -> bool {
        self.e == [0u64; 4]
    }
    /// returns -self
    #[inline(always)]
    pub fn neg(&self) -> Self {
        Secp256k1Fr {
            e: (-Fr::new_unchecked(BigInt(self.e))).0 .0,
        }
    }
    /// returns self + other
    #[inline(always)]
    pub fn add(&self, other: &Secp256k1Fr) -> Self {
        Secp256k1Fr {
            e: (Fr::new_unchecked(BigInt(self.e)) + Fr::new_unchecked(BigInt(other.e)))
                .0
                 .0,
        }
    }
    /// returns self - other
    #[inline(always)]
    pub fn sub(&self, other: &Secp256k1Fr) -> Self {
        Secp256k1Fr {
            e: (Fr::new_unchecked(BigInt(self.e)) - Fr::new_unchecked(BigInt(other.e)))
                .0
                 .0,
        }
    }
    /// returns 2*self
    #[inline(always)]
    pub fn dbl(&self) -> Self {
        Secp256k1Fr {
            e: (Fr::new_unchecked(BigInt(self.e)).double()).0 .0,
        }
    }
    /// returns 3*self
    #[inline(always)]
    pub fn tpl(&self) -> Self {
        self.dbl().add(self)
    }
    /// returns self * other
    /// uses custom inline for performance
    #[cfg(all(
        not(feature = "host"),
        any(target_arch = "riscv32", target_arch = "riscv64")
    ))]
    #[inline(always)]
    pub fn mul(&self, other: &Secp256k1Fr) -> Self {
        let mut e = [0u64; 4];
        unsafe {
            use crate::{INLINE_OPCODE, SECP256K1_FUNCT7, SECP256K1_MULR_FUNCT3};
            core::arch::asm!(
                ".insn r {opcode}, {funct3}, {funct7}, {rd}, {rs1}, {rs2}",
                opcode = const INLINE_OPCODE,
                funct3 = const SECP256K1_MULR_FUNCT3,
                funct7 = const SECP256K1_FUNCT7,
                rd = in(reg) e.as_mut_ptr(),
                rs1 = in(reg) self.e.as_ptr(),
                rs2 = in(reg) other.e.as_ptr(),
                options(nostack)
            );
        }
        if is_fr_non_canonical(&e) {
            hcf();
        }
        Secp256k1Fr::from_u64_arr_unchecked(&e[0..4].try_into().unwrap())
    }
    #[cfg(all(
        not(feature = "host"),
        not(any(target_arch = "riscv32", target_arch = "riscv64"))
    ))]
    pub fn mul(&self, _other: &Secp256k1Fr) -> Self {
        panic!("Secp256k1Fr::mul called on non-RISC-V target without host feature");
    }
    #[cfg(feature = "host")]
    #[inline(always)]
    pub fn mul(&self, other: &Secp256k1Fr) -> Self {
        Secp256k1Fr {
            e: (Fr::new(BigInt(self.e)) * Fr::new(BigInt(other.e)))
                .into_bigint()
                .0,
        }
    }
    /// returns self^2
    /// uses custom inline for performance
    #[cfg(all(
        not(feature = "host"),
        any(target_arch = "riscv32", target_arch = "riscv64")
    ))]
    #[inline(always)]
    pub fn square(&self) -> Self {
        let mut e = [0u64; 4];
        unsafe {
            use crate::{INLINE_OPCODE, SECP256K1_FUNCT7, SECP256K1_SQUARER_FUNCT3};
            core::arch::asm!(
                ".insn r {opcode}, {funct3}, {funct7}, {rd}, {rs1}, x0",
                opcode = const INLINE_OPCODE,
                funct3 = const SECP256K1_SQUARER_FUNCT3,
                funct7 = const SECP256K1_FUNCT7,
                rd = in(reg) e.as_mut_ptr(),
                rs1 = in(reg) self.e.as_ptr(),
                options(nostack)
            );
        }
        if is_fr_non_canonical(&e) {
            hcf();
        }
        Secp256k1Fr::from_u64_arr_unchecked(&e[0..4].try_into().unwrap())
    }
    #[cfg(all(
        not(feature = "host"),
        not(any(target_arch = "riscv32", target_arch = "riscv64"))
    ))]
    pub fn square(&self) -> Self {
        panic!("Secp256k1Fr::square called on non-RISC-V target without host feature");
    }
    #[cfg(feature = "host")]
    #[inline(always)]
    pub fn square(&self) -> Self {
        Secp256k1Fr {
            e: Fr::new(BigInt(self.e)).square().into_bigint().0,
        }
    }
    /// returns self / other
    /// uses custom inline for performance
    /// assumes that other is non-zero
    #[cfg(all(
        not(feature = "host"),
        any(target_arch = "riscv32", target_arch = "riscv64")
    ))]
    #[inline(always)]
    fn div_assume_nonzero(&self, other: &Secp256k1Fr) -> Self {
        let mut e = [0u64; 4];
        unsafe {
            use crate::{INLINE_OPCODE, SECP256K1_DIVR_FUNCT3, SECP256K1_FUNCT7};
            core::arch::asm!(
                ".insn r {opcode}, {funct3}, {funct7}, {rd}, {rs1}, {rs2}",
                opcode = const INLINE_OPCODE,
                funct3 = const SECP256K1_DIVR_FUNCT3,
                funct7 = const SECP256K1_FUNCT7,
                rd = in(reg) e.as_mut_ptr(),
                rs1 = in(reg) self.e.as_ptr(),
                rs2 = in(reg) other.e.as_ptr(),
                options(nostack)
            );
        }
        if is_fr_non_canonical(&e) {
            hcf();
        }
        Secp256k1Fr::from_u64_arr_unchecked(&e[0..4].try_into().unwrap())
    }
    /// panics and spoils the proof if other is zero
    /// returns self / other
    /// uses custom inline for performance
    #[cfg(all(
        not(feature = "host"),
        any(target_arch = "riscv32", target_arch = "riscv64")
    ))]
    #[inline(always)]
    pub fn div(&self, other: &Secp256k1Fr) -> Self {
        // spoil proof if other == 0
        if other.is_zero() {
            hcf();
        }
        self.div_assume_nonzero(other)
    }
    #[cfg(all(
        not(feature = "host"),
        not(any(target_arch = "riscv32", target_arch = "riscv64"))
    ))]
    pub fn div_assume_nonzero(&self, _other: &Secp256k1Fr) -> Self {
        panic!("Secp256k1Fr::div_assume_nonzero called on non-RISC-V target without host feature");
    }
    #[cfg(all(
        not(feature = "host"),
        not(any(target_arch = "riscv32", target_arch = "riscv64"))
    ))]
    pub fn div(&self, _other: &Secp256k1Fr) -> Self {
        panic!("Secp256k1Fr::div called on non-RISC-V target without host feature");
    }
    /// assumes other != 0
    #[cfg(feature = "host")]
    #[inline(always)]
    pub fn div_assume_nonzero(&self, other: &Secp256k1Fr) -> Self {
        Secp256k1Fr {
            e: (Fr::new(BigInt(self.e)) / Fr::new(BigInt(other.e)))
                .into_bigint()
                .0,
        }
    }
    /// checks other != 0 then calls div_assume_nonzero
    #[cfg(feature = "host")]
    #[inline(always)]
    pub fn div(&self, other: &Secp256k1Fr) -> Self {
        if other.is_zero() {
            panic!("division by zero in Secp256k1Fr::div");
        }
        self.div_assume_nonzero(other)
    }
}

/// secp256k1 point in affine form
/// (as a pair of base field elements in montgomery form)
/// infinity is represented as (0, 0) because this point is not on the curve
#[derive(Clone, PartialEq, Debug)]
pub struct Secp256k1Point {
    x: Secp256k1Fq,
    y: Secp256k1Fq,
}

impl Secp256k1Point {
    /// creates a new Secp256k1Point from two Secp256k1Fq elements
    /// returns the point if it is on the curve
    /// else returns Err(Secp256k1Error)
    #[inline(always)]
    pub fn new(x: Secp256k1Fq, y: Secp256k1Fq) -> Result<Self, Secp256k1Error> {
        let p = Secp256k1Point { x, y };
        if p.is_on_curve() {
            Ok(p)
        } else {
            Err(Secp256k1Error::NotOnCurve)
        }
    }
    /// creates a new Secp256k1Point from two Secp256k1Fq elements
    /// performs no checks to ensure that the point is on the curve
    #[inline(always)]
    pub fn new_unchecked(x: Secp256k1Fq, y: Secp256k1Fq) -> Self {
        Secp256k1Point { x, y }
    }
    /// converts the point to a [u64; 8] array
    #[inline(always)]
    pub fn to_u64_arr(&self) -> [u64; 8] {
        let mut arr = [0u64; 8];
        arr[0..4].copy_from_slice(&self.x.e());
        arr[4..8].copy_from_slice(&self.y.e());
        arr
    }
    /// creates a Secp256k1Point from a [u64; 8] array in normal form
    /// performs checks to ensure that the point is on the curve
    /// and that the coordinates are well formed
    #[inline(always)]
    pub fn from_u64_arr(arr: &[u64; 8]) -> Result<Self, Secp256k1Error> {
        let x = Secp256k1Fq::from_u64_arr(&[arr[0], arr[1], arr[2], arr[3]])?;
        let y = Secp256k1Fq::from_u64_arr(&[arr[4], arr[5], arr[6], arr[7]])?;
        Secp256k1Point::new(x, y)
    }
    /// creates a Secp256k1Point from a [u64; 8] array
    /// which is assumed to correspond to x and y coordinates in montgomery form
    /// performs no checks to ensure that the point is on the curve
    /// or that the coordinates are well formed
    #[inline(always)]
    pub fn from_u64_arr_unchecked(arr: &[u64; 8]) -> Self {
        let x = Secp256k1Fq::from_u64_arr_unchecked(&[arr[0], arr[1], arr[2], arr[3]]);
        let y = Secp256k1Fq::from_u64_arr_unchecked(&[arr[4], arr[5], arr[6], arr[7]]);
        Secp256k1Point { x, y }
    }
    /// get x coordinate
    #[inline(always)]
    pub fn x(&self) -> Secp256k1Fq {
        self.x.clone()
    }
    /// get y coordinate
    #[inline(always)]
    pub fn y(&self) -> Secp256k1Fq {
        self.y.clone()
    }
    /// get generator
    #[inline(always)]
    pub fn generator() -> Self {
        Secp256k1Point {
            x: Secp256k1Fq::from_u64_arr_unchecked(&[
                0x59F2815B16F81798,
                0x029BFCDB2DCE28D9,
                0x55A06295CE870B07,
                0x79BE667EF9DCBBAC,
            ]),
            y: Secp256k1Fq::from_u64_arr_unchecked(&[
                0x9C47D08FFB10D4B8,
                0xFD17B448A6855419,
                0x5DA4FBFC0E1108A8,
                0x483ADA7726A3C465,
            ]),
        }
    }
    /// 2^128 * generator
    #[inline(always)]
    pub fn generator_times_2_pow_128() -> Self {
        Secp256k1Point::from_u64_arr_unchecked(&[
            1980251557031362778,
            16756863388851544885,
            10536665754535663150,
            10333713660726959923,
            17455036783537422210,
            13540684701581249533,
            16005107816708579677,
            7361871559633811846,
        ])
    }
    /// (2^128 + 1) * generator
    #[inline(always)]
    pub fn generator_times_2_pow_128_plus_1() -> Self {
        Secp256k1Point::from_u64_arr_unchecked(&[
            7189408038385401609,
            3397911576708611646,
            15755510341721275955,
            10029532112266168108,
            2794201169108470329,
            645887947755712873,
            4931174713096550634,
            2084093948640608460,
        ])
    }
    /// generator with endomorphism applied
    #[inline(always)]
    pub fn generator_w_endomorphism() -> Self {
        Secp256k1Point::from_u64_arr_unchecked(&[
            12086430238909173707,
            9739108988881332621,
            12322133038695719717,
            13595490868124457095,
            11261198710074299576,
            18237243440184513561,
            6747795201694173352,
            5204712524664259685,
        ])
    }
    /// returns the point at infinity (0, 0)
    #[inline(always)]
    pub fn infinity() -> Self {
        Secp256k1Point {
            x: Secp256k1Fq::zero(),
            y: Secp256k1Fq::zero(),
        }
    }
    /// returns true if the point is the point at infinity (0, 0)
    #[inline(always)]
    pub fn is_infinity(&self) -> bool {
        self.x.is_zero() && self.y.is_zero()
    }
    /// checks if the point is on the secp256k1 curve
    #[inline(always)]
    pub fn is_on_curve(&self) -> bool {
        self.is_infinity()
            || self.y.square() == (self.x.square().mul(&self.x).add(&Secp256k1Fq::seven()))
    }
    /// negates a point on the secp256k1 curve
    #[inline(always)]
    pub fn neg(&self) -> Self {
        if self.is_infinity() {
            Secp256k1Point::infinity()
        } else {
            Secp256k1Point {
                x: self.x.clone(),
                y: self.y.neg(),
            }
        }
    }
    /// doubles a point on the secp256k1 curve
    #[inline(always)]
    pub fn double(&self) -> Self {
        if self.y.is_zero() {
            Secp256k1Point::infinity()
        } else {
            let s = self.x.square().tpl().div_assume_nonzero(&self.y.dbl());
            let x2 = s.square().sub(&self.x.dbl());
            let y2 = s.mul(&(self.x.sub(&x2))).sub(&self.y);
            Secp256k1Point { x: x2, y: y2 }
        }
    }
    /// adds two points on the secp256k1 curve
    #[inline(always)]
    pub fn add(&self, other: &Secp256k1Point) -> Self {
        // if either point is at infinity, return the other point
        if self.is_infinity() {
            other.clone()
        } else if other.is_infinity() {
            self.clone()
        // if the points are equal, perform point doubling
        } else if self.x == other.x && self.y == other.y {
            self.double()
        // if the x coordinates are equal but the y coordinates are not, return the point at infinity
        } else if self.x == other.x && self.y != other.y {
            Secp256k1Point::infinity()
        // if the x coordinates are not equal and not infinity, perform standard point addition
        } else {
            let s = (self.y.sub(&other.y)).div_assume_nonzero(&self.x.sub(&other.x));
            let x2 = s.square().sub(&self.x.add(&other.x));
            let y2 = s.mul(&(self.x.sub(&x2))).sub(&self.y);
            Secp256k1Point { x: x2, y: y2 }
        }
    }
    /// computes 2*self + other using an optimized formula that saves one field operation
    #[inline(always)]
    pub fn double_and_add(&self, other: &Secp256k1Point) -> Self {
        if self.is_infinity() {
            other.clone()
        } else if other.is_infinity() {
            self.add(self)
        } else if self.x == other.x && self.y == other.y {
            self.add(self).add(other)
        } else if self.x == other.x && self.y != other.y {
            // self + other = infinity, so 2*self + other = self
            self.clone()
        } else {
            let ns = (self.y.sub(&other.y)).div_assume_nonzero(&other.x.sub(&self.x));
            let nx2 = other.x.sub(&ns.square());
            let t = self
                .y
                .dbl()
                .div_assume_nonzero(&self.x.dbl().add(&nx2))
                .add(&ns);
            let x3 = t.square().add(&nx2);
            let y3 = t.mul(&(self.x.sub(&x3))).sub(&self.y);
            Secp256k1Point { x: x3, y: y3 }
        }
    }
    // returns lambda * self
    // where lambda is 0x5363ad4cc05c30e0a5261c028812645a122e22ea20816678df02967c1b23bd72
    #[inline(always)]
    pub fn endomorphism(&self) -> Self {
        if self.is_infinity() {
            Secp256k1Point::infinity()
        } else {
            // beta = 0x7AE96A2B657C07106E64479EAC3434E99CF0497512F58995C1396C28719501EE
            let beta = Secp256k1Fq::from_u64_arr_unchecked(&[
                0xc1396c28719501ee,
                0x9cf0497512f58995,
                0x6e64479eac3434e9,
                0x7ae96a2b657c0710,
            ]);
            Secp256k1Point {
                x: self.x.mul(&beta),
                y: self.y.clone(),
            }
        }
    }
    // given k, return k1 and k2 such that
    // k = k1 + k2 * lambda
    // and |k1|, |k2| < 2^128
    // based on the implementation found in ec/src/scalar_mul/glv.rs in arkworks
    #[cfg(all(
        not(feature = "host"),
        any(target_arch = "riscv32", target_arch = "riscv64")
    ))]
    #[inline(always)]
    pub fn decompose_scalar(k: &Secp256k1Fr) -> [(bool, u128); 2] {
        // get non-deterministic decomposition
        let mut out = [0u64; 6];
        unsafe {
            use crate::{INLINE_OPCODE, SECP256K1_FUNCT7, SECP256K1_GLVR_ADV_FUNCT3};
            core::arch::asm!(
                ".insn r {opcode}, {funct3}, {funct7}, {rd}, {rs1}, x0",
                opcode = const INLINE_OPCODE,
                funct3 = const SECP256K1_GLVR_ADV_FUNCT3,
                funct7 = const SECP256K1_FUNCT7,
                rd = in(reg) out.as_mut_ptr(),
                rs1 = in(reg) k.e.as_ptr(),
                options(nostack)
            );
        }
        // check that decomposition is correct
        // this is check that k1 + k2 * lambda == k (mod r)/
        let lambda = Secp256k1Fr::from_u64_arr_unchecked(&[
            0xdf02967c1b23bd72,
            0x122e22ea20816678,
            0xa5261c028812645a,
            0x5363ad4cc05c30e0,
        ]);
        let mut k1 = Secp256k1Fr::from_u64_arr_unchecked(&[out[1], out[2], 0u64, 0u64]);
        if out[0] == 1u64 {
            k1 = k1.neg();
        }
        let mut k2 = Secp256k1Fr::from_u64_arr_unchecked(&[out[4], out[5], 0u64, 0u64]);
        if out[3] == 1u64 {
            k2 = k2.neg();
        }
        let recomposed_k = k1.add(&k2.mul(&lambda));
        if recomposed_k != *k {
            hcf(); // panic and spoil proof if decomposition is incorrect
        }
        // return as (sign, abs_value) pairs
        [
            (out[0] == 1u64, (out[1] as u128) | ((out[2] as u128) << 64)),
            (out[3] == 1u64, (out[4] as u128) | ((out[5] as u128) << 64)),
        ]
    }
    #[cfg(all(
        not(feature = "host"),
        not(any(target_arch = "riscv32", target_arch = "riscv64"))
    ))]
    pub fn decompose_scalar(_k: &Secp256k1Fr) -> [(bool, u128); 2] {
        panic!("Secp256k1Point::decompose_scalar called on non-RISC-V target without host feature");
    }
    #[cfg(feature = "host")]
    #[inline(always)]
    pub fn decompose_scalar(k: &Secp256k1Fr) -> [(bool, u128); 2] {
        let k: NBigInt = Fr::new(BigInt(k.e)).into_bigint().into();
        let r = NBigInt::from_bytes_le(
            Sign::Plus,
            &[
                65, 65, 54, 208, 140, 94, 210, 191, 59, 160, 72, 175, 230, 220, 174, 186, 254, 255,
                255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
            ],
        );
        let a1 = NBigInt::from_bytes_le(
            Sign::Plus,
            &[
                21, 235, 132, 146, 228, 144, 108, 232, 205, 107, 212, 167, 33, 210, 134, 48,
            ],
        );
        let b1 = NBigInt::from_bytes_le(
            Sign::Plus,
            &[
                195, 228, 191, 10, 169, 127, 84, 111, 40, 136, 14, 1, 214, 126, 67, 228,
            ],
        );
        let a2 = NBigInt::from_bytes_le(
            Sign::Plus,
            &[
                216, 207, 68, 157, 141, 16, 193, 87, 246, 243, 226, 168, 247, 80, 202, 20, 1,
            ],
        );
        let beta_1 = {
            let (mut div, rem) = (&k * &a1).div_rem(&r);
            if (&rem + &rem) > r {
                div += NBigInt::from_bytes_le(Sign::Plus, &[1u8]);
            }
            div
        };
        let beta_2 = {
            let (mut div, rem) = (&k * &b1).div_rem(&r);
            if (&rem + &rem) > r {
                div += NBigInt::from_bytes_le(Sign::Plus, &[1u8]);
            }
            div
        };
        let k1 = &k - &beta_1 * &a1 - &beta_2 * &a2;
        let k2 = &beta_1 * &b1 - &beta_2 * &a1;
        // return as (sign, abs_value) pairs
        let to_sign_abs = |n: NBigInt| -> (bool, u128) {
            let (sign, bytes) = n.to_bytes_le();
            // pad bytes to 16 bytes
            let mut bytes_padded = bytes.clone();
            while bytes_padded.len() < 16 {
                bytes_padded.push(0u8);
            }
            let abs_value = u128::from_le_bytes(bytes_padded[..16].try_into().unwrap());
            (sign == Sign::Minus, abs_value)
        };
        [to_sign_abs(k1), to_sign_abs(k2)]
    }
}

// ECDSA signature verification function + helpers

#[inline(always)]
fn scalars_to_index(scalars: &[u128; 4], bit_index: usize) -> usize {
    let mut idx = 0;
    for (j, scalar) in scalars.iter().enumerate() {
        if (scalar >> bit_index) & 1 == 1 {
            idx |= 1 << j;
        }
    }
    idx
}

// performs a 4x128-bit scalar multiplication
// first two points assumed to be generator and 2^128 * generator
#[inline(always)]
fn secp256k1_4x128_inner_scalar_mul(
    scalars: [u128; 4],
    points: [Secp256k1Point; 2],
) -> Secp256k1Point {
    let mut lookup = Vec::<Secp256k1Point>::with_capacity(16);
    lookup.push(Secp256k1Point::infinity());
    lookup.push(Secp256k1Point::generator());
    lookup.push(Secp256k1Point::generator_times_2_pow_128());
    lookup.push(Secp256k1Point::generator_times_2_pow_128_plus_1());
    lookup.push(points[0].clone());
    lookup.push(lookup[1].add(&lookup[4]));
    lookup.push(lookup[2].add(&lookup[4]));
    lookup.push(lookup[1].add(&lookup[6]));
    lookup.push(points[1].clone());
    for i in 1..8 {
        lookup.push(lookup[i].add(&lookup[8]));
    }
    let mut res = lookup[scalars_to_index(&scalars, 127)].clone();
    for i in (0..127).rev() {
        let idx = scalars_to_index(&scalars, i);
        if idx != 0 {
            res = res.double_and_add(&lookup[idx]);
        } else {
            res = res.double();
        }
    }
    res
}

// if cond is true, negate x, otherwise return x unchanged
#[inline(always)]
fn conditional_negate(x: Secp256k1Point, cond: bool) -> Secp256k1Point {
    if cond {
        x.neg()
    } else {
        x
    }
}

/// verify an ECDSA signature
/// z is the hash of the message being signed
/// r and s are the signature components
/// q is the uncompressed public key point
/// returns Ok(()) if the signature is valid
/// returns Err(Secp256k1Error) if at any point, the verification fails
#[inline(always)]
pub fn ecdsa_verify(
    z: Secp256k1Fr,
    r: Secp256k1Fr,
    s: Secp256k1Fr,
    q: Secp256k1Point,
) -> Result<(), Secp256k1Error> {
    // step 0: check that q is not infinity
    if q.is_infinity() {
        return Result::Err(Secp256k1Error::QAtInfinity);
    }
    // step 1: check that r and s are in the correct range
    if r.is_zero() || s.is_zero() {
        return Result::Err(Secp256k1Error::ROrSZero);
    }
    // step 2: compute u1 = z / s (mod r) and u2 = r / s (mod r)
    let u1 = z.div_assume_nonzero(&s);
    let u2 = r.div_assume_nonzero(&s);
    // step 3: compute R = u1 * G + u2 * q
    // 3.1: perform the glv scalar decomposition
    let decomp_u = u1.as_u128_pair();
    let decomp_v = Secp256k1Point::decompose_scalar(&u2);
    // 3.2: get decomposed scalars as a 4x128-bit array
    //let scalars = [decomp_u[0].1, decomp_u[1].1, decomp_v[0].1, decomp_v[1].1];
    let scalars = [decomp_u.0, decomp_u.1, decomp_v[0].1, decomp_v[1].1];
    // 3.3: prepare Q, and lambda*Q, appropriately negated
    let points = [
        conditional_negate(q.clone(), decomp_v[0].0),
        conditional_negate(q.endomorphism(), decomp_v[1].0),
    ];
    // 3.4: perform the 4x128-bit scalar multiplication
    let r_claim = secp256k1_4x128_inner_scalar_mul(scalars, points);
    // step 4: check that r == R.x mod n.
    // We implement the `mod n` as a single conditional subtraction on the bigint:
    // for secp256k1, `0 <= x_R < p` and `p < 2n`, so `x_R mod n` is either `x_R` or `x_R - n`.
    let mut rx = r_claim.x;
    if is_fr_non_canonical(&rx.e()) {
        rx = rx.sub(&Secp256k1Fq::from_u64_arr_unchecked(&Fr::MODULUS.0));
    }
    if rx.e() != r.e() {
        return Result::Err(Secp256k1Error::RxMismatch);
    }
    // if all checks passed, return Ok(())
    Result::Ok(())
}
