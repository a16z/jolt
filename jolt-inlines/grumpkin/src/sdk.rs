//! grumpkin operations optimized for Jolt zkVM.

#[cfg(feature = "host")]
use ark_ff::BigInteger;
use ark_ff::{AdditiveGroup, BigInt, Field, PrimeField, Zero};
use ark_grumpkin::{Fq, Fr};

#[cfg(feature = "host")]
use num_bigint::BigInt as NBigInt;
#[cfg(feature = "host")]
use num_bigint::Sign;
#[cfg(feature = "host")]
use num_integer::Integer;

use serde::{Deserialize, Serialize};

/// Returns `true` iff `x >= p` (Fq modulus), i.e., `x` is non-canonical.
/// Specialized: since p's upper 3 limbs are all u64::MAX, x >= p iff
/// all upper 3 limbs are MAX and limb[0] >= Fq::MODULUS.0[0].
#[cfg(all(
    not(feature = "host"),
    any(target_arch = "riscv32", target_arch = "riscv64")
))]
#[inline(always)]
fn is_fq_non_canonical(x: &[u64; 4]) -> bool {
    let m = Fq::MODULUS.0;
    if x[3] > m[3] {
        return true;
    }
    if x[3] < m[3] {
        return false;
    }
    if x[2] > m[2] {
        return true;
    }
    if x[2] < m[2] {
        return false;
    }
    if x[1] > m[1] {
        return true;
    }
    if x[1] < m[1] {
        return false;
    }
    x[0] >= m[0]
}

/// Returns `true` iff `x >= n` (Fr modulus), i.e., `x` is non-canonical.
#[cfg(all(
    not(feature = "host"),
    any(target_arch = "riscv32", target_arch = "riscv64")
))]
#[inline(always)]
fn is_fr_non_canonical(x: &[u64; 4]) -> bool {
    let m = Fr::MODULUS.0;
    if x[3] > m[3] {
        return true;
    }
    if x[3] < m[3] {
        return false;
    }
    if x[2] > m[2] {
        return true;
    }
    if x[2] < m[2] {
        return false;
    }
    if x[1] > m[1] {
        return true;
    }
    if x[1] < m[1] {
        return false;
    }
    x[0] >= m[0]
}

#[cfg(all(
    not(feature = "host"),
    any(target_arch = "riscv32", target_arch = "riscv64")
))]
#[inline(always)]
fn is_not_equal(x: &[u64; 4], y: &[u64; 4]) -> bool {
    x[0] != y[0] || x[1] != y[1] || x[2] != y[2] || x[3] != y[3]
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

impl<T> UnwrapOrSpoilProof<T> for Result<T, GrumpkinError> {
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

/// Error types for grumpkin operations
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum GrumpkinError {
    InvalidFqElement, // input array does not correspond to a valid Fq element
    InvalidFrElement, // input array does not correspond to a valid Fr element
    NotOnCurve,       // point is not on the grumpkin curve
}

// Grumpkin GLV endomorphism constants (Montgomery form).
// The endomorphism is (x, y) -> (beta * x, y), where beta^3 = 1.
pub(crate) const GRUMPKIN_ENDO_BETA_LIMBS: [u64; 4] = [
    244305545194690131,
    8351807910065594880,
    14266533074055306532,
    404339206190769364,
];

#[allow(dead_code)]
pub(crate) const GRUMPKIN_GLV_LAMBDA_LIMBS: [u64; 4] = [
    3697675806616062876,
    9065277094688085689,
    6918009208039626314,
    2775033306905974752,
];

/// grumpkin base field element
/// in montgomery form
/// as a wrapper around the arkworks implementation
/// so that various operations can be replaced with inlines
#[derive(Clone, PartialEq, Debug)]
pub struct GrumpkinFq {
    e: ark_grumpkin::Fq,
}

impl GrumpkinFq {
    /// creates a new GrumpkinFq element from an Fq element
    #[inline(always)]
    pub fn new(e: Fq) -> Self {
        GrumpkinFq { e }
    }
    /// creates a new GrumpkinFq element from a [u64; 4] array
    /// performs conversion to montgomery form
    /// returns Err(GrumpkinError) if the array does not correspond to a valid Fq element
    #[inline(always)]
    pub fn from_u64_arr(arr: &[u64; 4]) -> Result<Self, GrumpkinError> {
        // attempt to create a new Fq element from the array
        let e = Fq::from_bigint(BigInt(*arr));
        // if valid, return element, else return error
        match e {
            Some(val) => Ok(GrumpkinFq { e: val }),
            None => Err(GrumpkinError::InvalidFqElement),
        }
    }
    /// creates a new GrumpkinFq element from a [u64; 4] array (unchecked)
    /// the array is assumed to be in canonical montgomery form
    #[inline(always)]
    pub fn from_u64_arr_unchecked(arr: &[u64; 4]) -> Self {
        GrumpkinFq {
            e: Fq::new_unchecked(BigInt(*arr)),
        }
    }
    /// get inner Fq type
    #[inline(always)]
    pub fn fq(&self) -> Fq {
        self.e
    }
    /// returns the additive identity element (0)
    #[inline(always)]
    pub fn zero() -> Self {
        GrumpkinFq { e: Fq::zero() }
    }
    /// returns -17 in Fq
    #[inline(always)]
    pub fn negative_seventeen() -> Self {
        // derived from Fq::from(-17i64)
        // precomputed to avoid recomputation in point doubling
        GrumpkinFq {
            e: Fq::new_unchecked(BigInt([
                0xdd7056026000005a,
                0x223fa97acb319311,
                0xcc388229877910c0,
                0x34394632b724eaa,
            ])),
        }
    }
    /// returns true if the element is zero
    #[inline(always)]
    pub fn is_zero(&self) -> bool {
        self.e.is_zero()
    }
    /// returns -self
    #[inline(always)]
    pub fn neg(&self) -> Self {
        GrumpkinFq { e: -self.e }
    }
    /// returns self + other
    #[inline(always)]
    pub fn add(&self, other: &GrumpkinFq) -> Self {
        GrumpkinFq {
            e: self.e + other.e,
        }
    }
    /// returns self - other
    #[inline(always)]
    pub fn sub(&self, other: &GrumpkinFq) -> Self {
        GrumpkinFq {
            e: self.e - other.e,
        }
    }
    /// returns 2*self
    #[inline(always)]
    pub fn dbl(&self) -> Self {
        GrumpkinFq { e: self.e.double() }
    }
    /// returns 3*self
    #[inline(always)]
    pub fn tpl(&self) -> Self {
        GrumpkinFq {
            e: self.e.double() + self.e,
        }
    }
    /// returns self * other
    #[inline(always)]
    pub fn mul(&self, other: &GrumpkinFq) -> Self {
        GrumpkinFq {
            e: self.e * other.e,
        }
    }
    /// returns self^2
    #[inline(always)]
    pub fn square(&self) -> Self {
        GrumpkinFq { e: self.e.square() }
    }
    /// returns self / other
    /// uses custom inline for performance
    /// costs nearly the same as multiplication
    #[cfg(all(
        not(feature = "host"),
        any(target_arch = "riscv32", target_arch = "riscv64")
    ))]
    #[inline(always)]
    fn div_impl(&self, other: &GrumpkinFq) -> Self {
        // get inverse as pure advice
        let mut c = GrumpkinFq::zero();
        unsafe {
            use crate::{GRUMPKIN_DIVQ_ADV_FUNCT3, GRUMPKIN_FUNCT7, INLINE_OPCODE};
            core::arch::asm!(
                ".insn r {opcode}, {funct3}, {funct7}, {rd}, {rs1}, {rs2}",
                opcode = const INLINE_OPCODE,
                funct3 = const GRUMPKIN_DIVQ_ADV_FUNCT3,
                funct7 = const GRUMPKIN_FUNCT7,
                rd = in(reg) c.e.0.0.as_mut_ptr(),
                rs1 = in(reg) self.e.0.0.as_ptr(),
                rs2 = in(reg) other.e.0.0.as_ptr(),
                options(nostack)
            );
        }
        // compute tmp = other * c
        let tmp = other.mul(&c);
        // if not canonical (>= p), or other * c is not equal to self, panic
        if is_fq_non_canonical(&c.e.0 .0) || is_not_equal(&tmp.e.0 .0, &self.e.0 .0) {
            // literal assembly to induce a panic (spoils the proof)
            // merely using assert_eq! here is insufficient as it doesn't
            // spoil the proof
            hcf();
        }
        c
    }
    /// returns self / other
    /// uses custom inline for performance
    /// costs nearly the same as multiplication
    ///
    /// # Proof soundness
    /// In guest builds, division uses a non-deterministic ("advice") inverse `c` and then checks
    /// that `other * c == self` and that `c` is canonical. If `other == 0`, then `0/0` would pass
    /// the multiplicative check for any canonical `c`, so we spoil the proof explicitly.
    #[cfg(all(
        not(feature = "host"),
        any(target_arch = "riscv32", target_arch = "riscv64")
    ))]
    #[inline(always)]
    pub fn div(&self, other: &GrumpkinFq) -> Self {
        // 0/0 would pass the correctness check for any c, so reject it explicitly.
        if other.is_zero() {
            hcf();
        }
        self.div_impl(other)
    }
    /// Same as [`Self::div`], but assumes `other != 0`.
    #[cfg(all(
        not(feature = "host"),
        any(target_arch = "riscv32", target_arch = "riscv64")
    ))]
    #[inline(always)]
    pub fn div_unchecked(&self, other: &GrumpkinFq) -> Self {
        self.div_impl(other)
    }
    #[cfg(all(
        not(feature = "host"),
        not(any(target_arch = "riscv32", target_arch = "riscv64"))
    ))]
    pub fn div(&self, _other: &GrumpkinFq) -> Self {
        panic!("GrumpkinFq::div called on non-RISC-V target without host feature");
    }
    #[cfg(all(
        not(feature = "host"),
        not(any(target_arch = "riscv32", target_arch = "riscv64"))
    ))]
    pub fn div_unchecked(&self, _other: &GrumpkinFq) -> Self {
        panic!("GrumpkinFq::div_unchecked called on non-RISC-V target without host feature");
    }
    #[cfg(feature = "host")]
    #[inline(always)]
    pub fn div(&self, other: &GrumpkinFq) -> Self {
        GrumpkinFq {
            e: self.e / other.e,
        }
    }
    /// Same as [`Self::div`], but assumes `other != 0`.
    #[cfg(feature = "host")]
    #[inline(always)]
    pub fn div_unchecked(&self, other: &GrumpkinFq) -> Self {
        self.div(other)
    }
}

/// grumpkin scalar field element
/// in montgomery form
/// as a wrapper around the arkworks implementation
/// so that various operations can be replaced with inlines
#[derive(Clone, PartialEq, Debug)]
pub struct GrumpkinFr {
    e: ark_grumpkin::Fr,
}

impl GrumpkinFr {
    /// creates a new GrumpkinFr element from an Fr element
    #[inline(always)]
    pub fn new(e: Fr) -> Self {
        GrumpkinFr { e }
    }
    /// creates a new GrumpkinFr element from a [u64; 4] array
    /// performs conversion to montgomery form
    /// returns Err(GrumpkinError) if the array does not correspond to a valid Fr element
    #[inline(always)]
    pub fn from_u64_arr(arr: &[u64; 4]) -> Result<Self, GrumpkinError> {
        // attempt to create a new Fr element from the array
        let e = Fr::from_bigint(BigInt(*arr));
        // if valid, return element, else return error
        match e {
            Some(val) => Ok(GrumpkinFr { e: val }),
            None => Err(GrumpkinError::InvalidFrElement),
        }
    }
    /// creates a new GrumpkinFr element from a [u64; 4] array (unchecked)
    /// the array is assumed to be in canonical montgomery form
    #[inline(always)]
    pub fn from_u64_arr_unchecked(arr: &[u64; 4]) -> Self {
        GrumpkinFr {
            e: Fr::new_unchecked(BigInt(*arr)),
        }
    }
    /// get inner Fr type
    #[inline(always)]
    pub fn fr(&self) -> Fr {
        self.e
    }
    /// returns the additive identity element (0)
    #[inline(always)]
    pub fn zero() -> Self {
        GrumpkinFr { e: Fr::zero() }
    }
    /// returns true if the element is zero
    #[inline(always)]
    pub fn is_zero(&self) -> bool {
        self.e.is_zero()
    }
    /// returns -self
    #[inline(always)]
    pub fn neg(&self) -> Self {
        GrumpkinFr { e: -self.e }
    }
    /// returns self + other
    #[inline(always)]
    pub fn add(&self, other: &GrumpkinFr) -> Self {
        GrumpkinFr {
            e: self.e + other.e,
        }
    }
    /// returns self - other
    #[inline(always)]
    pub fn sub(&self, other: &GrumpkinFr) -> Self {
        GrumpkinFr {
            e: self.e - other.e,
        }
    }
    /// returns self * other
    #[inline(always)]
    pub fn mul(&self, other: &GrumpkinFr) -> Self {
        GrumpkinFr {
            e: self.e * other.e,
        }
    }
    /// returns self^2
    #[inline(always)]
    pub fn square(&self) -> Self {
        GrumpkinFr { e: self.e.square() }
    }
    /// returns self / other
    /// uses custom inline for performance
    /// costs nearly the same as multiplication
    #[cfg(all(
        not(feature = "host"),
        any(target_arch = "riscv32", target_arch = "riscv64")
    ))]
    #[inline(always)]
    fn div_impl(&self, other: &GrumpkinFr) -> Self {
        // get inverse as pure advice
        let mut c = GrumpkinFr::zero();
        unsafe {
            use crate::{GRUMPKIN_DIVR_ADV_FUNCT3, GRUMPKIN_FUNCT7, INLINE_OPCODE};
            core::arch::asm!(
                ".insn r {opcode}, {funct3}, {funct7}, {rd}, {rs1}, {rs2}",
                opcode = const INLINE_OPCODE,
                funct3 = const GRUMPKIN_DIVR_ADV_FUNCT3,
                funct7 = const GRUMPKIN_FUNCT7,
                rd = in(reg) c.e.0.0.as_mut_ptr(),
                rs1 = in(reg) self.e.0.0.as_ptr(),
                rs2 = in(reg) other.e.0.0.as_ptr(),
                options(nostack)
            );
        }
        // compute tmp = other * c
        let tmp = other.mul(&c);
        // if not canonical (>= n), or other * c is not equal to self, panic
        if is_fr_non_canonical(&c.e.0 .0) || is_not_equal(&tmp.e.0 .0, &self.e.0 .0) {
            // literal assembly to induce a panic (spoils the proof)
            // merely using assert_eq! here is insufficient as it doesn't
            // spoil the proof
            hcf();
        }
        c
    }
    /// returns self / other
    /// uses custom inline for performance
    /// costs nearly the same as multiplication
    ///
    /// # Proof soundness
    /// In guest builds, division uses a non-deterministic ("advice") inverse `c` and then checks
    /// that `other * c == self` and that `c` is canonical. If `other == 0`, then `0/0` would pass
    /// the multiplicative check for any canonical `c`, so we spoil the proof explicitly.
    #[cfg(all(
        not(feature = "host"),
        any(target_arch = "riscv32", target_arch = "riscv64")
    ))]
    #[inline(always)]
    pub fn div(&self, other: &GrumpkinFr) -> Self {
        // 0/0 would pass the correctness check for any c, so reject it explicitly.
        if other.is_zero() {
            hcf();
        }
        self.div_impl(other)
    }
    /// Same as [`Self::div`], but assumes `other != 0`.
    #[cfg(all(
        not(feature = "host"),
        any(target_arch = "riscv32", target_arch = "riscv64")
    ))]
    #[inline(always)]
    pub fn div_unchecked(&self, other: &GrumpkinFr) -> Self {
        self.div_impl(other)
    }
    #[cfg(all(
        not(feature = "host"),
        not(any(target_arch = "riscv32", target_arch = "riscv64"))
    ))]
    pub fn div(&self, _other: &GrumpkinFr) -> Self {
        panic!("GrumpkinFr::div called on non-RISC-V target without host feature");
    }
    #[cfg(all(
        not(feature = "host"),
        not(any(target_arch = "riscv32", target_arch = "riscv64"))
    ))]
    pub fn div_unchecked(&self, _other: &GrumpkinFr) -> Self {
        panic!("GrumpkinFr::div_unchecked called on non-RISC-V target without host feature");
    }
    #[cfg(feature = "host")]
    #[inline(always)]
    pub fn div(&self, other: &GrumpkinFr) -> Self {
        GrumpkinFr {
            e: self.e / other.e,
        }
    }
    /// Same as [`Self::div`], but assumes `other != 0`.
    #[cfg(feature = "host")]
    #[inline(always)]
    pub fn div_unchecked(&self, other: &GrumpkinFr) -> Self {
        self.div(other)
    }
}

/// grumpkin point in affine form
/// (as a pair of base field elements in montgomery form)
/// infinity is represented as (0, 0) because this point is not on the curve
#[derive(Clone, PartialEq, Debug)]
pub struct GrumpkinPoint {
    x: GrumpkinFq,
    y: GrumpkinFq,
}

impl GrumpkinPoint {
    /// creates a new GrumpkinPoint from two GrumpkinFq elements
    /// returns the point if it is on the curve
    /// else returns Err(GrumpkinError)
    #[inline(always)]
    pub fn new(x: GrumpkinFq, y: GrumpkinFq) -> Result<Self, GrumpkinError> {
        let p = GrumpkinPoint { x, y };
        if p.is_on_curve() {
            Ok(p)
        } else {
            Err(GrumpkinError::NotOnCurve)
        }
    }
    /// creates a new GrumpkinPoint from two GrumpkinFq elements
    /// performs no checks to ensure that the point is on the curve
    #[inline(always)]
    pub fn new_unchecked(x: GrumpkinFq, y: GrumpkinFq) -> Self {
        GrumpkinPoint { x, y }
    }
    /// Converts the point to a `[u64; 8]` array **in Montgomery form**.
    ///
    /// This is the raw arkworks internal representation of `(x, y)` and is **not**
    /// compatible with [`Self::from_u64_arr`], which expects canonical (non-Montgomery)
    /// limbs.
    ///
    /// - For a canonical (non-Montgomery) encoding, use [`Self::to_u64_arr_canonical`].
    /// - To parse this encoding, use [`Self::from_u64_arr_unchecked`] (caller must ensure
    ///   limbs are canonical and the point is on-curve).
    #[inline(always)]
    pub fn to_u64_arr(&self) -> [u64; 8] {
        let mut arr = [0u64; 8];
        arr[0..4].copy_from_slice(&self.x.e.0 .0);
        arr[4..8].copy_from_slice(&self.y.e.0 .0);
        arr
    }

    /// Converts the point to a `[u64; 8]` array in canonical (non-Montgomery) form.
    #[inline(always)]
    pub fn to_u64_arr_canonical(&self) -> [u64; 8] {
        let mut arr = [0u64; 8];
        arr[0..4].copy_from_slice(&self.x.e.into_bigint().0);
        arr[4..8].copy_from_slice(&self.y.e.into_bigint().0);
        arr
    }

    /// Creates a GrumpkinPoint from a `[u64; 8]` array in canonical (non-Montgomery) form.
    /// Performs checks to ensure that the point is on the curve and that the coordinates
    /// are well formed.
    #[inline(always)]
    pub fn from_u64_arr(arr: &[u64; 8]) -> Result<Self, GrumpkinError> {
        let x = GrumpkinFq::from_u64_arr(&[arr[0], arr[1], arr[2], arr[3]])?;
        let y = GrumpkinFq::from_u64_arr(&[arr[4], arr[5], arr[6], arr[7]])?;
        GrumpkinPoint::new(x, y)
    }
    /// creates a GrumpkinPoint from a [u64; 8] array
    /// which is assumed to correspond to x and y coordinates in montgomery form
    /// performs no checks to ensure that the point is on the curve
    /// or that the coordinates are well formed
    #[inline(always)]
    pub fn from_u64_arr_unchecked(arr: &[u64; 8]) -> Self {
        let x = GrumpkinFq::from_u64_arr_unchecked(&[arr[0], arr[1], arr[2], arr[3]]);
        let y = GrumpkinFq::from_u64_arr_unchecked(&[arr[4], arr[5], arr[6], arr[7]]);
        GrumpkinPoint { x, y }
    }
    /// get x coordinate
    #[inline(always)]
    pub fn x(&self) -> GrumpkinFq {
        self.x.clone()
    }
    /// get y coordinate
    #[inline(always)]
    pub fn y(&self) -> GrumpkinFq {
        self.y.clone()
    }
    /// get generator
    #[inline(always)]
    pub fn generator() -> Self {
        GrumpkinPoint {
            x: GrumpkinFq::new(ark_grumpkin::G_GENERATOR_X),
            y: GrumpkinFq::new(ark_grumpkin::G_GENERATOR_Y),
        }
    }
    /// generator with endomorphism applied
    #[inline(always)]
    pub fn generator_w_endomorphism() -> Self {
        Self::generator().endomorphism()
    }
    /// returns beta * self where beta is the GLV endomorphism coefficient
    #[inline(always)]
    pub fn endomorphism(&self) -> Self {
        if self.is_infinity() {
            GrumpkinPoint::infinity()
        } else {
            let beta = GrumpkinFq::from_u64_arr_unchecked(&GRUMPKIN_ENDO_BETA_LIMBS);
            GrumpkinPoint {
                x: self.x.mul(&beta),
                y: self.y.clone(),
            }
        }
    }
    /// given k, return k1 and k2 such that
    /// k = k1 + k2 * lambda
    /// and |k1|, |k2| < 2^128
    /// based on the implementation found in ec/src/scalar_mul/glv.rs in arkworks
    #[cfg(all(
        not(feature = "host"),
        any(target_arch = "riscv32", target_arch = "riscv64")
    ))]
    #[inline(always)]
    pub fn decompose_scalar(k: &GrumpkinFr) -> [(bool, u128); 2] {
        // get non-deterministic decomposition
        let mut out = [0u64; 6];
        unsafe {
            use crate::{GRUMPKIN_FUNCT7, GRUMPKIN_GLVR_ADV_FUNCT3, INLINE_OPCODE};
            core::arch::asm!(
                ".insn r {opcode}, {funct3}, {funct7}, {rd}, {rs1}, x0",
                opcode = const INLINE_OPCODE,
                funct3 = const GRUMPKIN_GLVR_ADV_FUNCT3,
                funct7 = const GRUMPKIN_FUNCT7,
                rd = in(reg) out.as_mut_ptr(),
                rs1 = in(reg) k.e.0.0.as_ptr(),
                options(nostack)
            );
        }
        // check that decomposition is correct
        // this is check that k1 + k2 * lambda == k (mod r)
        let lambda = Fr::new_unchecked(BigInt {
            0: GRUMPKIN_GLV_LAMBDA_LIMBS,
        });
        let mut k1 = Fr::from_bigint(BigInt {
            0: [out[1], out[2], 0u64, 0u64],
        })
        .unwrap();
        if out[0] == 1u64 {
            k1 = -k1;
        }
        let mut k2 = Fr::from_bigint(BigInt {
            0: [out[4], out[5], 0u64, 0u64],
        })
        .unwrap();
        if out[3] == 1u64 {
            k2 = -k2;
        }
        let recomposed_k = k1 + k2 * lambda;
        if recomposed_k != k.e {
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
    pub fn decompose_scalar(_k: &GrumpkinFr) -> [(bool, u128); 2] {
        panic!("GrumpkinPoint::decompose_scalar called on non-RISC-V target without host feature");
    }
    #[cfg(feature = "host")]
    #[inline(always)]
    pub fn decompose_scalar(k: &GrumpkinFr) -> [(bool, u128); 2] {
        let k_fr = k.e;
        let k_bigint: NBigInt = k_fr.into_bigint().into();
        let r = NBigInt::from_bytes_le(Sign::Plus, &Fr::MODULUS.to_bytes_le());
        let n11 = NBigInt::from(147946756881789319000765030803803410729i128);
        let n12 = NBigInt::from(-9931322734385697762i128);
        let n21 = NBigInt::from(9931322734385697762i128);
        let n22 = NBigInt::from(147946756881789319010696353538189108491i128);
        let beta_1 = {
            let (mut div, rem) = (&k_bigint * &n22).div_rem(&r);
            if (&rem + &rem) > r {
                div += NBigInt::from(1u8);
            }
            div
        };
        let beta_2 = {
            let n12_neg = -n12.clone();
            let (mut div, rem) = (&k_bigint * &n12_neg).div_rem(&r);
            if (&rem + &rem) > r {
                div += NBigInt::from(1u8);
            }
            div
        };
        let k1 = &k_bigint - &beta_1 * &n11 - &beta_2 * &n21;
        let k2 = -(&beta_1 * &n12 + &beta_2 * &n22);
        // convert k1, k2 to absolute values and signs
        let serialize_k = |k: NBigInt| -> (u64, [u64; 2]) {
            let sign = if k.sign() == Sign::Minus { 1u64 } else { 0u64 };
            let abs_k = if sign == 1 { -k } else { k };
            let bytes = abs_k.to_bytes_le().1;
            assert!(
                bytes.len() <= 16,
                "GLV decomposition produced out-of-range half-scalar"
            );
            let mut arr = [0u64; 2];
            for (i, b) in bytes.iter().enumerate() {
                arr[i / 8] |= (*b as u64) << ((i % 8) * 8);
            }
            (sign, arr)
        };
        let (s1, k1_arr) = serialize_k(k1);
        let (s2, k2_arr) = serialize_k(k2);
        let k1_u = (k1_arr[0] as u128) | ((k1_arr[1] as u128) << 64);
        let k2_u = (k2_arr[0] as u128) | ((k2_arr[1] as u128) << 64);
        let out = [(s1 == 1u64, k1_u), (s2 == 1u64, k2_u)];
        debug_assert!(
            {
                let lambda = Fr::new_unchecked(BigInt(GRUMPKIN_GLV_LAMBDA_LIMBS));
                let mut k1_fr_check = Fr::from(out[0].1);
                if out[0].0 {
                    k1_fr_check = -k1_fr_check;
                }
                let mut k2_fr_check = Fr::from(out[1].1);
                if out[1].0 {
                    k2_fr_check = -k2_fr_check;
                }
                let recomposed = k1_fr_check + k2_fr_check * lambda;
                recomposed == k_fr
            },
            "GLV decomposition recomposition failed"
        );
        out
    }
    /// returns the point at infinity (0, 0)
    #[inline(always)]
    pub fn infinity() -> Self {
        GrumpkinPoint {
            x: GrumpkinFq::zero(),
            y: GrumpkinFq::zero(),
        }
    }
    /// returns true if the point is the point at infinity (0, 0)
    #[inline(always)]
    pub fn is_infinity(&self) -> bool {
        self.x.e.is_zero() && self.y.e.is_zero()
    }
    /// checks if the point is on the grumpkin curve
    #[inline(always)]
    pub fn is_on_curve(&self) -> bool {
        self.is_infinity()
            || self.y.square()
                == (self
                    .x
                    .square()
                    .mul(&self.x)
                    .add(&GrumpkinFq::negative_seventeen()))
    }
    /// negates a point on the grumpkin curve
    #[inline(always)]
    pub fn neg(&self) -> Self {
        if self.is_infinity() {
            GrumpkinPoint::infinity()
        } else {
            GrumpkinPoint {
                x: self.x.clone(),
                y: self.y.neg(),
            }
        }
    }
    /// doubles a point on the grumpkin curve
    #[inline(always)]
    pub fn double(&self) -> Self {
        if self.y.is_zero() {
            GrumpkinPoint::infinity()
        } else {
            let s = self.x.square().tpl().div_unchecked(&self.y.dbl());
            let x2 = s.square().sub(&self.x.dbl());
            let y2 = s.mul(&(self.x.sub(&x2))).sub(&self.y);
            GrumpkinPoint { x: x2, y: y2 }
        }
    }
    /// adds two points on the grumpkin curve
    #[inline(always)]
    pub fn add(&self, other: &GrumpkinPoint) -> Self {
        // Fast path: handle infinity and x-equality with minimal redundant work.
        if self.is_infinity() {
            return other.clone();
        }
        if other.is_infinity() {
            return self.clone();
        }

        // If x-coordinates match, either we're doubling (same point) or adding inverses (infinity).
        if self.x == other.x {
            if self.y == other.y {
                return self.double();
            }
            return GrumpkinPoint::infinity();
        }

        let dy = self.y.sub(&other.y);
        let dx = self.x.sub(&other.x);
        let s = dy.div_unchecked(&dx);
        let x2 = s.square().sub(&self.x).sub(&other.x);
        let y2 = s.mul(&(self.x.sub(&x2))).sub(&self.y);
        GrumpkinPoint { x: x2, y: y2 }
    }
    // specialty routine for computing res = 2*self + other
    // tries to avoid doubling as much as possible
    // and avoids computing unnecessary intermediate values
    #[inline(always)]
    pub fn double_and_add(&self, other: &GrumpkinPoint) -> Self {
        // if self is infinity, then return other
        if self.is_infinity() {
            other.clone()
        // if other is infinity, then return 2*self
        } else if other.is_infinity() {
            self.add(self)
        // if self is equal to other, naive double and add
        } else if self.x == other.x && self.y == other.y {
            self.add(self).add(other)
        // if self and other are inverses, return self
        } else if self.x == other.x && self.y != other.y {
            self.clone()
        // general case, compute (self + other) + self
        // saving an operation in the middle
        // note that (self + other) cannot equal infinity or self here
        // so no special cases needed
        } else {
            let s = (self.y.sub(&other.y)).div_unchecked(&self.x.sub(&other.x));
            let x2 = s.square().sub(&self.x).sub(&other.x);
            // Edge case: if x2 == x1 then (x1 - x2) = 0 and the slope used for the
            // second addition is undefined. This occurs for valid inputs such as
            // `other = -2*self`, where the true result is infinity.
            //
            // Fall back to the generic path, which correctly handles these cases.
            let denom = self.x.sub(&x2);
            if denom.is_zero() {
                return self.double().add(other);
            }

            let t = self.y.dbl().div(&denom).sub(&s);
            let x3 = t.square().sub(&self.x).sub(&x2);
            let y3 = t.mul(&(self.x.sub(&x3))).sub(&self.y);
            GrumpkinPoint { x: x3, y: y3 }
        }
    }
}
