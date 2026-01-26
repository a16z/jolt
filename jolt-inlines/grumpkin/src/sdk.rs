//! grumpkin operations optimized for Jolt zkVM.

use ark_ff::{AdditiveGroup, BigInt, Field, PrimeField, Zero};
use ark_grumpkin::{Fq, Fr};

use serde::{Deserialize, Serialize};

/// Returns `true` iff `x >= p` (Fq modulus), i.e., `x` is non-canonical.
/// Manually unrolled for performance.
#[cfg(all(
    not(feature = "host"),
    any(target_arch = "riscv32", target_arch = "riscv64")
))]
#[inline(always)]
fn is_fq_non_canonical(x: &[u64; 4]) -> bool {
    if x[3] < Fq::MODULUS.0[3] {
        return false;
    } else if x[3] > Fq::MODULUS.0[3] {
        return true;
    } else if x[2] < Fq::MODULUS.0[2] {
        return false;
    } else if x[2] > Fq::MODULUS.0[2] {
        return true;
    } else if x[1] < Fq::MODULUS.0[1] {
        return false;
    } else if x[1] > Fq::MODULUS.0[1] {
        return true;
    } else {
        x[0] >= Fq::MODULUS.0[0]
    }
}

/// Returns `true` iff `x >= n` (Fr modulus), i.e., `x` is non-canonical.
/// Manually unrolled for performance.
#[cfg(all(
    not(feature = "host"),
    any(target_arch = "riscv32", target_arch = "riscv64")
))]
#[inline(always)]
fn is_fr_non_canonical(x: &[u64; 4]) -> bool {
    if x[3] < Fr::MODULUS.0[3] {
        return false;
    } else if x[3] > Fr::MODULUS.0[3] {
        return true;
    } else if x[2] < Fr::MODULUS.0[2] {
        return false;
    } else if x[2] > Fr::MODULUS.0[2] {
        return true;
    } else if x[1] < Fr::MODULUS.0[1] {
        return false;
    } else if x[1] > Fr::MODULUS.0[1] {
        return true;
    } else {
        x[0] >= Fr::MODULUS.0[0]
    }
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
    /// two variants: one that assumes other != 0,
    /// and one that panics if other == 0 (spoiling the proof)
    /// then calling the former
    #[cfg(all(
        not(feature = "host"),
        any(target_arch = "riscv32", target_arch = "riscv64")
    ))]
    #[inline(always)]
    pub fn div_assume_nonzero(&self, other: &GrumpkinFq) -> Self {
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
        // if not canonical (>= n), or other * c is not equal to self, panic
        if is_fq_non_canonical(&c.e.0 .0) || is_not_equal(&tmp.e.0 .0, &self.e.0 .0) {
            // literal assembly to induce a panic (spoils the proof)
            // merely using assert_eq! here is insufficient as it doesn't
            // spoil the proof
            hcf();
        }
        c
    }
    #[cfg(all(
        not(feature = "host"),
        any(target_arch = "riscv32", target_arch = "riscv64")
    ))]
    #[inline(always)]
    pub fn div(&self, other: &GrumpkinFq) -> Self {
        // panic and spoil the proof if dividing by zero
        if other.is_zero() {
            hcf();
        }
        self.div_assume_nonzero(other)
    }
    #[cfg(all(
        not(feature = "host"),
        not(any(target_arch = "riscv32", target_arch = "riscv64"))
    ))]
    pub fn div_assume_nonzero(&self, _other: &GrumpkinFq) -> Self {
        panic!("GrumpkinFq::div_assume_nonzero called on non-RISC-V target without host feature");
    }
    #[cfg(all(
        not(feature = "host"),
        not(any(target_arch = "riscv32", target_arch = "riscv64"))
    ))]
    pub fn div(&self, _other: &GrumpkinFq) -> Self {
        panic!("GrumpkinFq::div called on non-RISC-V target without host feature");
    }
    #[cfg(feature = "host")]
    #[inline(always)]
    pub fn div_assume_nonzero(&self, other: &GrumpkinFq) -> Self {
        GrumpkinFq {
            e: self.e / other.e,
        }
    }
    #[cfg(feature = "host")]
    #[inline(always)]
    pub fn div(&self, other: &GrumpkinFq) -> Self {
        if other.is_zero() {
            panic!("division by zero in GrumpkinFq::div");
        }
        self.div_assume_nonzero(other)
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
    /// two variants: one that assumes other != 0,
    /// and one that panics if other == 0 (spoiling the proof)
    /// then calling the former
    #[cfg(all(
        not(feature = "host"),
        any(target_arch = "riscv32", target_arch = "riscv64")
    ))]
    #[inline(always)]
    pub fn div_assume_nonzero(&self, other: &GrumpkinFr) -> Self {
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
    #[cfg(all(
        not(feature = "host"),
        any(target_arch = "riscv32", target_arch = "riscv64")
    ))]
    #[inline(always)]
    pub fn div(&self, other: &GrumpkinFr) -> Self {
        // panic and spoil the proof if dividing by zero
        if other.is_zero() {
            hcf();
        }
        self.div_assume_nonzero(other)
    }
    #[cfg(all(
        not(feature = "host"),
        not(any(target_arch = "riscv32", target_arch = "riscv64"))
    ))]
    pub fn div_assume_nonzero(&self, _other: &GrumpkinFr) -> Self {
        panic!("GrumpkinFr::div_assume_nonzero called on non-RISC-V target without host feature");
    }
    #[cfg(all(
        not(feature = "host"),
        not(any(target_arch = "riscv32", target_arch = "riscv64"))
    ))]
    pub fn div(&self, _other: &GrumpkinFr) -> Self {
        panic!("GrumpkinFr::div called on non-RISC-V target without host feature");
    }
    #[cfg(feature = "host")]
    #[inline(always)]
    pub fn div_assume_nonzero(&self, other: &GrumpkinFr) -> Self {
        GrumpkinFr {
            e: self.e / other.e,
        }
    }
    #[cfg(feature = "host")]
    #[inline(always)]
    pub fn div(&self, other: &GrumpkinFr) -> Self {
        if other.is_zero() {
            panic!("division by zero in GrumpkinFr::div");
        }
        self.div_assume_nonzero(other)
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
    /// converts the point to a [u64; 8] array
    #[inline(always)]
    pub fn to_u64_arr(&self) -> [u64; 8] {
        let mut arr = [0u64; 8];
        arr[0..4].copy_from_slice(&self.x.e.0 .0);
        arr[4..8].copy_from_slice(&self.y.e.0 .0);
        arr
    }
    /// creates a GrumpkinPoint from a [u64; 8] array in normal form
    /// performs checks to ensure that the point is on the curve
    /// and that the coordinates are well formed
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
            let s = self.x.square().tpl().div_assume_nonzero(&self.y.dbl());
            let x2 = s.square().sub(&self.x.dbl());
            let y2 = s.mul(&(self.x.sub(&x2))).sub(&self.y);
            GrumpkinPoint { x: x2, y: y2 }
        }
    }
    /// adds two points on the grumpkin curve
    #[inline(always)]
    pub fn add(&self, other: &GrumpkinPoint) -> Self {
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
            GrumpkinPoint::infinity()
        // if the x coordinates are not equal and not infinity, perform standard point addition
        } else {
            let s = (self.y.sub(&other.y)).div_assume_nonzero(&self.x.sub(&other.x));
            let x2 = s.square().sub(&self.x.add(&other.x));
            let y2 = s.mul(&(self.x.sub(&x2))).sub(&self.y);
            GrumpkinPoint { x: x2, y: y2 }
        }
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
            let ns = (self.y.sub(&other.y)).div_assume_nonzero(&other.x.sub(&self.x));
            let nx2 = other.x.sub(&ns.square());
            let t = self
                .y
                .dbl()
                .div_assume_nonzero(&self.x.dbl().add(&nx2))
                .add(&ns);
            let x3 = t.square().add(&nx2);
            let y3 = t.mul(&(self.x.sub(&x3))).sub(&self.y);
            GrumpkinPoint { x: x3, y: y3 }
        }
    }
}
