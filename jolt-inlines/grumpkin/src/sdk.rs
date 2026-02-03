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

/// Halt-and-catch-fire: makes proof unsatisfiable
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

/// Unwrap that spoils proof on error (vs `.unwrap()` which allows valid proof of panic)
pub trait UnwrapOrSpoilProof<T> {
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

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum GrumpkinError {
    InvalidFqElement,
    InvalidFrElement,
    NotOnCurve,
}

/// Wrapper around ark_grumpkin::Fq with inline-accelerated division
#[derive(Clone, PartialEq, Debug)]
pub struct GrumpkinFq {
    e: ark_grumpkin::Fq,
}

impl GrumpkinFq {
    #[inline(always)]
    pub fn new(e: Fq) -> Self {
        GrumpkinFq { e }
    }
    /// Converts from standard form to Montgomery. Returns error if >= modulus.
    #[inline(always)]
    pub fn from_u64_arr(arr: &[u64; 4]) -> Result<Self, GrumpkinError> {
        Fq::from_bigint(BigInt(*arr))
            .map(|e| GrumpkinFq { e })
            .ok_or(GrumpkinError::InvalidFqElement)
    }
    /// SAFETY: input must be in canonical Montgomery form
    #[inline(always)]
    pub fn from_u64_arr_unchecked(arr: &[u64; 4]) -> Self {
        GrumpkinFq {
            e: Fq::new_unchecked(BigInt(*arr)),
        }
    }
    #[inline(always)]
    pub fn fq(&self) -> Fq {
        self.e
    }
    #[inline(always)]
    pub fn zero() -> Self {
        GrumpkinFq { e: Fq::zero() }
    }
    /// Precomputed -17 for curve equation y² = x³ - 17
    #[inline(always)]
    pub fn negative_seventeen() -> Self {
        GrumpkinFq {
            e: Fq::new_unchecked(BigInt([
                0xdd7056026000005a,
                0x223fa97acb319311,
                0xcc388229877910c0,
                0x34394632b724eaa,
            ])),
        }
    }
    #[inline(always)]
    pub fn is_zero(&self) -> bool {
        self.e.is_zero()
    }
    #[inline(always)]
    pub fn neg(&self) -> Self {
        GrumpkinFq { e: -self.e }
    }
    #[inline(always)]
    pub fn add(&self, other: &GrumpkinFq) -> Self {
        GrumpkinFq {
            e: self.e + other.e,
        }
    }
    #[inline(always)]
    pub fn sub(&self, other: &GrumpkinFq) -> Self {
        GrumpkinFq {
            e: self.e - other.e,
        }
    }
    #[inline(always)]
    pub fn dbl(&self) -> Self {
        GrumpkinFq { e: self.e.double() }
    }
    #[inline(always)]
    pub fn tpl(&self) -> Self {
        GrumpkinFq {
            e: self.e.double() + self.e,
        }
    }
    #[inline(always)]
    pub fn mul(&self, other: &GrumpkinFq) -> Self {
        GrumpkinFq {
            e: self.e * other.e,
        }
    }
    #[inline(always)]
    pub fn square(&self) -> Self {
        GrumpkinFq { e: self.e.square() }
    }
    /// SAFETY: caller must ensure other != 0
    #[cfg(all(
        not(feature = "host"),
        any(target_arch = "riscv32", target_arch = "riscv64")
    ))]
    #[inline(always)]
    pub fn div_unchecked(&self, other: &GrumpkinFq) -> Self {
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
        let tmp = other.mul(&c);
        // Verify advice: c must be canonical and other * c == self
        if is_fq_non_canonical(&c.e.0 .0) || is_not_equal(&tmp.e.0 .0, &self.e.0 .0) {
            hcf(); // Spoils proof - assert_eq! alone doesn't suffice
        }
        c
    }
    #[cfg(all(
        not(feature = "host"),
        any(target_arch = "riscv32", target_arch = "riscv64")
    ))]
    #[inline(always)]
    pub fn div(&self, other: &GrumpkinFq) -> Self {
        if other.is_zero() {
            hcf();
        }
        self.div_unchecked(other)
    }
    #[cfg(all(
        not(feature = "host"),
        not(any(target_arch = "riscv32", target_arch = "riscv64"))
    ))]
    pub fn div_unchecked(&self, _other: &GrumpkinFq) -> Self {
        panic!("GrumpkinFq::div_unchecked called on non-RISC-V target without host feature");
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
    pub fn div_unchecked(&self, other: &GrumpkinFq) -> Self {
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
        self.div_unchecked(other)
    }
}

/// Wrapper around ark_grumpkin::Fr with inline-accelerated division
#[derive(Clone, PartialEq, Debug)]
pub struct GrumpkinFr {
    e: ark_grumpkin::Fr,
}

impl GrumpkinFr {
    #[inline(always)]
    pub fn new(e: Fr) -> Self {
        GrumpkinFr { e }
    }
    /// Converts from standard form to Montgomery. Returns error if >= modulus.
    #[inline(always)]
    pub fn from_u64_arr(arr: &[u64; 4]) -> Result<Self, GrumpkinError> {
        Fr::from_bigint(BigInt(*arr))
            .map(|e| GrumpkinFr { e })
            .ok_or(GrumpkinError::InvalidFrElement)
    }
    /// SAFETY: input must be in canonical Montgomery form
    #[inline(always)]
    pub fn from_u64_arr_unchecked(arr: &[u64; 4]) -> Self {
        GrumpkinFr {
            e: Fr::new_unchecked(BigInt(*arr)),
        }
    }
    #[inline(always)]
    pub fn fr(&self) -> Fr {
        self.e
    }
    #[inline(always)]
    pub fn zero() -> Self {
        GrumpkinFr { e: Fr::zero() }
    }
    #[inline(always)]
    pub fn is_zero(&self) -> bool {
        self.e.is_zero()
    }
    #[inline(always)]
    pub fn neg(&self) -> Self {
        GrumpkinFr { e: -self.e }
    }
    #[inline(always)]
    pub fn add(&self, other: &GrumpkinFr) -> Self {
        GrumpkinFr {
            e: self.e + other.e,
        }
    }
    #[inline(always)]
    pub fn sub(&self, other: &GrumpkinFr) -> Self {
        GrumpkinFr {
            e: self.e - other.e,
        }
    }
    #[inline(always)]
    pub fn mul(&self, other: &GrumpkinFr) -> Self {
        GrumpkinFr {
            e: self.e * other.e,
        }
    }
    #[inline(always)]
    pub fn square(&self) -> Self {
        GrumpkinFr { e: self.e.square() }
    }
    /// SAFETY: caller must ensure other != 0
    #[cfg(all(
        not(feature = "host"),
        any(target_arch = "riscv32", target_arch = "riscv64")
    ))]
    #[inline(always)]
    pub fn div_unchecked(&self, other: &GrumpkinFr) -> Self {
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
        let tmp = other.mul(&c);
        // Verify advice: c must be canonical and other * c == self
        if is_fr_non_canonical(&c.e.0 .0) || is_not_equal(&tmp.e.0 .0, &self.e.0 .0) {
            hcf(); // Spoils proof - assert_eq! alone doesn't suffice
        }
        c
    }
    #[cfg(all(
        not(feature = "host"),
        any(target_arch = "riscv32", target_arch = "riscv64")
    ))]
    #[inline(always)]
    pub fn div(&self, other: &GrumpkinFr) -> Self {
        if other.is_zero() {
            hcf();
        }
        self.div_unchecked(other)
    }
    #[cfg(all(
        not(feature = "host"),
        not(any(target_arch = "riscv32", target_arch = "riscv64"))
    ))]
    pub fn div_unchecked(&self, _other: &GrumpkinFr) -> Self {
        panic!("GrumpkinFr::div_unchecked called on non-RISC-V target without host feature");
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
    pub fn div_unchecked(&self, other: &GrumpkinFr) -> Self {
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
        self.div_unchecked(other)
    }
}

/// Affine point on Grumpkin curve y² = x³ - 17. Infinity = (0, 0).
#[derive(Clone, PartialEq, Debug)]
pub struct GrumpkinPoint {
    x: GrumpkinFq,
    y: GrumpkinFq,
}

impl GrumpkinPoint {
    #[inline(always)]
    pub fn new(x: GrumpkinFq, y: GrumpkinFq) -> Result<Self, GrumpkinError> {
        let p = GrumpkinPoint { x, y };
        if p.is_on_curve() {
            Ok(p)
        } else {
            Err(GrumpkinError::NotOnCurve)
        }
    }
    #[inline(always)]
    pub fn new_unchecked(x: GrumpkinFq, y: GrumpkinFq) -> Self {
        GrumpkinPoint { x, y }
    }
    /// Returns Montgomery form. Use `from_u64_arr_unchecked` to round-trip.
    #[inline(always)]
    pub fn to_u64_arr(&self) -> [u64; 8] {
        let mut arr = [0u64; 8];
        arr[0..4].copy_from_slice(&self.x.e.0 .0);
        arr[4..8].copy_from_slice(&self.y.e.0 .0);
        arr
    }
    /// Converts from standard form, validates on-curve.
    #[inline(always)]
    pub fn from_u64_arr(arr: &[u64; 8]) -> Result<Self, GrumpkinError> {
        let x = GrumpkinFq::from_u64_arr(&[arr[0], arr[1], arr[2], arr[3]])?;
        let y = GrumpkinFq::from_u64_arr(&[arr[4], arr[5], arr[6], arr[7]])?;
        GrumpkinPoint::new(x, y)
    }
    /// SAFETY: input must be canonical Montgomery form and on-curve (or (0,0) for infinity)
    #[inline(always)]
    pub fn from_u64_arr_unchecked(arr: &[u64; 8]) -> Self {
        let x = GrumpkinFq::from_u64_arr_unchecked(&[arr[0], arr[1], arr[2], arr[3]]);
        let y = GrumpkinFq::from_u64_arr_unchecked(&[arr[4], arr[5], arr[6], arr[7]]);
        GrumpkinPoint { x, y }
    }
    #[inline(always)]
    pub fn x(&self) -> GrumpkinFq {
        self.x.clone()
    }
    #[inline(always)]
    pub fn y(&self) -> GrumpkinFq {
        self.y.clone()
    }
    #[inline(always)]
    pub fn generator() -> Self {
        GrumpkinPoint {
            x: GrumpkinFq::new(ark_grumpkin::G_GENERATOR_X),
            y: GrumpkinFq::new(ark_grumpkin::G_GENERATOR_Y),
        }
    }
    /// (0, 0) represents infinity since it's not on the curve y² = x³ - 17
    #[inline(always)]
    pub fn infinity() -> Self {
        GrumpkinPoint {
            x: GrumpkinFq::zero(),
            y: GrumpkinFq::zero(),
        }
    }
    #[inline(always)]
    pub fn is_infinity(&self) -> bool {
        self.x.e.is_zero() && self.y.e.is_zero()
    }
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
    #[inline(always)]
    pub fn add(&self, other: &GrumpkinPoint) -> Self {
        if self.is_infinity() {
            other.clone()
        } else if other.is_infinity() {
            self.clone()
        } else if self.x == other.x && self.y == other.y {
            self.double()
        } else if self.x == other.x {
            GrumpkinPoint::infinity()
        } else {
            let s = (self.y.sub(&other.y)).div_unchecked(&self.x.sub(&other.x));
            let x2 = s.square().sub(&self.x.add(&other.x));
            let y2 = s.mul(&(self.x.sub(&x2))).sub(&self.y);
            GrumpkinPoint { x: x2, y: y2 }
        }
    }
    /// computes 2*self + other using an optimized formula that saves one field operation
    /// compared to naive double-then-add
    ///
    /// Formula derivation: Let R = P + Q (intermediate point). We compute R + P directly
    /// using the identity: the slope from P to R+P can be expressed in terms of
    /// known quantities without explicitly computing R's y-coordinate.
    #[inline(always)]
    pub fn double_and_add(&self, other: &GrumpkinPoint) -> Self {
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
            // ns = negated slope of line through P and Q
            let ns = (self.y.sub(&other.y)).div_unchecked(&other.x.sub(&self.x));
            let nx2 = other.x.sub(&ns.square());
            let divisor = self.x.dbl().add(&nx2);
            // When divisor = 0, result is infinity. This happens when Q = -2P.
            if divisor.is_zero() {
                return GrumpkinPoint::infinity();
            }
            let t = self.y.dbl().div_unchecked(&divisor).add(&ns);
            let x3 = t.square().add(&nx2);
            let y3 = t.mul(&(self.x.sub(&x3))).sub(&self.y);
            GrumpkinPoint { x: x3, y: y3 }
        }
    }
}
