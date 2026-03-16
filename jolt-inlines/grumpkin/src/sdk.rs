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

use jolt_inlines_sdk::ec::{AffinePoint, CurveParams, ECField};
pub use jolt_inlines_sdk::{hcf, UnwrapOrSpoilProof};

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
    pub fn div_assume_nonzero(&self, other: &GrumpkinFq) -> Self {
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

impl ECField for GrumpkinFq {
    type Error = GrumpkinError;
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
        self.e.0 .0
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
    pub fn div_assume_nonzero(&self, other: &GrumpkinFr) -> Self {
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

#[derive(Clone)]
pub struct GrumpkinCurve;

impl CurveParams<GrumpkinFq> for GrumpkinCurve {
    type Error = GrumpkinError;
    const DOUBLE_AND_ADD_DIVISOR_CHECK: bool = true;
    fn curve_b() -> GrumpkinFq {
        GrumpkinFq::negative_seventeen()
    }
    fn not_on_curve_error() -> Self::Error {
        GrumpkinError::NotOnCurve
    }
}

pub type GrumpkinPoint = AffinePoint<GrumpkinFq, GrumpkinCurve>;

pub trait GrumpkinPointExt {
    fn generator() -> GrumpkinPoint;
}

impl GrumpkinPointExt for GrumpkinPoint {
    #[inline(always)]
    fn generator() -> Self {
        Self::new_unchecked(
            GrumpkinFq::new(ark_grumpkin::G_GENERATOR_X),
            GrumpkinFq::new(ark_grumpkin::G_GENERATOR_Y),
        )
    }
}
