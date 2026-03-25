//! grumpkin operations optimized for Jolt zkVM.

use ark_ff::{AdditiveGroup, BigInt, Field, PrimeField, Zero};
use ark_grumpkin::{Fq, Fr};
use core::marker::PhantomData;

use serde::{Deserialize, Serialize};

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
pub use jolt_inlines_sdk::{spoil_proof, UnwrapOrSpoilProof};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum GrumpkinError {
    InvalidFqElement,
    InvalidFrElement,
    NotOnCurve,
}

/// Both `Fq` and `Fr` are `Fp<Config, 4>`, so we can access `.0` (BigInt<4>)
/// and `.0.0` ([u64; 4]) uniformly. These helpers bridge the concrete `Fp` layout
/// through the `PrimeField` trait bound.
pub trait GrumpkinFieldConfig: 'static {
    type ArkField: PrimeField + Field + Copy;
    const DIV_FUNCT3: u32;

    fn from_bigint(repr: BigInt<4>) -> Option<Self::ArkField>;
    fn new_unchecked(repr: BigInt<4>) -> Self::ArkField;
    /// Borrow the inner Montgomery limbs without copying.
    fn limbs(e: &Self::ArkField) -> &[u64; 4];
    fn limbs_mut(e: &mut Self::ArkField) -> &mut [u64; 4];

    #[cfg(all(
        not(feature = "host"),
        any(target_arch = "riscv32", target_arch = "riscv64")
    ))]
    fn is_non_canonical(limbs: &[u64; 4]) -> bool;

    fn invalid_element_error() -> GrumpkinError;
}

pub enum GrumpkinFqConfig {}

impl GrumpkinFieldConfig for GrumpkinFqConfig {
    type ArkField = Fq;
    const DIV_FUNCT3: u32 = crate::GRUMPKIN_DIVQ_ADV_FUNCT3;

    #[inline(always)]
    fn from_bigint(repr: BigInt<4>) -> Option<Fq> {
        Fq::from_bigint(repr)
    }
    #[inline(always)]
    fn new_unchecked(repr: BigInt<4>) -> Fq {
        Fq::new_unchecked(repr)
    }
    #[inline(always)]
    fn limbs(e: &Fq) -> &[u64; 4] {
        &e.0 .0
    }
    #[inline(always)]
    fn limbs_mut(e: &mut Fq) -> &mut [u64; 4] {
        &mut e.0 .0
    }

    #[cfg(all(
        not(feature = "host"),
        any(target_arch = "riscv32", target_arch = "riscv64")
    ))]
    #[inline(always)]
    fn is_non_canonical(limbs: &[u64; 4]) -> bool {
        is_fq_non_canonical(limbs)
    }

    #[inline(always)]
    fn invalid_element_error() -> GrumpkinError {
        GrumpkinError::InvalidFqElement
    }
}

pub enum GrumpkinFrConfig {}

impl GrumpkinFieldConfig for GrumpkinFrConfig {
    type ArkField = Fr;
    const DIV_FUNCT3: u32 = crate::GRUMPKIN_DIVR_ADV_FUNCT3;

    #[inline(always)]
    fn from_bigint(repr: BigInt<4>) -> Option<Fr> {
        Fr::from_bigint(repr)
    }
    #[inline(always)]
    fn new_unchecked(repr: BigInt<4>) -> Fr {
        Fr::new_unchecked(repr)
    }
    #[inline(always)]
    fn limbs(e: &Fr) -> &[u64; 4] {
        &e.0 .0
    }
    #[inline(always)]
    fn limbs_mut(e: &mut Fr) -> &mut [u64; 4] {
        &mut e.0 .0
    }

    #[cfg(all(
        not(feature = "host"),
        any(target_arch = "riscv32", target_arch = "riscv64")
    ))]
    #[inline(always)]
    fn is_non_canonical(limbs: &[u64; 4]) -> bool {
        is_fr_non_canonical(limbs)
    }

    #[inline(always)]
    fn invalid_element_error() -> GrumpkinError {
        GrumpkinError::InvalidFrElement
    }
}

pub struct GrumpkinField<C: GrumpkinFieldConfig> {
    e: C::ArkField,
    _phantom: PhantomData<C>,
}

pub type GrumpkinFq = GrumpkinField<GrumpkinFqConfig>;
pub type GrumpkinFr = GrumpkinField<GrumpkinFrConfig>;

impl<C: GrumpkinFieldConfig> Clone for GrumpkinField<C> {
    #[inline(always)]
    fn clone(&self) -> Self {
        Self {
            e: self.e,
            _phantom: PhantomData,
        }
    }
}

impl<C: GrumpkinFieldConfig> PartialEq for GrumpkinField<C> {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.e == other.e
    }
}

impl<C: GrumpkinFieldConfig> core::fmt::Debug for GrumpkinField<C> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("GrumpkinField").field("e", &self.e).finish()
    }
}

impl<C: GrumpkinFieldConfig> GrumpkinField<C> {
    #[inline(always)]
    pub fn from_u64_arr(arr: &[u64; 4]) -> Result<Self, GrumpkinError> {
        C::from_bigint(BigInt(*arr))
            .map(|e| Self {
                e,
                _phantom: PhantomData,
            })
            .ok_or(C::invalid_element_error())
    }

    /// SAFETY: input must be in canonical Montgomery form
    #[inline(always)]
    pub fn from_u64_arr_unchecked(arr: &[u64; 4]) -> Self {
        Self {
            e: C::new_unchecked(BigInt(*arr)),
            _phantom: PhantomData,
        }
    }

    #[inline(always)]
    pub fn zero() -> Self {
        Self {
            e: C::ArkField::zero(),
            _phantom: PhantomData,
        }
    }

    #[inline(always)]
    pub fn is_zero(&self) -> bool {
        self.e.is_zero()
    }

    #[inline(always)]
    pub fn neg(&self) -> Self {
        Self {
            e: -self.e,
            _phantom: PhantomData,
        }
    }

    #[inline(always)]
    pub fn add(&self, other: &Self) -> Self {
        Self {
            e: self.e + other.e,
            _phantom: PhantomData,
        }
    }

    #[inline(always)]
    pub fn sub(&self, other: &Self) -> Self {
        Self {
            e: self.e - other.e,
            _phantom: PhantomData,
        }
    }

    #[inline(always)]
    pub fn dbl(&self) -> Self {
        Self {
            e: self.e.double(),
            _phantom: PhantomData,
        }
    }

    #[inline(always)]
    pub fn tpl(&self) -> Self {
        Self {
            e: self.e.double() + self.e,
            _phantom: PhantomData,
        }
    }

    #[inline(always)]
    pub fn mul(&self, other: &Self) -> Self {
        Self {
            e: self.e * other.e,
            _phantom: PhantomData,
        }
    }

    #[inline(always)]
    pub fn square(&self) -> Self {
        Self {
            e: self.e.square(),
            _phantom: PhantomData,
        }
    }

    /// SAFETY: caller must ensure other != 0
    #[cfg(all(
        not(feature = "host"),
        any(target_arch = "riscv32", target_arch = "riscv64")
    ))]
    #[inline(always)]
    pub fn div_assume_nonzero(&self, other: &Self) -> Self {
        let mut c = Self::zero();
        unsafe {
            use crate::{GRUMPKIN_FUNCT7, INLINE_OPCODE};
            core::arch::asm!(
                ".insn r {opcode}, {funct3}, {funct7}, {rd}, {rs1}, {rs2}",
                opcode = const INLINE_OPCODE,
                funct3 = const C::DIV_FUNCT3,
                funct7 = const GRUMPKIN_FUNCT7,
                rd = in(reg) C::limbs_mut(&mut c.e).as_mut_ptr(),
                rs1 = in(reg) C::limbs(&self.e).as_ptr(),
                rs2 = in(reg) C::limbs(&other.e).as_ptr(),
                options(nostack)
            );
        }
        let tmp = other.mul(&c);
        if C::is_non_canonical(C::limbs(&c.e)) || is_not_equal(C::limbs(&tmp.e), C::limbs(&self.e))
        {
            spoil_proof();
        }
        c
    }

    #[cfg(all(
        not(feature = "host"),
        any(target_arch = "riscv32", target_arch = "riscv64")
    ))]
    #[inline(always)]
    pub fn div(&self, other: &Self) -> Self {
        if other.is_zero() {
            spoil_proof();
        }
        self.div_assume_nonzero(other)
    }

    #[cfg(all(
        not(feature = "host"),
        not(any(target_arch = "riscv32", target_arch = "riscv64"))
    ))]
    pub fn div_assume_nonzero(&self, _other: &Self) -> Self {
        panic!(
            "GrumpkinField::div_assume_nonzero called on non-RISC-V target without host feature"
        );
    }

    #[cfg(all(
        not(feature = "host"),
        not(any(target_arch = "riscv32", target_arch = "riscv64"))
    ))]
    pub fn div(&self, _other: &Self) -> Self {
        panic!("GrumpkinField::div called on non-RISC-V target without host feature");
    }

    #[cfg(feature = "host")]
    #[inline(always)]
    pub fn div_assume_nonzero(&self, other: &Self) -> Self {
        Self {
            e: self.e / other.e,
            _phantom: PhantomData,
        }
    }

    #[cfg(feature = "host")]
    #[inline(always)]
    pub fn div(&self, other: &Self) -> Self {
        if other.is_zero() {
            panic!("division by zero in GrumpkinField::div");
        }
        self.div_assume_nonzero(other)
    }
}

// Operator impls for &GrumpkinField<C>

impl<C: GrumpkinFieldConfig> core::ops::Add<&GrumpkinField<C>> for &GrumpkinField<C> {
    type Output = GrumpkinField<C>;
    #[inline(always)]
    fn add(self, rhs: &GrumpkinField<C>) -> GrumpkinField<C> {
        GrumpkinField {
            e: self.e + rhs.e,
            _phantom: PhantomData,
        }
    }
}

impl<C: GrumpkinFieldConfig> core::ops::Sub<&GrumpkinField<C>> for &GrumpkinField<C> {
    type Output = GrumpkinField<C>;
    #[inline(always)]
    fn sub(self, rhs: &GrumpkinField<C>) -> GrumpkinField<C> {
        GrumpkinField {
            e: self.e - rhs.e,
            _phantom: PhantomData,
        }
    }
}

impl<C: GrumpkinFieldConfig> core::ops::Mul<&GrumpkinField<C>> for &GrumpkinField<C> {
    type Output = GrumpkinField<C>;
    #[inline(always)]
    fn mul(self, rhs: &GrumpkinField<C>) -> GrumpkinField<C> {
        GrumpkinField {
            e: self.e * rhs.e,
            _phantom: PhantomData,
        }
    }
}

impl<C: GrumpkinFieldConfig> core::ops::Neg for &GrumpkinField<C> {
    type Output = GrumpkinField<C>;
    #[inline(always)]
    fn neg(self) -> GrumpkinField<C> {
        GrumpkinField {
            e: -self.e,
            _phantom: PhantomData,
        }
    }
}

// Operator impls for GrumpkinField<C> (delegate to &self)

impl<C: GrumpkinFieldConfig> core::ops::Add<&GrumpkinField<C>> for GrumpkinField<C> {
    type Output = GrumpkinField<C>;
    #[inline(always)]
    fn add(self, rhs: &GrumpkinField<C>) -> GrumpkinField<C> {
        &self + rhs
    }
}

impl<C: GrumpkinFieldConfig> core::ops::Sub<&GrumpkinField<C>> for GrumpkinField<C> {
    type Output = GrumpkinField<C>;
    #[inline(always)]
    fn sub(self, rhs: &GrumpkinField<C>) -> GrumpkinField<C> {
        &self - rhs
    }
}

impl<C: GrumpkinFieldConfig> core::ops::Mul<&GrumpkinField<C>> for GrumpkinField<C> {
    type Output = GrumpkinField<C>;
    #[inline(always)]
    fn mul(self, rhs: &GrumpkinField<C>) -> GrumpkinField<C> {
        &self * rhs
    }
}

impl<C: GrumpkinFieldConfig> core::ops::Neg for GrumpkinField<C> {
    type Output = GrumpkinField<C>;
    #[inline(always)]
    fn neg(self) -> GrumpkinField<C> {
        -&self
    }
}

// Fq-specific methods

impl GrumpkinField<GrumpkinFqConfig> {
    #[inline(always)]
    pub fn new(e: Fq) -> Self {
        Self {
            e,
            _phantom: PhantomData,
        }
    }

    #[inline(always)]
    pub fn fq(&self) -> Fq {
        self.e
    }

    /// Precomputed -17 for curve equation y^2 = x^3 - 17
    #[inline(always)]
    pub fn negative_seventeen() -> Self {
        Self {
            e: Fq::new_unchecked(BigInt([
                0xdd7056026000005a,
                0x223fa97acb319311,
                0xcc388229877910c0,
                0x34394632b724eaa,
            ])),
            _phantom: PhantomData,
        }
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
    fn dbl(&self) -> Self {
        self.dbl()
    }
    #[inline(always)]
    fn tpl(&self) -> Self {
        self.tpl()
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
        *GrumpkinFqConfig::limbs(&self.e)
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

// Fr-specific methods

impl GrumpkinField<GrumpkinFrConfig> {
    #[inline(always)]
    pub fn new(e: Fr) -> Self {
        Self {
            e,
            _phantom: PhantomData,
        }
    }

    #[inline(always)]
    pub fn fr(&self) -> Fr {
        self.e
    }
}

// Curve definition

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
