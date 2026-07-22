//! grumpkin operations optimized for Jolt zkVM.

use ark_ff::{AdditiveGroup, BigInt, Field, PrimeField, Zero};
use ark_grumpkin::{Fq, Fr};

use core::marker::PhantomData;
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
pub use jolt_inlines_sdk::{spoil_proof, UnwrapOrSpoilProof};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum GrumpkinError {
    InvalidFqElement,
    InvalidFrElement,
    NotOnCurve,
    InvalidGlvSignWord(u64),
}

// Grumpkin GLV endomorphism constants in Montgomery form.
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

#[inline(always)]
#[cfg(any(
    all(test, feature = "host"),
    target_arch = "riscv32",
    target_arch = "riscv64"
))]
pub(crate) fn decode_glv_sign_word(sign: u64) -> Result<bool, GrumpkinError> {
    jolt_inlines_sdk::decode_sign_word(sign).ok_or(GrumpkinError::InvalidGlvSignWord(sign))
}

pub trait GrumpkinFieldConfig {
    type ArkField: AdditiveGroup + Field + PrimeField + Zero + Copy;

    const DIV_FUNCT3: u32;

    fn from_bigint(limbs: BigInt<4>) -> Option<Self::ArkField>;
    fn new_unchecked(limbs: BigInt<4>) -> Self::ArkField;
    fn limbs(e: &Self::ArkField) -> [u64; 4];
    fn limbs_ptr(e: &Self::ArkField) -> *const u64;
    fn limbs_mut_ptr(e: &mut Self::ArkField) -> *mut u64;
    fn invalid_element_error() -> GrumpkinError;
    fn div_by_zero_message() -> &'static str;

    #[cfg(all(
        not(feature = "host"),
        any(target_arch = "riscv32", target_arch = "riscv64")
    ))]
    fn is_non_canonical(limbs: &[u64; 4]) -> bool;
}

#[derive(Clone)]
pub struct GrumpkinFqConfig;

impl GrumpkinFieldConfig for GrumpkinFqConfig {
    type ArkField = Fq;

    const DIV_FUNCT3: u32 = crate::GRUMPKIN_DIVQ_ADV_FUNCT3;

    fn from_bigint(limbs: BigInt<4>) -> Option<Self::ArkField> {
        Fq::from_bigint(limbs)
    }

    fn new_unchecked(limbs: BigInt<4>) -> Self::ArkField {
        Fq::new_unchecked(limbs)
    }

    fn limbs(e: &Self::ArkField) -> [u64; 4] {
        e.0 .0
    }

    fn limbs_ptr(e: &Self::ArkField) -> *const u64 {
        e.0 .0.as_ptr()
    }

    fn limbs_mut_ptr(e: &mut Self::ArkField) -> *mut u64 {
        e.0 .0.as_mut_ptr()
    }

    fn invalid_element_error() -> GrumpkinError {
        GrumpkinError::InvalidFqElement
    }

    fn div_by_zero_message() -> &'static str {
        "division by zero in GrumpkinFq::div"
    }

    #[cfg(all(
        not(feature = "host"),
        any(target_arch = "riscv32", target_arch = "riscv64")
    ))]
    fn is_non_canonical(limbs: &[u64; 4]) -> bool {
        is_fq_non_canonical(limbs)
    }
}

#[derive(Clone)]
pub struct GrumpkinFrConfig;

impl GrumpkinFieldConfig for GrumpkinFrConfig {
    type ArkField = Fr;

    const DIV_FUNCT3: u32 = crate::GRUMPKIN_DIVR_ADV_FUNCT3;

    fn from_bigint(limbs: BigInt<4>) -> Option<Self::ArkField> {
        Fr::from_bigint(limbs)
    }

    fn new_unchecked(limbs: BigInt<4>) -> Self::ArkField {
        Fr::new_unchecked(limbs)
    }

    fn limbs(e: &Self::ArkField) -> [u64; 4] {
        e.0 .0
    }

    fn limbs_ptr(e: &Self::ArkField) -> *const u64 {
        e.0 .0.as_ptr()
    }

    fn limbs_mut_ptr(e: &mut Self::ArkField) -> *mut u64 {
        e.0 .0.as_mut_ptr()
    }

    fn invalid_element_error() -> GrumpkinError {
        GrumpkinError::InvalidFrElement
    }

    fn div_by_zero_message() -> &'static str {
        "division by zero in GrumpkinFr::div"
    }

    #[cfg(all(
        not(feature = "host"),
        any(target_arch = "riscv32", target_arch = "riscv64")
    ))]
    fn is_non_canonical(limbs: &[u64; 4]) -> bool {
        is_fr_non_canonical(limbs)
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
        f.debug_struct("GrumpkinField")
            .field("e", &C::limbs(&self.e))
            .finish()
    }
}

impl<C: GrumpkinFieldConfig> GrumpkinField<C> {
    #[inline(always)]
    pub fn new(e: C::ArkField) -> Self {
        Self {
            e,
            _phantom: PhantomData,
        }
    }

    #[inline(always)]
    pub fn from_u64_arr(arr: &[u64; 4]) -> Result<Self, GrumpkinError> {
        C::from_bigint(BigInt(*arr))
            .map(Self::new)
            .ok_or(C::invalid_element_error())
    }

    #[inline(always)]
    pub fn from_u64_arr_unchecked(arr: &[u64; 4]) -> Self {
        Self::new(C::new_unchecked(BigInt(*arr)))
    }

    #[inline(always)]
    pub fn zero() -> Self {
        Self::new(C::ArkField::zero())
    }

    #[inline(always)]
    pub fn is_zero(&self) -> bool {
        self.e.is_zero()
    }

    #[inline(always)]
    pub fn neg(&self) -> Self {
        Self::new(-self.e)
    }

    #[inline(always)]
    pub fn add(&self, other: &Self) -> Self {
        Self::new(self.e + other.e)
    }

    #[inline(always)]
    pub fn sub(&self, other: &Self) -> Self {
        Self::new(self.e - other.e)
    }

    #[inline(always)]
    pub fn dbl(&self) -> Self {
        Self::new(self.e.double())
    }

    #[inline(always)]
    pub fn tpl(&self) -> Self {
        Self::new(self.e.double() + self.e)
    }

    #[inline(always)]
    pub fn mul(&self, other: &Self) -> Self {
        Self::new(self.e * other.e)
    }

    #[inline(always)]
    pub fn square(&self) -> Self {
        Self::new(self.e.square())
    }

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
                rd = in(reg) C::limbs_mut_ptr(&mut c.e),
                rs1 = in(reg) C::limbs_ptr(&self.e),
                rs2 = in(reg) C::limbs_ptr(&other.e),
                options(nostack)
            );
        }
        let tmp = other.mul(&c);
        if C::is_non_canonical(&C::limbs(&c.e))
            || is_not_equal(&C::limbs(&tmp.e), &C::limbs(&self.e))
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
        Self::new(self.e / other.e)
    }

    #[cfg(feature = "host")]
    #[inline(always)]
    pub fn div(&self, other: &Self) -> Self {
        if other.is_zero() {
            panic!("{}", C::div_by_zero_message());
        }
        self.div_assume_nonzero(other)
    }
}

impl GrumpkinFq {
    #[inline(always)]
    pub fn fq(&self) -> Fq {
        self.e
    }

    #[inline(always)]
    pub fn negative_seventeen() -> Self {
        Self::new(Fq::new_unchecked(BigInt([
            0xdd7056026000005a,
            0x223fa97acb319311,
            0xcc388229877910c0,
            0x34394632b724eaa,
        ])))
    }
}

impl GrumpkinFr {
    #[inline(always)]
    pub fn fr(&self) -> Fr {
        self.e
    }

    /// GLV scalar decomposition: returns (k1, k2) such that
    /// self = k1 + k2 * lambda (mod r) and |k1|, |k2| < 2^128.
    /// Each entry is (is_negative, abs_value).
    #[inline(always)]
    pub fn glv_decompose(&self) -> [(bool, u128); 2] {
        decompose_scalar_impl(self)
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
    fn generator_w_endomorphism() -> GrumpkinPoint;
    fn endomorphism(&self) -> GrumpkinPoint;
}

impl GrumpkinPointExt for GrumpkinPoint {
    #[inline(always)]
    fn generator() -> Self {
        Self::new_unchecked(
            GrumpkinFq::new(ark_grumpkin::G_GENERATOR_X),
            GrumpkinFq::new(ark_grumpkin::G_GENERATOR_Y),
        )
    }

    #[inline(always)]
    fn generator_w_endomorphism() -> GrumpkinPoint {
        Self::generator().endomorphism()
    }

    #[inline(always)]
    fn endomorphism(&self) -> GrumpkinPoint {
        if self.is_infinity() {
            Self::infinity()
        } else {
            let beta = GrumpkinFq::from_u64_arr_unchecked(&GRUMPKIN_ENDO_BETA_LIMBS);
            Self::new_unchecked(self.x().mul(&beta), self.y())
        }
    }
}

#[cfg(all(
    not(feature = "host"),
    any(target_arch = "riscv32", target_arch = "riscv64")
))]
fn decompose_scalar_impl(k: &GrumpkinFr) -> [(bool, u128); 2] {
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

    let k1_sign = decode_glv_sign_word(out[0]).unwrap_or_spoil_proof();
    let k2_sign = decode_glv_sign_word(out[3]).unwrap_or_spoil_proof();
    let lambda = GrumpkinFr::from_u64_arr_unchecked(&GRUMPKIN_GLV_LAMBDA_LIMBS);
    let mut k1 = GrumpkinFr::from_u64_arr(&[out[1], out[2], 0u64, 0u64]).unwrap_or_spoil_proof();
    if k1_sign {
        k1 = k1.neg();
    }
    let mut k2 = GrumpkinFr::from_u64_arr(&[out[4], out[5], 0u64, 0u64]).unwrap_or_spoil_proof();
    if k2_sign {
        k2 = k2.neg();
    }
    let recomposed_k = k1.add(&k2.mul(&lambda));
    if recomposed_k != *k {
        spoil_proof();
    }
    [
        (k1_sign, (out[1] as u128) | ((out[2] as u128) << 64)),
        (k2_sign, (out[4] as u128) | ((out[5] as u128) << 64)),
    ]
}

#[cfg(all(
    not(feature = "host"),
    not(any(target_arch = "riscv32", target_arch = "riscv64"))
))]
fn decompose_scalar_impl(_k: &GrumpkinFr) -> [(bool, u128); 2] {
    panic!("decompose_scalar called on non-RISC-V target without host feature");
}

#[cfg(feature = "host")]
fn decompose_scalar_impl(k: &GrumpkinFr) -> [(bool, u128); 2] {
    crate::glv::decompose_scalar(k.e.into_bigint().into())
}
