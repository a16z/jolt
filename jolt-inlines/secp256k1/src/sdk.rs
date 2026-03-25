//! secp256k1 operations optimized for Jolt zkVM.

use ark_ff::AdditiveGroup;
#[cfg(feature = "host")]
use ark_ff::Field;
use ark_ff::{BigInt, PrimeField};
use ark_secp256k1::{Fq, Fr};
use core::marker::PhantomData;

use jolt_inlines_sdk::ec::{AffinePoint, CurveParams, ECField};

extern crate alloc;
use alloc::vec::Vec;
use serde::{Deserialize, Serialize};

#[inline(always)]
fn is_fq_non_canonical(x: &[u64; 4]) -> bool {
    x[3] == u64::MAX && x[2] == u64::MAX && x[1] == u64::MAX && x[0] >= Fq::MODULUS.0[0]
}

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

pub use jolt_inlines_sdk::{spoil_proof, UnwrapOrSpoilProof};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Secp256k1Error {
    InvalidFqElement,
    InvalidFrElement,
    NotOnCurve,
    QAtInfinity,
    ROrSZero,
    RxMismatch,
    InvalidGlvSignWord(u64),
}

#[inline(always)]
#[cfg(any(
    all(test, feature = "host"),
    target_arch = "riscv32",
    target_arch = "riscv64"
))]
pub(crate) fn decode_glv_sign_word(sign: u64) -> Result<bool, Secp256k1Error> {
    match sign {
        0 => Ok(false),
        1 => Ok(true),
        _ => Err(Secp256k1Error::InvalidGlvSignWord(sign)),
    }
}

/// Configuration for a secp256k1 field (base or scalar).
///
/// Limbs are stored in non-Montgomery form. Addition and subtraction reinterpret
/// raw limbs as Montgomery-form arkworks elements — this is correct because
/// modular add/sub is representation-independent. Multiplication and division
/// use proper Montgomery conversion via `from_limbs_to_mont` / `from_mont_to_limbs`.
pub trait Secp256k1FieldConfig: 'static {
    const MUL_FUNCT3: u32;
    const SQUARE_FUNCT3: u32;
    const DIV_FUNCT3: u32;

    fn is_non_canonical(limbs: &[u64; 4]) -> bool;
    fn invalid_element_error() -> Secp256k1Error;

    /// Reinterpret raw limbs as a Montgomery-form element (no conversion).
    /// Used for add/sub/neg/dbl where Montgomery vs non-Montgomery is irrelevant.
    fn add_raw(a: &[u64; 4], b: &[u64; 4]) -> [u64; 4];
    fn sub_raw(a: &[u64; 4], b: &[u64; 4]) -> [u64; 4];
    fn neg_raw(a: &[u64; 4]) -> [u64; 4];
    fn dbl_raw(a: &[u64; 4]) -> [u64; 4];

    /// Convert raw limbs into Montgomery form, perform multiplication, convert back.
    #[cfg(feature = "host")]
    fn mul_host(a: &[u64; 4], b: &[u64; 4]) -> [u64; 4];
    #[cfg(feature = "host")]
    fn square_host(a: &[u64; 4]) -> [u64; 4];
    #[cfg(feature = "host")]
    fn div_host(a: &[u64; 4], b: &[u64; 4]) -> [u64; 4];
}

pub enum FqConfig {}

impl Secp256k1FieldConfig for FqConfig {
    const MUL_FUNCT3: u32 = crate::SECP256K1_MULQ_FUNCT3;
    const SQUARE_FUNCT3: u32 = crate::SECP256K1_SQUAREQ_FUNCT3;
    const DIV_FUNCT3: u32 = crate::SECP256K1_DIVQ_FUNCT3;

    #[inline(always)]
    fn is_non_canonical(limbs: &[u64; 4]) -> bool {
        is_fq_non_canonical(limbs)
    }
    #[inline(always)]
    fn invalid_element_error() -> Secp256k1Error {
        Secp256k1Error::InvalidFqElement
    }
    #[inline(always)]
    fn add_raw(a: &[u64; 4], b: &[u64; 4]) -> [u64; 4] {
        (Fq::new_unchecked(BigInt(*a)) + Fq::new_unchecked(BigInt(*b)))
            .0
             .0
    }
    #[inline(always)]
    fn sub_raw(a: &[u64; 4], b: &[u64; 4]) -> [u64; 4] {
        (Fq::new_unchecked(BigInt(*a)) - Fq::new_unchecked(BigInt(*b)))
            .0
             .0
    }
    #[inline(always)]
    fn neg_raw(a: &[u64; 4]) -> [u64; 4] {
        (-Fq::new_unchecked(BigInt(*a))).0 .0
    }
    #[inline(always)]
    fn dbl_raw(a: &[u64; 4]) -> [u64; 4] {
        Fq::new_unchecked(BigInt(*a)).double().0 .0
    }
    #[cfg(feature = "host")]
    #[inline(always)]
    fn mul_host(a: &[u64; 4], b: &[u64; 4]) -> [u64; 4] {
        (Fq::new(BigInt(*a)) * Fq::new(BigInt(*b))).into_bigint().0
    }
    #[cfg(feature = "host")]
    #[inline(always)]
    fn square_host(a: &[u64; 4]) -> [u64; 4] {
        Fq::new(BigInt(*a)).square().into_bigint().0
    }
    #[cfg(feature = "host")]
    #[inline(always)]
    fn div_host(a: &[u64; 4], b: &[u64; 4]) -> [u64; 4] {
        (Fq::new(BigInt(*a)) / Fq::new(BigInt(*b))).into_bigint().0
    }
}

pub enum FrConfig {}

impl Secp256k1FieldConfig for FrConfig {
    const MUL_FUNCT3: u32 = crate::SECP256K1_MULR_FUNCT3;
    const SQUARE_FUNCT3: u32 = crate::SECP256K1_SQUARER_FUNCT3;
    const DIV_FUNCT3: u32 = crate::SECP256K1_DIVR_FUNCT3;

    #[inline(always)]
    fn is_non_canonical(limbs: &[u64; 4]) -> bool {
        is_fr_non_canonical(limbs)
    }
    #[inline(always)]
    fn invalid_element_error() -> Secp256k1Error {
        Secp256k1Error::InvalidFrElement
    }
    #[inline(always)]
    fn add_raw(a: &[u64; 4], b: &[u64; 4]) -> [u64; 4] {
        (Fr::new_unchecked(BigInt(*a)) + Fr::new_unchecked(BigInt(*b)))
            .0
             .0
    }
    #[inline(always)]
    fn sub_raw(a: &[u64; 4], b: &[u64; 4]) -> [u64; 4] {
        (Fr::new_unchecked(BigInt(*a)) - Fr::new_unchecked(BigInt(*b)))
            .0
             .0
    }
    #[inline(always)]
    fn neg_raw(a: &[u64; 4]) -> [u64; 4] {
        (-Fr::new_unchecked(BigInt(*a))).0 .0
    }
    #[inline(always)]
    fn dbl_raw(a: &[u64; 4]) -> [u64; 4] {
        Fr::new_unchecked(BigInt(*a)).double().0 .0
    }
    #[cfg(feature = "host")]
    #[inline(always)]
    fn mul_host(a: &[u64; 4], b: &[u64; 4]) -> [u64; 4] {
        (Fr::new(BigInt(*a)) * Fr::new(BigInt(*b))).into_bigint().0
    }
    #[cfg(feature = "host")]
    #[inline(always)]
    fn square_host(a: &[u64; 4]) -> [u64; 4] {
        Fr::new(BigInt(*a)).square().into_bigint().0
    }
    #[cfg(feature = "host")]
    #[inline(always)]
    fn div_host(a: &[u64; 4], b: &[u64; 4]) -> [u64; 4] {
        (Fr::new(BigInt(*a)) / Fr::new(BigInt(*b))).into_bigint().0
    }
}

pub struct Secp256k1Field<C: Secp256k1FieldConfig> {
    e: [u64; 4],
    _phantom: PhantomData<C>,
}

impl<C: Secp256k1FieldConfig> Clone for Secp256k1Field<C> {
    #[inline(always)]
    fn clone(&self) -> Self {
        Self {
            e: self.e,
            _phantom: PhantomData,
        }
    }
}

impl<C: Secp256k1FieldConfig> PartialEq for Secp256k1Field<C> {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.e == other.e
    }
}

impl<C: Secp256k1FieldConfig> core::fmt::Debug for Secp256k1Field<C> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Secp256k1Field")
            .field("e", &self.e)
            .finish()
    }
}

pub type Secp256k1Fq = Secp256k1Field<FqConfig>;
pub type Secp256k1Fr = Secp256k1Field<FrConfig>;

impl<C: Secp256k1FieldConfig> Secp256k1Field<C> {
    #[inline(always)]
    pub fn from_u64_arr(arr: &[u64; 4]) -> Result<Self, Secp256k1Error> {
        if C::is_non_canonical(arr) {
            return Err(C::invalid_element_error());
        }
        Ok(Self {
            e: *arr,
            _phantom: PhantomData,
        })
    }

    #[inline(always)]
    pub fn from_u64_arr_unchecked(arr: &[u64; 4]) -> Self {
        Self {
            e: *arr,
            _phantom: PhantomData,
        }
    }

    #[inline(always)]
    pub fn e(&self) -> [u64; 4] {
        self.e
    }

    #[inline(always)]
    pub fn zero() -> Self {
        Self {
            e: [0u64; 4],
            _phantom: PhantomData,
        }
    }

    #[inline(always)]
    pub fn is_zero(&self) -> bool {
        self.e == [0u64; 4]
    }

    #[inline(always)]
    pub fn dbl(&self) -> Self {
        Self {
            e: C::dbl_raw(&self.e),
            _phantom: PhantomData,
        }
    }

    #[inline(always)]
    pub fn tpl(&self) -> Self {
        &self.dbl() + self
    }

    #[cfg(all(
        not(feature = "host"),
        any(target_arch = "riscv32", target_arch = "riscv64")
    ))]
    #[inline(always)]
    pub fn mul(&self, other: &Self) -> Self {
        let mut e = [0u64; 4];
        // SAFETY: inline instruction writes exactly 4 u64s to `e`
        unsafe {
            core::arch::asm!(
                ".insn r {opcode}, {funct3}, {funct7}, {rd}, {rs1}, {rs2}",
                opcode = const crate::INLINE_OPCODE,
                funct3 = const C::MUL_FUNCT3,
                funct7 = const crate::SECP256K1_FUNCT7,
                rd = in(reg) e.as_mut_ptr(),
                rs1 = in(reg) self.e.as_ptr(),
                rs2 = in(reg) other.e.as_ptr(),
                options(nostack)
            );
        }
        if C::is_non_canonical(&e) {
            spoil_proof();
        }
        Self::from_u64_arr_unchecked(&e[0..4].try_into().unwrap())
    }

    #[cfg(all(
        not(feature = "host"),
        not(any(target_arch = "riscv32", target_arch = "riscv64"))
    ))]
    pub fn mul(&self, _other: &Self) -> Self {
        panic!("Secp256k1Field::mul called on non-RISC-V target without host feature");
    }

    #[cfg(feature = "host")]
    #[inline(always)]
    pub fn mul(&self, other: &Self) -> Self {
        Self {
            e: C::mul_host(&self.e, &other.e),
            _phantom: PhantomData,
        }
    }

    #[cfg(all(
        not(feature = "host"),
        any(target_arch = "riscv32", target_arch = "riscv64")
    ))]
    #[inline(always)]
    pub fn square(&self) -> Self {
        let mut e = [0u64; 4];
        // SAFETY: inline instruction writes exactly 4 u64s to `e`
        unsafe {
            core::arch::asm!(
                ".insn r {opcode}, {funct3}, {funct7}, {rd}, {rs1}, x0",
                opcode = const crate::INLINE_OPCODE,
                funct3 = const C::SQUARE_FUNCT3,
                funct7 = const crate::SECP256K1_FUNCT7,
                rd = in(reg) e.as_mut_ptr(),
                rs1 = in(reg) self.e.as_ptr(),
                options(nostack)
            );
        }
        if C::is_non_canonical(&e) {
            spoil_proof();
        }
        Self::from_u64_arr_unchecked(&e[0..4].try_into().unwrap())
    }

    #[cfg(all(
        not(feature = "host"),
        not(any(target_arch = "riscv32", target_arch = "riscv64"))
    ))]
    pub fn square(&self) -> Self {
        panic!("Secp256k1Field::square called on non-RISC-V target without host feature");
    }

    #[cfg(feature = "host")]
    #[inline(always)]
    pub fn square(&self) -> Self {
        Self {
            e: C::square_host(&self.e),
            _phantom: PhantomData,
        }
    }

    #[cfg(all(
        not(feature = "host"),
        any(target_arch = "riscv32", target_arch = "riscv64")
    ))]
    #[inline(always)]
    fn div_assume_nonzero(&self, other: &Self) -> Self {
        let mut e = [0u64; 4];
        // SAFETY: inline instruction writes exactly 4 u64s to `e`
        unsafe {
            core::arch::asm!(
                ".insn r {opcode}, {funct3}, {funct7}, {rd}, {rs1}, {rs2}",
                opcode = const crate::INLINE_OPCODE,
                funct3 = const C::DIV_FUNCT3,
                funct7 = const crate::SECP256K1_FUNCT7,
                rd = in(reg) e.as_mut_ptr(),
                rs1 = in(reg) self.e.as_ptr(),
                rs2 = in(reg) other.e.as_ptr(),
                options(nostack)
            );
        }
        if C::is_non_canonical(&e) {
            spoil_proof();
        }
        Self::from_u64_arr_unchecked(&e[0..4].try_into().unwrap())
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
            "Secp256k1Field::div_assume_nonzero called on non-RISC-V target without host feature"
        );
    }

    #[cfg(all(
        not(feature = "host"),
        not(any(target_arch = "riscv32", target_arch = "riscv64"))
    ))]
    pub fn div(&self, _other: &Self) -> Self {
        panic!("Secp256k1Field::div called on non-RISC-V target without host feature");
    }

    #[cfg(feature = "host")]
    #[inline(always)]
    pub fn div_assume_nonzero(&self, other: &Self) -> Self {
        Self {
            e: C::div_host(&self.e, &other.e),
            _phantom: PhantomData,
        }
    }

    #[cfg(feature = "host")]
    #[inline(always)]
    pub fn div(&self, other: &Self) -> Self {
        if other.is_zero() {
            panic!("division by zero in Secp256k1Field::div");
        }
        self.div_assume_nonzero(other)
    }
}

impl<C: Secp256k1FieldConfig> core::ops::Add<&Secp256k1Field<C>> for &Secp256k1Field<C> {
    type Output = Secp256k1Field<C>;
    #[inline(always)]
    fn add(self, rhs: &Secp256k1Field<C>) -> Secp256k1Field<C> {
        Secp256k1Field {
            e: C::add_raw(&self.e, &rhs.e),
            _phantom: PhantomData,
        }
    }
}

impl<C: Secp256k1FieldConfig> core::ops::Sub<&Secp256k1Field<C>> for &Secp256k1Field<C> {
    type Output = Secp256k1Field<C>;
    #[inline(always)]
    fn sub(self, rhs: &Secp256k1Field<C>) -> Secp256k1Field<C> {
        Secp256k1Field {
            e: C::sub_raw(&self.e, &rhs.e),
            _phantom: PhantomData,
        }
    }
}

impl<C: Secp256k1FieldConfig> core::ops::Mul<&Secp256k1Field<C>> for &Secp256k1Field<C> {
    type Output = Secp256k1Field<C>;
    #[inline(always)]
    fn mul(self, rhs: &Secp256k1Field<C>) -> Secp256k1Field<C> {
        Secp256k1Field::mul(self, rhs)
    }
}

impl<C: Secp256k1FieldConfig> core::ops::Neg for &Secp256k1Field<C> {
    type Output = Secp256k1Field<C>;
    #[inline(always)]
    fn neg(self) -> Secp256k1Field<C> {
        Secp256k1Field {
            e: C::neg_raw(&self.e),
            _phantom: PhantomData,
        }
    }
}

impl<C: Secp256k1FieldConfig> core::ops::Add<&Secp256k1Field<C>> for Secp256k1Field<C> {
    type Output = Secp256k1Field<C>;
    #[inline(always)]
    fn add(self, rhs: &Secp256k1Field<C>) -> Secp256k1Field<C> {
        &self + rhs
    }
}

impl<C: Secp256k1FieldConfig> core::ops::Sub<&Secp256k1Field<C>> for Secp256k1Field<C> {
    type Output = Secp256k1Field<C>;
    #[inline(always)]
    fn sub(self, rhs: &Secp256k1Field<C>) -> Secp256k1Field<C> {
        &self - rhs
    }
}

impl<C: Secp256k1FieldConfig> core::ops::Mul<&Secp256k1Field<C>> for Secp256k1Field<C> {
    type Output = Secp256k1Field<C>;
    #[inline(always)]
    fn mul(self, rhs: &Secp256k1Field<C>) -> Secp256k1Field<C> {
        Secp256k1Field::mul(&self, rhs)
    }
}

impl<C: Secp256k1FieldConfig> core::ops::Neg for Secp256k1Field<C> {
    type Output = Secp256k1Field<C>;
    #[inline(always)]
    fn neg(self) -> Secp256k1Field<C> {
        -&self
    }
}

// Fq-specific methods

impl Secp256k1Field<FqConfig> {
    #[inline(always)]
    pub fn seven() -> Self {
        Self {
            e: [7u64, 0u64, 0u64, 0u64],
            _phantom: PhantomData,
        }
    }
}

impl ECField for Secp256k1Fq {
    type Error = Secp256k1Error;
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

// Fr-specific methods

impl Secp256k1Field<FrConfig> {
    #[inline(always)]
    pub fn as_u128_pair(&self) -> (u128, u128) {
        let low = self.e[0] as u128 + ((self.e[1] as u128) << 64);
        let high = self.e[2] as u128 + ((self.e[3] as u128) << 64);
        (low, high)
    }

    #[inline(always)]
    pub fn glv_decompose(&self) -> [(bool, u128); 2] {
        decompose_scalar_impl(self)
    }
}

// Curve definition

#[derive(Clone)]
pub struct Secp256k1Curve;

impl CurveParams<Secp256k1Fq> for Secp256k1Curve {
    type Error = Secp256k1Error;
    fn curve_b() -> Secp256k1Fq {
        Secp256k1Fq::seven()
    }
    fn not_on_curve_error() -> Self::Error {
        Secp256k1Error::NotOnCurve
    }
}

pub type Secp256k1Point = AffinePoint<Secp256k1Fq, Secp256k1Curve>;

pub trait Secp256k1PointExt {
    fn generator() -> Secp256k1Point;
    fn generator_times_2_pow_128() -> Secp256k1Point;
    fn generator_times_2_pow_128_plus_1() -> Secp256k1Point;
    fn generator_w_endomorphism() -> Secp256k1Point;
    fn endomorphism(&self) -> Secp256k1Point;
}

impl Secp256k1PointExt for Secp256k1Point {
    #[inline(always)]
    fn generator() -> Self {
        Self::new_unchecked(
            Secp256k1Fq::from_u64_arr_unchecked(&[
                0x59F2815B16F81798,
                0x029BFCDB2DCE28D9,
                0x55A06295CE870B07,
                0x79BE667EF9DCBBAC,
            ]),
            Secp256k1Fq::from_u64_arr_unchecked(&[
                0x9C47D08FFB10D4B8,
                0xFD17B448A6855419,
                0x5DA4FBFC0E1108A8,
                0x483ADA7726A3C465,
            ]),
        )
    }

    #[inline(always)]
    fn generator_times_2_pow_128() -> Secp256k1Point {
        Self::from_u64_arr_unchecked(&[
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

    #[inline(always)]
    fn generator_times_2_pow_128_plus_1() -> Secp256k1Point {
        Self::from_u64_arr_unchecked(&[
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

    #[inline(always)]
    fn generator_w_endomorphism() -> Secp256k1Point {
        Self::from_u64_arr_unchecked(&[
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

    // lambda = 0x5363ad4cc05c30e0a5261c028812645a122e22ea20816678df02967c1b23bd72
    #[inline(always)]
    fn endomorphism(&self) -> Secp256k1Point {
        if self.is_infinity() {
            Self::infinity()
        } else {
            let beta = Secp256k1Fq::from_u64_arr_unchecked(&[
                0xc1396c28719501ee,
                0x9cf0497512f58995,
                0x6e64479eac3434e9,
                0x7ae96a2b657c0710,
            ]);
            Self::new_unchecked(self.x() * &beta, self.y())
        }
    }
}

// GLV scalar decomposition

#[cfg(all(
    not(feature = "host"),
    any(target_arch = "riscv32", target_arch = "riscv64")
))]
fn decompose_scalar_impl(k: &Secp256k1Fr) -> [(bool, u128); 2] {
    let mut out = [0u64; 6];
    // SAFETY: inline instruction writes exactly 6 u64s to `out`
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
    let k1_sign = decode_glv_sign_word(out[0]).unwrap_or_spoil_proof();
    let k2_sign = decode_glv_sign_word(out[3]).unwrap_or_spoil_proof();
    let lambda = Secp256k1Fr::from_u64_arr_unchecked(&[
        0xdf02967c1b23bd72,
        0x122e22ea20816678,
        0xa5261c028812645a,
        0x5363ad4cc05c30e0,
    ]);
    let mut k1 = Secp256k1Fr::from_u64_arr_unchecked(&[out[1], out[2], 0u64, 0u64]);
    if k1_sign {
        k1 = -k1;
    }
    let mut k2 = Secp256k1Fr::from_u64_arr_unchecked(&[out[4], out[5], 0u64, 0u64]);
    if k2_sign {
        k2 = -k2;
    }
    let recomposed_k = &k1 + &k2.mul(&lambda);
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
fn decompose_scalar_impl(_k: &Secp256k1Fr) -> [(bool, u128); 2] {
    panic!("decompose_scalar called on non-RISC-V target without host feature");
}

#[cfg(feature = "host")]
fn decompose_scalar_impl(k: &Secp256k1Fr) -> [(bool, u128); 2] {
    let k = Fr::new(BigInt(k.e)).into_bigint().into();
    crate::glv::decompose_scalar(k)
}

// ECDSA signature verification

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

#[inline(always)]
fn conditional_negate(x: Secp256k1Point, cond: bool) -> Secp256k1Point {
    if cond {
        x.neg()
    } else {
        x
    }
}

#[inline(always)]
pub fn ecdsa_verify(
    z: Secp256k1Fr,
    r: Secp256k1Fr,
    s: Secp256k1Fr,
    q: Secp256k1Point,
) -> Result<(), Secp256k1Error> {
    if q.is_infinity() {
        return Result::Err(Secp256k1Error::QAtInfinity);
    }
    if r.is_zero() || s.is_zero() {
        return Result::Err(Secp256k1Error::ROrSZero);
    }
    let u1 = z.div_assume_nonzero(&s);
    let u2 = r.div_assume_nonzero(&s);
    let decomp_u = u1.as_u128_pair();
    let decomp_v = u2.glv_decompose();
    let scalars = [decomp_u.0, decomp_u.1, decomp_v[0].1, decomp_v[1].1];
    let points = [
        conditional_negate(q.clone(), decomp_v[0].0),
        conditional_negate(q.endomorphism(), decomp_v[1].0),
    ];
    let r_claim = secp256k1_4x128_inner_scalar_mul(scalars, points);
    // x_R mod n: for secp256k1, 0 <= x_R < p and p < 2n, so at most one subtraction
    let mut rx = r_claim.x();
    if is_fr_non_canonical(&rx.e()) {
        rx = &rx - &Secp256k1Fq::from_u64_arr_unchecked(&Fr::MODULUS.0);
    }
    if rx.e() != r.e() {
        return Result::Err(Secp256k1Error::RxMismatch);
    }
    Result::Ok(())
}
