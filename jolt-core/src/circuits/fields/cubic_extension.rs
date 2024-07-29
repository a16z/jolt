use ark_ff::{
    fields::{CubicExtField, Field},
    CubicExtConfig, PrimeField, Zero,
};
use ark_r1cs_std::{
    fields::{fp::FpVar, FieldOpsBounds, FieldVar},
    impl_bounded_ops,
    prelude::*,
    ToConstraintFieldGadget,
};
use ark_relations::r1cs::{ConstraintSystemRef, Namespace, SynthesisError};
use ark_std::vec::Vec;
use core::{borrow::Borrow, marker::PhantomData};
use derivative::Derivative;

/// This struct is the `R1CS` equivalent of the cubic extension field type
/// in `ark-ff`, i.e. `ark_ff::CubicExtField`.
#[derive(Derivative)]
#[derivative(Debug(bound = "BF: core::fmt::Debug"), Clone(bound = "BF: Clone"))]
#[must_use]
pub struct CubicExtVar<BF, ConstraintF, P>
where
    BF: FieldVar<P::BaseField, ConstraintF>,
    ConstraintF: PrimeField,
    P: CubicExtVarConfig<BF, ConstraintF>,
    for<'a> &'a BF: FieldOpsBounds<'a, P::BaseField, BF>,
{
    /// The zero-th coefficient of this field element.
    pub c0: BF,
    /// The first coefficient of this field element.
    pub c1: BF,
    /// The second coefficient of this field element.
    pub c2: BF,
    #[derivative(Debug = "ignore")]
    _params: PhantomData<(P, ConstraintF)>,
}

/// This trait describes parameters that are used to implement arithmetic for
/// `CubicExtVar`.
pub trait CubicExtVarConfig<BF, ConstraintF>: CubicExtConfig
where
    BF: FieldVar<Self::BaseField, ConstraintF>,
    ConstraintF: PrimeField,
    for<'a> &'a BF: FieldOpsBounds<'a, Self::BaseField, BF>,
{
    /// Multiply the base field of the `CubicExtVar` by the appropriate
    /// Frobenius coefficient. This is equivalent to
    /// `Self::mul_base_field_by_frob_coeff(c1, c2, power)`.
    fn mul_base_field_vars_by_frob_coeff(c1: &mut BF, c2: &mut BF, power: usize);
}

impl<BF, ConstraintF, P> CubicExtVar<BF, ConstraintF, P>
where
    BF: FieldVar<P::BaseField, ConstraintF>,
    ConstraintF: PrimeField,
    P: CubicExtVarConfig<BF, ConstraintF>,
    for<'a> &'a BF: FieldOpsBounds<'a, P::BaseField, BF>,
{
    /// Constructs a `CubicExtVar` from the underlying coefficients.
    #[inline]
    pub fn new(c0: BF, c1: BF, c2: BF) -> Self {
        let _params = PhantomData;
        Self {
            c0,
            c1,
            c2,
            _params,
        }
    }

    /// Multiplies a variable of the base field by the cubic nonresidue
    /// `P::NONRESIDUE` that is used to construct the extension field.
    #[inline]
    pub fn mul_base_field_by_nonresidue(fe: &BF) -> Result<BF, SynthesisError> {
        Ok(fe * P::NONRESIDUE)
    }

    /// Multiplies `self` by a constant from the base field.
    #[inline]
    pub fn mul_by_base_field_constant(&self, fe: P::BaseField) -> Self {
        let c0 = &self.c0 * fe;
        let c1 = &self.c1 * fe;
        let c2 = &self.c2 * fe;
        Self::new(c0, c1, c2)
    }

    /// Sets `self = self.mul_by_base_field_constant(fe)`.
    #[inline]
    pub fn mul_assign_by_base_field_constant(&mut self, fe: P::BaseField) {
        *self = (&*self).mul_by_base_field_constant(fe);
    }
}

impl<BF, ConstraintF, P> R1CSVar<ConstraintF> for CubicExtVar<BF, ConstraintF, P>
where
    BF: FieldVar<P::BaseField, ConstraintF>,
    ConstraintF: PrimeField,
    for<'a> &'a BF: FieldOpsBounds<'a, P::BaseField, BF>,
    P: CubicExtVarConfig<BF, ConstraintF>,
{
    type Value = CubicExtField<P>;

    fn cs(&self) -> ConstraintSystemRef<ConstraintF> {
        [&self.c0, &self.c1, &self.c2].cs()
    }

    #[inline]
    fn value(&self) -> Result<Self::Value, SynthesisError> {
        match (self.c0.value(), self.c1.value(), self.c2.value()) {
            (Ok(c0), Ok(c1), Ok(c2)) => Ok(CubicExtField::new(c0, c1, c2)),
            (..) => Err(SynthesisError::AssignmentMissing),
        }
    }
}

impl<BF, ConstraintF, P> From<Boolean<ConstraintF>> for CubicExtVar<BF, ConstraintF, P>
where
    BF: FieldVar<P::BaseField, ConstraintF>,
    ConstraintF: PrimeField,
    P: CubicExtVarConfig<BF, ConstraintF>,
    for<'a> &'a BF: FieldOpsBounds<'a, P::BaseField, BF>,
{
    fn from(other: Boolean<ConstraintF>) -> Self {
        let c0 = BF::from(other);
        let c1 = BF::zero();
        let c2 = BF::zero();
        Self::new(c0, c1, c2)
    }
}

impl<'a, BF, ConstraintF, P> FieldOpsBounds<'a, CubicExtField<P>, CubicExtVar<BF, ConstraintF, P>>
    for CubicExtVar<BF, ConstraintF, P>
where
    BF: FieldVar<P::BaseField, ConstraintF>,
    ConstraintF: PrimeField,
    P: CubicExtVarConfig<BF, ConstraintF>,
    for<'b> &'b BF: FieldOpsBounds<'b, P::BaseField, BF>,
{
}
impl<'a, BF, ConstraintF, P> FieldOpsBounds<'a, CubicExtField<P>, CubicExtVar<BF, ConstraintF, P>>
    for &'a CubicExtVar<BF, ConstraintF, P>
where
    BF: FieldVar<P::BaseField, ConstraintF>,
    ConstraintF: PrimeField,
    P: CubicExtVarConfig<BF, ConstraintF>,
    for<'b> &'b BF: FieldOpsBounds<'b, P::BaseField, BF>,
{
}

impl<BF, ConstraintF, P> FieldVar<CubicExtField<P>, ConstraintF> for CubicExtVar<BF, ConstraintF, P>
where
    BF: FieldVar<P::BaseField, ConstraintF>,
    ConstraintF: PrimeField,
    P: CubicExtVarConfig<BF, ConstraintF>,
    for<'a> &'a BF: FieldOpsBounds<'a, P::BaseField, BF>,
{
    fn zero() -> Self {
        let c0 = BF::zero();
        let c1 = BF::zero();
        let c2 = BF::zero();
        Self::new(c0, c1, c2)
    }

    fn one() -> Self {
        let c0 = BF::one();
        let c1 = BF::zero();
        let c2 = BF::zero();
        Self::new(c0, c1, c2)
    }

    fn constant(other: CubicExtField<P>) -> Self {
        let c0 = BF::constant(other.c0);
        let c1 = BF::constant(other.c1);
        let c2 = BF::constant(other.c2);
        Self::new(c0, c1, c2)
    }

    #[inline]
    #[tracing::instrument(target = "r1cs")]
    fn double(&self) -> Result<Self, SynthesisError> {
        let c0 = self.c0.double()?;
        let c1 = self.c1.double()?;
        let c2 = self.c2.double()?;
        Ok(Self::new(c0, c1, c2))
    }

    #[inline]
    #[tracing::instrument(target = "r1cs")]
    fn negate(&self) -> Result<Self, SynthesisError> {
        let mut result = self.clone();
        result.c0.negate_in_place()?;
        result.c1.negate_in_place()?;
        result.c2.negate_in_place()?;
        Ok(result)
    }

    /// Use the Chung-Hasan asymmetric squaring formula.
    ///
    /// (Devegili OhEig Scott Dahab --- Multiplication and Squaring on
    /// Abstract Pairing-Friendly
    /// Fields.pdf; Section 4 (CH-SQR2))
    #[inline]
    #[tracing::instrument(target = "r1cs")]
    fn square(&self) -> Result<Self, SynthesisError> {
        let a = self.c0.clone();
        let b = self.c1.clone();
        let c = self.c2.clone();

        let s0 = a.square()?;
        let ab = &a * &b;
        let s1 = ab.double()?;
        let s2 = (&a - &b + &c).square()?;
        let s3 = (&b * &c).double()?;
        let s4 = c.square()?;

        let c0 = Self::mul_base_field_by_nonresidue(&s3)? + &s0;
        let c1 = Self::mul_base_field_by_nonresidue(&s4)? + &s1;
        let c2 = s1 + &s2 + &s3 - &s0 - &s4;

        Ok(Self::new(c0, c1, c2))
    }

    #[tracing::instrument(target = "r1cs")]
    fn mul_equals(&self, other: &Self, result: &Self) -> Result<(), SynthesisError> {
        // Karatsuba multiplication for cubic extensions:
        //     v0 = A.c0 * B.c0
        //     v1 = A.c1 * B.c1
        //     v2 = A.c2 * B.c2
        //     result.c0 = v0 + β((a1 + a2)(b1 + b2) − v1 − v2)
        //     result.c1 = (a0 + a1)(b0 + b1) − v0 − v1 + βv2
        //     result.c2 = (a0 + a2)(b0 + b2) − v0 + v1 − v2,
        // We enforce this with six constraints:
        //
        //     v0 = A.c0 * B.c0
        //     v1 = A.c1 * B.c1
        //     v2 = A.c2 * B.c2
        //
        //     result.c0 - v0 + \beta*(v1 + v2) = β(a1 + a2)(b1 + b2))
        //     result.c1 + v0 + v1 - βv2        = (a0 + a1)(b0 + b1)
        //     result.c2 + v0 - v1 + v2         = (a0 + a2)(b0 + b2)
        // Reference:
        // "Multiplication and Squaring on Pairing-Friendly Fields"
        // Devegili, OhEigeartaigh, Scott, Dahab
        //
        // This implementation adapted from
        // https://github.com/ZencashOfficial/ginger-lib/blob/development/r1cs/gadgets/std/src/fields/fp3.rs
        let v0 = &self.c0 * &other.c0;
        let v1 = &self.c1 * &other.c1;
        let v2 = &self.c2 * &other.c2;

        // Check c0
        let nr_a1_plus_a2 = (&self.c1 + &self.c2) * P::NONRESIDUE;
        let b1_plus_b2 = &other.c1 + &other.c2;
        let nr_v1 = &v1 * P::NONRESIDUE;
        let nr_v2 = &v2 * P::NONRESIDUE;
        let to_check = &result.c0 - &v0 + &nr_v1 + &nr_v2;
        nr_a1_plus_a2.mul_equals(&b1_plus_b2, &to_check)?;

        // Check c1
        let a0_plus_a1 = &self.c0 + &self.c1;
        let b0_plus_b1 = &other.c0 + &other.c1;
        let to_check = &result.c1 - &nr_v2 + &v0 + &v1;
        a0_plus_a1.mul_equals(&b0_plus_b1, &to_check)?;

        // Check c2
        let a0_plus_a2 = &self.c0 + &self.c2;
        let b0_plus_b2 = &other.c0 + &other.c2;
        let to_check = &result.c2 + &v0 - &v1 + &v2;
        a0_plus_a2.mul_equals(&b0_plus_b2, &to_check)?;
        Ok(())
    }

    #[tracing::instrument(target = "r1cs")]
    fn inverse(&self) -> Result<Self, SynthesisError> {
        let mode = if self.is_constant() {
            AllocationMode::Constant
        } else {
            AllocationMode::Witness
        };
        let inverse = Self::new_variable(
            self.cs(),
            || {
                self.value()
                    .map(|f| f.inverse().unwrap_or_else(CubicExtField::zero))
            },
            mode,
        )?;
        self.mul_equals(&inverse, &Self::one())?;
        Ok(inverse)
    }

    #[tracing::instrument(target = "r1cs")]
    fn frobenius_map(&self, power: usize) -> Result<Self, SynthesisError> {
        let mut result = self.clone();
        result.c0.frobenius_map_in_place(power)?;
        result.c1.frobenius_map_in_place(power)?;
        result.c2.frobenius_map_in_place(power)?;

        P::mul_base_field_vars_by_frob_coeff(&mut result.c1, &mut result.c2, power);
        Ok(result)
    }
}

impl_bounded_ops!(
    CubicExtVar<BF, ConstraintF, P>,
    CubicExtField<P>,
    Add,
    add,
    AddAssign,
    add_assign,
    |this: &'a CubicExtVar<BF, ConstraintF, P>, other: &'a CubicExtVar<BF, ConstraintF, P>| {
        let c0 = &this.c0 + &other.c0;
        let c1 = &this.c1 + &other.c1;
        let c2 = &this.c2 + &other.c2;
        CubicExtVar::new(c0, c1, c2)
    },
    |this: &'a CubicExtVar<BF, ConstraintF, P>, other: CubicExtField<P>| {
        this + CubicExtVar::constant(other)
    },
    (BF: FieldVar<P::BaseField, ConstraintF>, ConstraintF: PrimeField, P: CubicExtVarConfig<BF, ConstraintF>),
    for<'b> &'b BF: FieldOpsBounds<'b, P::BaseField, BF>,
);
impl_bounded_ops!(
    CubicExtVar<BF, ConstraintF, P>,
    CubicExtField<P>,
    Sub,
    sub,
    SubAssign,
    sub_assign,
    |this: &'a CubicExtVar<BF, ConstraintF, P>, other: &'a CubicExtVar<BF, ConstraintF, P>| {
        let c0 = &this.c0 - &other.c0;
        let c1 = &this.c1 - &other.c1;
        let c2 = &this.c2 - &other.c2;
        CubicExtVar::new(c0, c1, c2)
    },
    |this: &'a CubicExtVar<BF, ConstraintF, P>, other: CubicExtField<P>| {
        this - CubicExtVar::constant(other)
    },
    (BF: FieldVar<P::BaseField, ConstraintF>, ConstraintF: PrimeField, P: CubicExtVarConfig<BF, ConstraintF>),
    for<'b> &'b BF: FieldOpsBounds<'b, P::BaseField, BF>,
);
impl_bounded_ops!(
    CubicExtVar<BF, ConstraintF, P>,
    CubicExtField<P>,
    Mul,
    mul,
    MulAssign,
    mul_assign,
    |this: &'a CubicExtVar<BF, ConstraintF, P>, other: &'a CubicExtVar<BF, ConstraintF, P>| {
        // Karatsuba multiplication for cubic extensions:
        //     v0 = A.c0 * B.c0
        //     v1 = A.c1 * B.c1
        //     v2 = A.c2 * B.c2
        //     result.c0 = v0 + β((a1 + a2)(b1 + b2) − v1 − v2)
        //     result.c1 = (a0 + a1)(b0 + b1) − v0 − v1 + βv2
        //     result.c2 = (a0 + a2)(b0 + b2) − v0 + v1 − v2,
        //
        // Reference:
        // "Multiplication and Squaring on Pairing-Friendly Fields"
        // Devegili, OhEigeartaigh, Scott, Dahab
        let v0 = &this.c0 * &other.c0;
        let v1 = &this.c1 * &other.c1;
        let v2 = &this.c2 * &other.c2;
        let c0 =
            (((&this.c1 + &this.c2) * (&other.c1 + &other.c2) - &v1 - &v2) * P::NONRESIDUE) + &v0 ;
        let c1 =
            (&this.c0 + &this.c1) * (&other.c0 + &other.c1) - &v0 - &v1 + (&v2 * P::NONRESIDUE);
        let c2 =
            (&this.c0 + &this.c2) * (&other.c0 + &other.c2) - &v0 + &v1 - &v2;

        CubicExtVar::new(c0, c1, c2)
    },
    |this: &'a CubicExtVar<BF, ConstraintF, P>, other: CubicExtField<P>| {
        this * CubicExtVar::constant(other)
    },
    (BF: FieldVar<P::BaseField, ConstraintF>, ConstraintF: PrimeField, P: CubicExtVarConfig<BF, ConstraintF>),
    for<'b> &'b BF: FieldOpsBounds<'b, P::BaseField, BF>,
);

impl<BF, ConstraintF, P> EqGadget<ConstraintF> for CubicExtVar<BF, ConstraintF, P>
where
    BF: FieldVar<P::BaseField, ConstraintF>,
    ConstraintF: PrimeField,
    P: CubicExtVarConfig<BF, ConstraintF>,
    for<'a> &'a BF: FieldOpsBounds<'a, P::BaseField, BF>,
{
    #[tracing::instrument(target = "r1cs")]
    fn is_eq(&self, other: &Self) -> Result<Boolean<ConstraintF>, SynthesisError> {
        let b0 = self.c0.is_eq(&other.c0)?;
        let b1 = self.c1.is_eq(&other.c1)?;
        let b2 = self.c2.is_eq(&other.c2)?;
        b0.and(&b1)?.and(&b2)
    }

    #[inline]
    #[tracing::instrument(target = "r1cs")]
    fn conditional_enforce_equal(
        &self,
        other: &Self,
        condition: &Boolean<ConstraintF>,
    ) -> Result<(), SynthesisError> {
        self.c0.conditional_enforce_equal(&other.c0, condition)?;
        self.c1.conditional_enforce_equal(&other.c1, condition)?;
        self.c2.conditional_enforce_equal(&other.c2, condition)?;
        Ok(())
    }

    #[inline]
    #[tracing::instrument(target = "r1cs")]
    fn conditional_enforce_not_equal(
        &self,
        other: &Self,
        condition: &Boolean<ConstraintF>,
    ) -> Result<(), SynthesisError> {
        let is_equal = self.is_eq(other)?;
        is_equal
            .and(condition)?
            .enforce_equal(&Boolean::Constant(false))
    }
}

impl<BF, ConstraintF, P> ToBitsGadget<ConstraintF> for CubicExtVar<BF, ConstraintF, P>
where
    BF: FieldVar<P::BaseField, ConstraintF>,
    ConstraintF: PrimeField,
    P: CubicExtVarConfig<BF, ConstraintF>,
    for<'a> &'a BF: FieldOpsBounds<'a, P::BaseField, BF>,
{
    #[tracing::instrument(target = "r1cs")]
    fn to_bits_le(&self) -> Result<Vec<Boolean<ConstraintF>>, SynthesisError> {
        let mut c0 = self.c0.to_bits_le()?;
        let mut c1 = self.c1.to_bits_le()?;
        let mut c2 = self.c2.to_bits_le()?;
        c0.append(&mut c1);
        c0.append(&mut c2);
        Ok(c0)
    }

    #[tracing::instrument(target = "r1cs")]
    fn to_non_unique_bits_le(&self) -> Result<Vec<Boolean<ConstraintF>>, SynthesisError> {
        let mut c0 = self.c0.to_non_unique_bits_le()?;
        let mut c1 = self.c1.to_non_unique_bits_le()?;
        let mut c2 = self.c2.to_non_unique_bits_le()?;
        c0.append(&mut c1);
        c0.append(&mut c2);
        Ok(c0)
    }
}

impl<BF, ConstraintF, P> ToBytesGadget<ConstraintF> for CubicExtVar<BF, ConstraintF, P>
where
    BF: FieldVar<P::BaseField, ConstraintF>,
    ConstraintF: PrimeField,
    P: CubicExtVarConfig<BF, ConstraintF>,
    for<'a> &'a BF: FieldOpsBounds<'a, P::BaseField, BF>,
{
    #[tracing::instrument(target = "r1cs")]
    fn to_bytes(&self) -> Result<Vec<UInt8<ConstraintF>>, SynthesisError> {
        let mut c0 = self.c0.to_bytes()?;
        let mut c1 = self.c1.to_bytes()?;
        let mut c2 = self.c2.to_bytes()?;
        c0.append(&mut c1);
        c0.append(&mut c2);

        Ok(c0)
    }

    #[tracing::instrument(target = "r1cs")]
    fn to_non_unique_bytes(&self) -> Result<Vec<UInt8<ConstraintF>>, SynthesisError> {
        let mut c0 = self.c0.to_non_unique_bytes()?;
        let mut c1 = self.c1.to_non_unique_bytes()?;
        let mut c2 = self.c2.to_non_unique_bytes()?;

        c0.append(&mut c1);
        c0.append(&mut c2);

        Ok(c0)
    }
}

impl<BF, ConstraintF, P> ToConstraintFieldGadget<ConstraintF> for CubicExtVar<BF, ConstraintF, P>
where
    BF: FieldVar<P::BaseField, ConstraintF>,
    ConstraintF: PrimeField,
    P: CubicExtVarConfig<BF, ConstraintF>,
    BF: ToConstraintFieldGadget<ConstraintF>,
    for<'a> &'a BF: FieldOpsBounds<'a, P::BaseField, BF>,
{
    #[tracing::instrument(target = "r1cs")]
    fn to_constraint_field(&self) -> Result<Vec<FpVar<ConstraintF>>, SynthesisError> {
        let mut res = Vec::new();

        res.extend_from_slice(&self.c0.to_constraint_field()?);
        res.extend_from_slice(&self.c1.to_constraint_field()?);
        res.extend_from_slice(&self.c2.to_constraint_field()?);

        Ok(res)
    }
}

impl<BF, ConstraintF, P> CondSelectGadget<ConstraintF> for CubicExtVar<BF, ConstraintF, P>
where
    BF: FieldVar<P::BaseField, ConstraintF>,
    ConstraintF: PrimeField,
    P: CubicExtVarConfig<BF, ConstraintF>,
    for<'a> &'a BF: FieldOpsBounds<'a, P::BaseField, BF>,
{
    #[inline]
    #[tracing::instrument(target = "r1cs")]
    fn conditionally_select(
        cond: &Boolean<ConstraintF>,
        true_value: &Self,
        false_value: &Self,
    ) -> Result<Self, SynthesisError> {
        let c0 = BF::conditionally_select(cond, &true_value.c0, &false_value.c0)?;
        let c1 = BF::conditionally_select(cond, &true_value.c1, &false_value.c1)?;
        let c2 = BF::conditionally_select(cond, &true_value.c2, &false_value.c2)?;
        Ok(Self::new(c0, c1, c2))
    }
}

impl<BF, ConstraintF, P> TwoBitLookupGadget<ConstraintF> for CubicExtVar<BF, ConstraintF, P>
where
    BF: FieldVar<P::BaseField, ConstraintF>
        + TwoBitLookupGadget<ConstraintF, TableConstant = P::BaseField>,
    ConstraintF: PrimeField,
    P: CubicExtVarConfig<BF, ConstraintF>,
    for<'a> &'a BF: FieldOpsBounds<'a, P::BaseField, BF>,
{
    type TableConstant = CubicExtField<P>;

    #[tracing::instrument(target = "r1cs")]
    fn two_bit_lookup(
        b: &[Boolean<ConstraintF>],
        c: &[Self::TableConstant],
    ) -> Result<Self, SynthesisError> {
        let c0s = c.iter().map(|f| f.c0).collect::<Vec<_>>();
        let c1s = c.iter().map(|f| f.c1).collect::<Vec<_>>();
        let c2s = c.iter().map(|f| f.c2).collect::<Vec<_>>();
        let c0 = BF::two_bit_lookup(b, &c0s)?;
        let c1 = BF::two_bit_lookup(b, &c1s)?;
        let c2 = BF::two_bit_lookup(b, &c2s)?;
        Ok(Self::new(c0, c1, c2))
    }
}

impl<BF, ConstraintF, P> ThreeBitCondNegLookupGadget<ConstraintF>
    for CubicExtVar<BF, ConstraintF, P>
where
    BF: FieldVar<P::BaseField, ConstraintF>
        + ThreeBitCondNegLookupGadget<ConstraintF, TableConstant = P::BaseField>,
    ConstraintF: PrimeField,
    P: CubicExtVarConfig<BF, ConstraintF>,
    for<'a> &'a BF: FieldOpsBounds<'a, P::BaseField, BF>,
{
    type TableConstant = CubicExtField<P>;

    #[tracing::instrument(target = "r1cs")]
    fn three_bit_cond_neg_lookup(
        b: &[Boolean<ConstraintF>],
        b0b1: &Boolean<ConstraintF>,
        c: &[Self::TableConstant],
    ) -> Result<Self, SynthesisError> {
        let c0s = c.iter().map(|f| f.c0).collect::<Vec<_>>();
        let c1s = c.iter().map(|f| f.c1).collect::<Vec<_>>();
        let c2s = c.iter().map(|f| f.c2).collect::<Vec<_>>();
        let c0 = BF::three_bit_cond_neg_lookup(b, b0b1, &c0s)?;
        let c1 = BF::three_bit_cond_neg_lookup(b, b0b1, &c1s)?;
        let c2 = BF::three_bit_cond_neg_lookup(b, b0b1, &c2s)?;
        Ok(Self::new(c0, c1, c2))
    }
}

impl<BF, ConstraintF, P> AllocVar<CubicExtField<P>, ConstraintF> for CubicExtVar<BF, ConstraintF, P>
where
    BF: FieldVar<P::BaseField, ConstraintF>,
    ConstraintF: PrimeField,
    P: CubicExtVarConfig<BF, ConstraintF>,
    for<'a> &'a BF: FieldOpsBounds<'a, P::BaseField, BF>,
{
    fn new_variable<T: Borrow<CubicExtField<P>>>(
        cs: impl Into<Namespace<ConstraintF>>,
        f: impl FnOnce() -> Result<T, SynthesisError>,
        mode: AllocationMode,
    ) -> Result<Self, SynthesisError> {
        let ns = cs.into();
        let cs = ns.cs();

        use SynthesisError::*;
        let (c0, c1, c2) = match f() {
            Ok(fe) => (Ok(fe.borrow().c0), Ok(fe.borrow().c1), Ok(fe.borrow().c2)),
            Err(_) => (
                Err(AssignmentMissing),
                Err(AssignmentMissing),
                Err(AssignmentMissing),
            ),
        };

        let c0 = BF::new_variable(ark_relations::ns!(cs, "c0"), || c0, mode)?;
        let c1 = BF::new_variable(ark_relations::ns!(cs, "c1"), || c1, mode)?;
        let c2 = BF::new_variable(ark_relations::ns!(cs, "c2"), || c2, mode)?;
        Ok(Self::new(c0, c1, c2))
    }
}
