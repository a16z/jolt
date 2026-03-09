//! Stub `JoltCurve` implementation for symbolic transpilation.
//!
//! The transpiler never performs actual curve operations (pairings, MSM, etc.),
//! but `JoltProof` and `SumcheckInstanceProof` now require a `C: JoltCurve`
//! type parameter. This module provides a minimal stub that satisfies the trait
//! bounds without doing any cryptographic work.
//!
//! All curve operations are `unimplemented!()` since they should never be
//! called during symbolic execution of stages 1-7.

use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Read, SerializationError, Valid, Write,
};
use jolt_core::curve::{JoltCurve, JoltGroupElement};
use jolt_core::field::JoltField;
use std::ops::{Add, AddAssign, Neg, Sub, SubAssign};

/// Stub group element for symbolic execution.
///
/// Never constructed at runtime; all operations are unimplemented.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct AstGroupElement;

impl Add for AstGroupElement {
    type Output = Self;
    fn add(self, _rhs: Self) -> Self {
        unimplemented!("AstGroupElement::add called during symbolic execution")
    }
}

impl<'a> Add<&'a AstGroupElement> for AstGroupElement {
    type Output = Self;
    fn add(self, _rhs: &'a AstGroupElement) -> Self {
        unimplemented!("AstGroupElement::add called during symbolic execution")
    }
}

impl Sub for AstGroupElement {
    type Output = Self;
    fn sub(self, _rhs: Self) -> Self {
        unimplemented!("AstGroupElement::sub called during symbolic execution")
    }
}

impl<'a> Sub<&'a AstGroupElement> for AstGroupElement {
    type Output = Self;
    fn sub(self, _rhs: &'a AstGroupElement) -> Self {
        unimplemented!("AstGroupElement::sub called during symbolic execution")
    }
}

impl Neg for AstGroupElement {
    type Output = Self;
    fn neg(self) -> Self {
        unimplemented!("AstGroupElement::neg called during symbolic execution")
    }
}

impl AddAssign for AstGroupElement {
    fn add_assign(&mut self, _rhs: Self) {
        unimplemented!("AstGroupElement::add_assign called during symbolic execution")
    }
}

impl SubAssign for AstGroupElement {
    fn sub_assign(&mut self, _rhs: Self) {
        unimplemented!("AstGroupElement::sub_assign called during symbolic execution")
    }
}

impl Valid for AstGroupElement {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl CanonicalSerialize for AstGroupElement {
    fn serialize_with_mode<W: Write>(
        &self,
        _writer: W,
        _compress: ark_serialize::Compress,
    ) -> Result<(), SerializationError> {
        Ok(())
    }

    fn serialized_size(&self, _compress: ark_serialize::Compress) -> usize {
        0
    }
}

impl CanonicalDeserialize for AstGroupElement {
    fn deserialize_with_mode<R: Read>(
        _reader: R,
        _compress: ark_serialize::Compress,
        _validate: ark_serialize::Validate,
    ) -> Result<Self, SerializationError> {
        Ok(Self)
    }
}

impl JoltGroupElement for AstGroupElement {
    fn zero() -> Self {
        AstGroupElement
    }

    fn is_zero(&self) -> bool {
        true
    }

    fn double(&self) -> Self {
        unimplemented!("AstGroupElement::double called during symbolic execution")
    }

    fn scalar_mul<F: JoltField>(&self, _scalar: &F) -> Self {
        unimplemented!("AstGroupElement::scalar_mul called during symbolic execution")
    }
}

/// Stub GT element for symbolic execution.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct AstGTElement;

impl Add for AstGTElement {
    type Output = Self;
    fn add(self, _rhs: Self) -> Self {
        unimplemented!("AstGTElement::add called during symbolic execution")
    }
}

impl<'a> Add<&'a AstGTElement> for AstGTElement {
    type Output = Self;
    fn add(self, _rhs: &'a AstGTElement) -> Self {
        unimplemented!("AstGTElement::add called during symbolic execution")
    }
}

impl AddAssign for AstGTElement {
    fn add_assign(&mut self, _rhs: Self) {
        unimplemented!("AstGTElement::add_assign called during symbolic execution")
    }
}

impl Valid for AstGTElement {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl CanonicalSerialize for AstGTElement {
    fn serialize_with_mode<W: Write>(
        &self,
        _writer: W,
        _compress: ark_serialize::Compress,
    ) -> Result<(), SerializationError> {
        Ok(())
    }

    fn serialized_size(&self, _compress: ark_serialize::Compress) -> usize {
        0
    }
}

impl CanonicalDeserialize for AstGTElement {
    fn deserialize_with_mode<R: Read>(
        _reader: R,
        _compress: ark_serialize::Compress,
        _validate: ark_serialize::Validate,
    ) -> Result<Self, SerializationError> {
        Ok(Self)
    }
}

/// Stub curve for symbolic execution.
///
/// Satisfies `JoltCurve` bounds required by `JoltProof<F, C, PCS, FS>` and
/// `SumcheckInstanceProof<F, C, FS>` without performing any real curve operations.
#[derive(Clone, Debug, Default)]
pub struct AstCurve;

impl JoltCurve for AstCurve {
    type G1 = AstGroupElement;
    type G2 = AstGroupElement;
    type GT = AstGTElement;

    fn g1_generator() -> Self::G1 {
        unimplemented!("AstCurve::g1_generator called during symbolic execution")
    }

    fn g2_generator() -> Self::G2 {
        unimplemented!("AstCurve::g2_generator called during symbolic execution")
    }

    fn pairing(_g1: &Self::G1, _g2: &Self::G2) -> Self::GT {
        unimplemented!("AstCurve::pairing called during symbolic execution")
    }

    fn multi_pairing(_g1s: &[Self::G1], _g2s: &[Self::G2]) -> Self::GT {
        unimplemented!("AstCurve::multi_pairing called during symbolic execution")
    }

    fn g1_msm<F: JoltField>(_bases: &[Self::G1], _scalars: &[F]) -> Self::G1 {
        unimplemented!("AstCurve::g1_msm called during symbolic execution")
    }

    fn g2_msm<F: JoltField>(_bases: &[Self::G2], _scalars: &[F]) -> Self::G2 {
        unimplemented!("AstCurve::g2_msm called during symbolic execution")
    }

    fn random_g1<R: rand_core::RngCore>(_rng: &mut R) -> Self::G1 {
        unimplemented!("AstCurve::random_g1 called during symbolic execution")
    }
}
