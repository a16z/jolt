use crate::field::JoltField;
use crate::msm::VariableBaseMSM;
use crate::utils::errors::ProofVerifyError;
use allocative::Allocative;
use ark_ff::biginteger::{S128, S64};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

/// A trait for small scalars ({u/i}{8/16/32/64/128})
pub trait SmallScalar:
    Copy + Ord + Sync + CanonicalSerialize + CanonicalDeserialize + Allocative
{
    /// Performs a field multiplication. Uses `JoltField::mul_{u/i}{64/128}` under the hood.
    fn field_mul<F: JoltField>(&self, n: F) -> F;
    /// Converts a small scalar into a (potentially Montgomery form) `JoltField` type
    fn to_field<F: JoltField>(self) -> F;
    /// Fused absolute-difference then multiply by a field element: returns |self - other| * r.
    /// Implementations should choose the most efficient mul (prefer mul_u64 where possible).
    fn diff_mul_field<F: JoltField>(self, other: Self, r: F) -> F {
        if self > other {
            self.field_mul(r) - other.field_mul(r)
        } else {
            other.field_mul(r) - self.field_mul(r)
        }
    }
    /// Perform multi-scalar multiplication with this scalar type using optimized implementations
    fn msm<G: VariableBaseMSM>(
        bases: &[G::MulBase],
        scalars: &[Self],
    ) -> Result<G, ProofVerifyError>
    where
        G::ScalarField: JoltField;
}

impl SmallScalar for bool {
    #[inline]
    fn field_mul<F: JoltField>(&self, n: F) -> F {
        if *self {
            n
        } else {
            F::zero()
        }
    }
    #[inline]
    fn to_field<F: JoltField>(self) -> F {
        if self {
            F::one()
        } else {
            F::zero()
        }
    }
    #[inline]
    fn diff_mul_field<F: JoltField>(self, other: Self, r: F) -> F {
        if self ^ other {
            r
        } else {
            F::zero()
        }
    }
    #[inline]
    fn msm<G: VariableBaseMSM>(
        bases: &[G::MulBase],
        scalars: &[Self],
    ) -> Result<G, ProofVerifyError>
    where
        G::ScalarField: JoltField,
    {
        if bases.len() != scalars.len() {
            return Err(ProofVerifyError::KeyLengthError(bases.len(), scalars.len()));
        }
        Ok(ark_ec::scalar_mul::variable_base::msm_binary::<G>(
            bases, scalars, false,
        ))
    }
}

impl SmallScalar for u8 {
    #[inline]
    fn field_mul<F: JoltField>(&self, n: F) -> F {
        n.mul_u64(*self as u64)
    }
    #[inline]
    fn to_field<F: JoltField>(self) -> F {
        F::from_u8(self)
    }
    #[inline]
    fn diff_mul_field<F: JoltField>(self, other: Self, r: F) -> F {
        r.mul_u64(self.abs_diff(other) as u64)
    }
    #[inline]
    fn msm<G: VariableBaseMSM>(
        bases: &[G::MulBase],
        scalars: &[Self],
    ) -> Result<G, ProofVerifyError>
    where
        G::ScalarField: JoltField,
    {
        <G as crate::msm::VariableBaseMSM>::msm_u8(bases, scalars)
    }
}
impl SmallScalar for u16 {
    #[inline]
    fn field_mul<F: JoltField>(&self, n: F) -> F {
        n.mul_u64(*self as u64)
    }
    #[inline]
    fn to_field<F: JoltField>(self) -> F {
        F::from_u16(self)
    }
    #[inline]
    fn diff_mul_field<F: JoltField>(self, other: Self, r: F) -> F {
        r.mul_u64(self.abs_diff(other) as u64)
    }
    #[inline]
    fn msm<G: VariableBaseMSM>(
        bases: &[G::MulBase],
        scalars: &[Self],
    ) -> Result<G, ProofVerifyError>
    where
        G::ScalarField: JoltField,
    {
        <G as crate::msm::VariableBaseMSM>::msm_u16(bases, scalars)
    }
}
impl SmallScalar for u32 {
    #[inline]
    fn field_mul<F: JoltField>(&self, n: F) -> F {
        n.mul_u64(*self as u64)
    }
    #[inline]
    fn to_field<F: JoltField>(self) -> F {
        F::from_u32(self)
    }
    #[inline]
    fn diff_mul_field<F: JoltField>(self, other: Self, r: F) -> F {
        r.mul_u64(self.abs_diff(other) as u64)
    }
    #[inline]
    fn msm<G: VariableBaseMSM>(
        bases: &[G::MulBase],
        scalars: &[Self],
    ) -> Result<G, ProofVerifyError>
    where
        G::ScalarField: JoltField,
    {
        <G as crate::msm::VariableBaseMSM>::msm_u32(bases, scalars)
    }
}
impl SmallScalar for u64 {
    #[inline]
    fn field_mul<F: JoltField>(&self, n: F) -> F {
        n.mul_u64(*self)
    }
    #[inline]
    fn to_field<F: JoltField>(self) -> F {
        F::from_u64(self)
    }
    #[inline]
    fn diff_mul_field<F: JoltField>(self, other: Self, r: F) -> F {
        r.mul_u64(self.abs_diff(other))
    }
    #[inline]
    fn msm<G: VariableBaseMSM>(
        bases: &[G::MulBase],
        scalars: &[Self],
    ) -> Result<G, ProofVerifyError>
    where
        G::ScalarField: JoltField,
    {
        <G as crate::msm::VariableBaseMSM>::msm_u64(bases, scalars)
    }
}
impl SmallScalar for i64 {
    #[inline]
    fn field_mul<F: JoltField>(&self, n: F) -> F {
        if self.is_negative() {
            -n.mul_u64(self.unsigned_abs())
        } else {
            n.mul_u64(*self as u64)
        }
    }
    #[inline]
    fn to_field<F: JoltField>(self) -> F {
        F::from_i64(self)
    }
    #[inline]
    fn diff_mul_field<F: JoltField>(self, other: Self, r: F) -> F {
        r.mul_u64(self.abs_diff(other))
    }
    #[inline]
    fn msm<G: VariableBaseMSM>(
        bases: &[G::MulBase],
        scalars: &[Self],
    ) -> Result<G, ProofVerifyError>
    where
        G::ScalarField: JoltField,
    {
        <G as crate::msm::VariableBaseMSM>::msm_i64(bases, scalars)
    }
}
impl SmallScalar for u128 {
    #[inline]
    fn field_mul<F: JoltField>(&self, n: F) -> F {
        n.mul_u128(*self)
    }
    #[inline]
    fn to_field<F: JoltField>(self) -> F {
        F::from_u128(self)
    }
    #[inline]
    fn diff_mul_field<F: JoltField>(self, other: Self, r: F) -> F {
        let diff = self.abs_diff(other);
        if diff == 0 {
            F::zero()
        } else {
            r.mul_u128(diff)
        }
    }
    #[inline]
    fn msm<G: VariableBaseMSM>(
        bases: &[G::MulBase],
        scalars: &[Self],
    ) -> Result<G, ProofVerifyError>
    where
        G::ScalarField: JoltField,
    {
        <G as crate::msm::VariableBaseMSM>::msm_u128(bases, scalars)
    }
}
impl SmallScalar for i128 {
    #[inline]
    fn field_mul<F: JoltField>(&self, n: F) -> F {
        n.mul_i128(*self)
    }
    #[inline]
    fn to_field<F: JoltField>(self) -> F {
        F::from_i128(self)
    }
    #[inline]
    fn diff_mul_field<F: JoltField>(self, other: Self, r: F) -> F {
        let diff = self.abs_diff(other);
        if diff == 0 {
            F::zero()
        } else {
            r.mul_u128(diff)
        }
    }
    #[inline]
    fn msm<G: VariableBaseMSM>(
        bases: &[G::MulBase],
        scalars: &[Self],
    ) -> Result<G, ProofVerifyError>
    where
        G::ScalarField: JoltField,
    {
        <G as crate::msm::VariableBaseMSM>::msm_i128(bases, scalars)
    }
}
impl SmallScalar for S64 {
    #[inline]
    fn field_mul<F: JoltField>(&self, n: F) -> F {
        if self.is_positive {
            n.mul_u64(self.magnitude.0[0])
        } else {
            -n.mul_u64(self.magnitude.0[0])
        }
    }
    #[inline]
    fn to_field<F: JoltField>(self) -> F {
        if self.is_positive {
            F::from_u64(self.magnitude.0[0])
        } else {
            -F::from_u64(self.magnitude.0[0])
        }
    }
    #[inline]
    fn diff_mul_field<F: JoltField>(self, other: Self, r: F) -> F {
        let a = self.to_i128();
        let b = other.to_i128();
        let diff = (a - b).unsigned_abs();
        r.mul_u64(diff as u64)
    }
    #[inline]
    fn msm<G: VariableBaseMSM>(
        bases: &[G::MulBase],
        scalars: &[Self],
    ) -> Result<G, ProofVerifyError>
    where
        G::ScalarField: JoltField,
    {
        <G as crate::msm::VariableBaseMSM>::msm_s64(bases, scalars)
    }
}
impl SmallScalar for S128 {
    #[inline]
    fn field_mul<F: JoltField>(&self, n: F) -> F {
        let mag = self.magnitude_as_u128();
        if self.is_positive {
            n.mul_u128(mag)
        } else {
            -n.mul_u128(mag)
        }
    }
    #[inline]
    fn to_field<F: JoltField>(self) -> F {
        if let Some(v) = self.to_i128() {
            F::from_i128(v)
        } else {
            let mag = self.magnitude_as_u128();
            if self.is_positive {
                F::from_u128(mag)
            } else {
                -F::from_u128(mag)
            }
        }
    }
    #[inline]
    fn msm<G: VariableBaseMSM>(
        bases: &[G::MulBase],
        scalars: &[Self],
    ) -> Result<G, ProofVerifyError>
    where
        G::ScalarField: JoltField,
    {
        <G as crate::msm::VariableBaseMSM>::msm_s128(bases, scalars)
    }
}
