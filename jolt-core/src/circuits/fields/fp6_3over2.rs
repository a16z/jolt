use crate::circuits::fields::{cubic_extension::*, fp2::*};
use ark_ff::{
    fields::{fp6_3over2::*, Fp2},
    CubicExtConfig, PrimeField,
};
use ark_relations::r1cs::SynthesisError;
use ark_std::ops::MulAssign;

/// A sextic extension field constructed as the tower of a
/// cubic extension over a quadratic extension field.
/// This is the R1CS equivalent of `ark_ff::fp6_3over3::Fp6<P>`.
pub type Fp6Var<P, ConstraintF> =
    CubicExtVar<Fp2Var<<P as Fp6Config>::Fp2Config, ConstraintF>, ConstraintF, Fp6ConfigWrapper<P>>;

impl<P, ConstraintF> CubicExtVarConfig<Fp2Var<P::Fp2Config, ConstraintF>, ConstraintF>
    for Fp6ConfigWrapper<P>
where
    P: Fp6Config,
    ConstraintF: PrimeField,
{
    fn mul_base_field_vars_by_frob_coeff(
        c1: &mut Fp2Var<P::Fp2Config, ConstraintF>,
        c2: &mut Fp2Var<P::Fp2Config, ConstraintF>,
        power: usize,
    ) {
        *c1 *= Self::FROBENIUS_COEFF_C1[power % Self::DEGREE_OVER_BASE_PRIME_FIELD];
        *c2 *= Self::FROBENIUS_COEFF_C2[power % Self::DEGREE_OVER_BASE_PRIME_FIELD];
    }
}

impl<P, ConstraintF> Fp6Var<P, ConstraintF>
where
    P: Fp6Config,
    ConstraintF: PrimeField,
{
    /// Multiplies `self` by a sparse element which has `c0 == c2 == zero`.
    pub fn mul_by_0_c1_0(
        &self,
        c1: &Fp2Var<P::Fp2Config, ConstraintF>,
    ) -> Result<Self, SynthesisError> {
        // Karatsuba multiplication
        // v0 = a0 * b0 = 0

        // v1 = a1 * b1
        let v1 = &self.c1 * c1;

        // v2 = a2 * b2 = 0

        let a1_plus_a2 = &self.c1 + &self.c2;
        let b1_plus_b2 = c1.clone();

        let a0_plus_a1 = &self.c0 + &self.c1;

        // c0 = (NONRESIDUE * ((a1 + a2)*(b1 + b2) - v1 - v2)) + v0
        //    = NONRESIDUE * ((a1 + a2) * b1 - v1)
        let c0 = &(a1_plus_a2 * &b1_plus_b2 - &v1) * P::NONRESIDUE;

        // c1 = (a0 + a1) * (b0 + b1) - v0 - v1 + NONRESIDUE * v2
        //    = (a0 + a1) * b1 - v1
        let c1 = a0_plus_a1 * c1 - &v1;
        // c2 = (a0 + a2) * (b0 + b2) - v0 - v2 + v1
        //    = v1
        let c2 = v1;
        Ok(Self::new(c0, c1, c2))
    }

    /// Multiplies `self` by a sparse element which has `c2 == zero`.
    pub fn mul_by_c0_c1_0(
        &self,
        c0: &Fp2Var<P::Fp2Config, ConstraintF>,
        c1: &Fp2Var<P::Fp2Config, ConstraintF>,
    ) -> Result<Self, SynthesisError> {
        let v0 = &self.c0 * c0;
        let v1 = &self.c1 * c1;
        // v2 = 0.

        let a1_plus_a2 = &self.c1 + &self.c2;
        let a0_plus_a1 = &self.c0 + &self.c1;
        let a0_plus_a2 = &self.c0 + &self.c2;

        let b1_plus_b2 = c1.clone();
        let b0_plus_b1 = c0 + c1;
        let b0_plus_b2 = c0.clone();

        let c0 = (&a1_plus_a2 * &b1_plus_b2 - &v1) * P::NONRESIDUE + &v0;

        let c1 = a0_plus_a1 * &b0_plus_b1 - &v0 - &v1;

        let c2 = a0_plus_a2 * &b0_plus_b2 - &v0 + &v1;

        Ok(Self::new(c0, c1, c2))
    }
}

impl<P, ConstraintF> MulAssign<Fp2<P::Fp2Config>> for Fp6Var<P, ConstraintF>
where
    P: Fp6Config,
    ConstraintF: PrimeField,
{
    fn mul_assign(&mut self, other: Fp2<P::Fp2Config>) {
        self.c0 *= other;
        self.c1 *= other;
        self.c2 *= other;
    }
}
