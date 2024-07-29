use crate::circuits::fields::{fp2::Fp2Var, fp6_3over2::Fp6Var, quadratic_extension::*};
use ark_ff::{
    fields::{fp12_2over3over2::*, Field},
    fp6_3over2::Fp6Config,
    PrimeField, QuadExtConfig,
};
use ark_r1cs_std::fields::FieldVar;
use ark_relations::r1cs::SynthesisError;

/// A degree-12 extension field constructed as the tower of a
/// quadratic extension over a cubic extension over a quadratic extension field.
/// This is the R1CS equivalent of `ark_ff::fp12_2over3over2::Fp12<P>`.
pub type Fp12Var<P, ConstraintF> = QuadExtVar<
    Fp6Var<<P as Fp12Config>::Fp6Config, ConstraintF>,
    ConstraintF,
    Fp12ConfigWrapper<P>,
>;

type Fp2Config<P> = <<P as Fp12Config>::Fp6Config as Fp6Config>::Fp2Config;

impl<P, ConstraintF> QuadExtVarConfig<Fp6Var<P::Fp6Config, ConstraintF>, ConstraintF>
    for Fp12ConfigWrapper<P>
where
    P: Fp12Config,
    ConstraintF: PrimeField,
{
    fn mul_base_field_var_by_frob_coeff(fe: &mut Fp6Var<P::Fp6Config, ConstraintF>, power: usize) {
        fe.c0 *= Self::FROBENIUS_COEFF_C1[power % Self::DEGREE_OVER_BASE_PRIME_FIELD];
        fe.c1 *= Self::FROBENIUS_COEFF_C1[power % Self::DEGREE_OVER_BASE_PRIME_FIELD];
        fe.c2 *= Self::FROBENIUS_COEFF_C1[power % Self::DEGREE_OVER_BASE_PRIME_FIELD];
    }
}

impl<P, ConstraintF> Fp12Var<P, ConstraintF>
where
    P: Fp12Config,
    ConstraintF: PrimeField,
{
    /// Multiplies by a sparse element of the form `(c0 = (c0, c1, 0), c1 = (0,
    /// d1, 0))`.
    #[inline]
    pub fn mul_by_014(
        &self,
        c0: &Fp2Var<Fp2Config<P>, ConstraintF>,
        c1: &Fp2Var<Fp2Config<P>, ConstraintF>,
        d1: &Fp2Var<Fp2Config<P>, ConstraintF>,
    ) -> Result<Self, SynthesisError> {
        let v0 = self.c0.mul_by_c0_c1_0(&c0, &c1)?;
        let v1 = self.c1.mul_by_0_c1_0(&d1)?;
        let new_c0 = Self::mul_base_field_by_nonresidue(&v1)? + &v0;

        let new_c1 = (&self.c0 + &self.c1).mul_by_c0_c1_0(&c0, &(c1 + d1))? - &v0 - &v1;
        Ok(Self::new(new_c0, new_c1))
    }

    /// Multiplies by a sparse element of the form `(c0 = (c0, 0, 0), c1 = (d0,
    /// d1, 0))`.
    #[inline]
    pub fn mul_by_034(
        &self,
        c0: &Fp2Var<Fp2Config<P>, ConstraintF>,
        d0: &Fp2Var<Fp2Config<P>, ConstraintF>,
        d1: &Fp2Var<Fp2Config<P>, ConstraintF>,
    ) -> Result<Self, SynthesisError> {
        let a0 = &self.c0.c0 * c0;
        let a1 = &self.c0.c1 * c0;
        let a2 = &self.c0.c2 * c0;
        let a = Fp6Var::new(a0, a1, a2);
        let b = self.c1.mul_by_c0_c1_0(&d0, &d1)?;

        let c0 = c0 + d0;
        let c1 = d1;
        let e = (&self.c0 + &self.c1).mul_by_c0_c1_0(&c0, &c1)?;
        let new_c1 = e - (&a + &b);
        let new_c0 = Self::mul_base_field_by_nonresidue(&b)? + &a;

        Ok(Self::new(new_c0, new_c1))
    }

    /// Squares `self` when `self` is in the cyclotomic subgroup.
    pub fn cyclotomic_square(&self) -> Result<Self, SynthesisError> {
        if characteristic_square_mod_6_is_one(Fp12::<P>::characteristic()) {
            let fp2_nr = <P::Fp6Config as Fp6Config>::NONRESIDUE;

            let z0 = &self.c0.c0;
            let z4 = &self.c0.c1;
            let z3 = &self.c0.c2;
            let z2 = &self.c1.c0;
            let z1 = &self.c1.c1;
            let z5 = &self.c1.c2;

            // t0 + t1*y = (z0 + z1*y)^2 = a^2
            let tmp = z0 * z1;
            let t0 = {
                let tmp1 = z0 + z1;
                let tmp2 = z1 * fp2_nr + z0;
                let tmp4 = &tmp * fp2_nr + &tmp;
                tmp1 * tmp2 - tmp4
            };
            let t1 = tmp.double()?;

            // t2 + t3*y = (z2 + z3*y)^2 = b^2
            let tmp = z2 * z3;
            let t2 = {
                // (z2 + &z3) * &(z2 + &(fp2_nr * &z3)) - &tmp - &(tmp * &fp2_nr);
                let tmp1 = z2 + z3;
                let tmp2 = z3 * fp2_nr + z2;
                let tmp4 = &tmp * fp2_nr + &tmp;
                tmp1 * tmp2 - tmp4
            };
            let t3 = tmp.double()?;

            // t4 + t5*y = (z4 + z5*y)^2 = c^2
            let tmp = z4 * z5;
            let t4 = {
                // (z4 + &z5) * &(z4 + &(fp2_nr * &z5)) - &tmp - &(tmp * &fp2_nr);
                let tmp1 = z4 + z5;
                let tmp2 = (z5 * fp2_nr) + z4;
                let tmp4 = (&tmp * fp2_nr) + &tmp;
                (tmp1 * tmp2) - tmp4
            };
            let t5 = tmp.double()?;

            // for A

            // z0 = 3 * t0 - 2 * z0
            let c0_c0 = (&t0 - z0).double()? + &t0;

            // z1 = 3 * t1 + 2 * z1
            let c1_c1 = (&t1 + z1).double()? + &t1;

            // for B

            // z2 = 3 * (xi * t5) + 2 * z2
            let c1_c0 = {
                let tmp = &t5 * fp2_nr;
                (z2 + &tmp).double()? + &tmp
            };

            // z3 = 3 * t4 - 2 * z3
            let c0_c2 = (&t4 - z3).double()? + &t4;

            // for C

            // z4 = 3 * t2 - 2 * z4
            let c0_c1 = (&t2 - z4).double()? + &t2;

            // z5 = 3 * t3 + 2 * z5
            let c1_c2 = (&t3 + z5).double()? + &t3;
            let c0 = Fp6Var::new(c0_c0, c0_c1, c0_c2);
            let c1 = Fp6Var::new(c1_c0, c1_c1, c1_c2);

            Ok(Self::new(c0, c1))
        } else {
            self.square()
        }
    }

    /// Like `Self::cyclotomic_exp`, but additionally uses cyclotomic squaring.
    pub fn optimized_cyclotomic_exp(
        &self,
        exponent: impl AsRef<[u64]>,
    ) -> Result<Self, SynthesisError> {
        use ark_ff::biginteger::arithmetic::find_naf;
        let mut res = Self::one();
        let self_inverse = self.unitary_inverse()?;

        let mut found_nonzero = false;
        let naf = find_naf(exponent.as_ref());

        for &value in naf.iter().rev() {
            if found_nonzero {
                res = res.cyclotomic_square()?;
            }

            if value != 0 {
                found_nonzero = true;

                if value > 0 {
                    res *= self;
                } else {
                    res *= &self_inverse;
                }
            }
        }

        Ok(res)
    }
}
