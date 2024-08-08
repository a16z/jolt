use crate::circuits::fields::quadratic_extension::*;
use ark_ff::fields::{Fp2Config, Fp2ConfigWrapper, QuadExtConfig};
use ark_ff::PrimeField;
use ark_r1cs_std::fields::fp::FpVar;
use ark_r1cs_std::fields::nonnative::NonNativeFieldVar;

/// A quadratic extension field constructed over a prime field.
/// This is the R1CS equivalent of `ark_ff::Fp2<P>`.
pub type Fp2Var<P, ConstraintF> = QuadExtVar<
    NonNativeFieldVar<<P as Fp2Config>::Fp, ConstraintF>,
    ConstraintF,
    Fp2ConfigWrapper<P>,
>;

impl<P, ConstraintF> QuadExtVarConfig<NonNativeFieldVar<P::Fp, ConstraintF>, ConstraintF>
    for Fp2ConfigWrapper<P>
where
    P: Fp2Config,
    ConstraintF: PrimeField,
{
    fn mul_base_field_var_by_frob_coeff(
        fe: &mut NonNativeFieldVar<P::Fp, ConstraintF>,
        power: usize,
    ) {
        *fe *= Self::FROBENIUS_COEFF_C1[power % Self::DEGREE_OVER_BASE_PRIME_FIELD];
    }
}
