use std::borrow::Borrow;

use crate::field::JoltField;
use crate::poly::multilinear_polynomial::MultilinearPolynomial;
use crate::poly::unipoly::UniPoly;
use crate::utils::errors::ProofVerifyError;
use ark_ec::scalar_mul::variable_base::{
    msm_binary, msm_i128, msm_i64, msm_s128, msm_s64, msm_u128, msm_u16, msm_u32, msm_u64, msm_u8,
    VariableBaseMSM as ArkVariableBaseMSM,
};
use ark_ec::{CurveGroup, ScalarMul};
use ark_ff::biginteger::{S128, S64};
use rayon::prelude::*;

// A very light wrapper around Ark5.0 VariableBaseMSM
pub trait VariableBaseMSM: ArkVariableBaseMSM
where
    Self: ScalarMul, // technically implied by ArkVariableBaseMSM, but explicitly mentioned to be
    // consistent with current Jolt Msm implementation.
    Self::ScalarField: JoltField,
{
    #[tracing::instrument(skip_all)]
    fn msm<U>(bases: &[Self::MulBase], poly: &U) -> Result<Self, ProofVerifyError>
    where
        U: Borrow<MultilinearPolynomial<Self::ScalarField>> + Sync,
    {
        match poly.borrow() {
            MultilinearPolynomial::LargeScalars(poly) => {
                let scalars: &[Self::ScalarField] = poly.evals_ref();
                ArkVariableBaseMSM::msm(bases, scalars).map_err(|_bad_index| {
                    ProofVerifyError::KeyLengthError(bases.len(), scalars.len())
                })
            }

            MultilinearPolynomial::U8Scalars(poly) => (bases.len() == poly.coeffs.len())
                .then(|| {
                    let scalars = &poly.coeffs;
                    if scalars.par_iter().all(|&s| s == 0) {
                        Self::zero()
                    } else if scalars.par_iter().all(|&s| s <= 1) {
                        let bool_scalars: Vec<bool> = scalars.par_iter().map(|&s| s == 1).collect();
                        msm_binary::<Self>(bases, &bool_scalars, false)
                    } else {
                        msm_u8::<Self>(bases, scalars, false)
                    }
                })
                .ok_or(ProofVerifyError::KeyLengthError(
                    bases.len(),
                    poly.coeffs.len(),
                )),
            MultilinearPolynomial::U16Scalars(poly) => (bases.len() == poly.coeffs.len())
                .then(|| msm_u16::<Self>(bases, &poly.coeffs, false))
                .ok_or(ProofVerifyError::KeyLengthError(
                    bases.len(),
                    poly.coeffs.len(),
                )),
            MultilinearPolynomial::U32Scalars(poly) => (bases.len() == poly.coeffs.len())
                .then(|| msm_u32::<Self>(bases, &poly.coeffs, false))
                .ok_or(ProofVerifyError::KeyLengthError(
                    bases.len(),
                    poly.coeffs.len(),
                )),

            MultilinearPolynomial::U64Scalars(poly) => (bases.len() == poly.coeffs.len())
                .then(|| msm_u64::<Self>(bases, &poly.coeffs, false))
                .ok_or(ProofVerifyError::KeyLengthError(
                    bases.len(),
                    poly.coeffs.len(),
                )),

            MultilinearPolynomial::I64Scalars(poly) => (bases.len() == poly.coeffs.len())
                .then(|| msm_i64::<Self>(bases, &poly.coeffs, false))
                .ok_or(ProofVerifyError::KeyLengthError(
                    bases.len(),
                    poly.coeffs.len(),
                )),
            _ => unimplemented!("This variant of MultilinearPolynomial is not yet handled"),
        }
    }

    #[tracing::instrument(skip_all)]
    fn msm_field_elements(
        bases: &[Self::MulBase],
        scalars: &[Self::ScalarField],
    ) -> Result<Self, ProofVerifyError> {
        ArkVariableBaseMSM::msm_serial(bases, scalars)
            .map_err(|_bad_index| ProofVerifyError::KeyLengthError(bases.len(), scalars.len()))
    }

    #[tracing::instrument(skip_all)]
    fn msm_u8(bases: &[Self::MulBase], scalars: &[u8]) -> Result<Self, ProofVerifyError> {
        (bases.len() == scalars.len())
            .then(|| {
                if scalars.par_iter().all(|&s| s <= 1) {
                    let bool_scalars: Vec<bool> = scalars.par_iter().map(|&s| s == 1).collect();
                    msm_binary::<Self>(bases, &bool_scalars, true)
                } else {
                    msm_u8::<Self>(bases, scalars, true)
                }
            })
            .ok_or(ProofVerifyError::KeyLengthError(bases.len(), scalars.len()))
    }

    #[tracing::instrument(skip_all)]
    fn msm_u16(bases: &[Self::MulBase], scalars: &[u16]) -> Result<Self, ProofVerifyError> {
        (bases.len() == scalars.len())
            .then(|| msm_u16::<Self>(bases, scalars, true))
            .ok_or(ProofVerifyError::KeyLengthError(bases.len(), scalars.len()))
    }

    #[tracing::instrument(skip_all)]
    fn msm_u32(bases: &[Self::MulBase], scalars: &[u32]) -> Result<Self, ProofVerifyError> {
        (bases.len() == scalars.len())
            .then(|| msm_u32::<Self>(bases, scalars, true))
            .ok_or(ProofVerifyError::KeyLengthError(bases.len(), scalars.len()))
    }

    #[tracing::instrument(skip_all)]
    fn msm_u64(bases: &[Self::MulBase], scalars: &[u64]) -> Result<Self, ProofVerifyError> {
        (bases.len() == scalars.len())
            .then(|| msm_u64::<Self>(bases, scalars, true))
            .ok_or(ProofVerifyError::KeyLengthError(bases.len(), scalars.len()))
    }

    #[tracing::instrument(skip_all)]
    fn msm_u128(bases: &[Self::MulBase], scalars: &[u128]) -> Result<Self, ProofVerifyError> {
        (bases.len() == scalars.len())
            .then(|| msm_u128::<Self>(bases, scalars, true))
            .ok_or(ProofVerifyError::KeyLengthError(bases.len(), scalars.len()))
    }

    #[tracing::instrument(skip_all)]
    fn msm_i64(bases: &[Self::MulBase], scalars: &[i64]) -> Result<Self, ProofVerifyError> {
        (bases.len() == scalars.len())
            .then(|| msm_i64::<Self>(bases, scalars, true))
            .ok_or(ProofVerifyError::KeyLengthError(bases.len(), scalars.len()))
    }

    #[tracing::instrument(skip_all)]
    fn msm_s64(bases: &[Self::MulBase], scalars: &[S64]) -> Result<Self, ProofVerifyError> {
        (bases.len() == scalars.len())
            .then(|| msm_s64::<Self>(bases, scalars, true))
            .ok_or(ProofVerifyError::KeyLengthError(bases.len(), scalars.len()))
    }

    #[tracing::instrument(skip_all)]
    fn msm_i128(bases: &[Self::MulBase], scalars: &[i128]) -> Result<Self, ProofVerifyError> {
        (bases.len() == scalars.len())
            .then(|| msm_i128::<Self>(bases, scalars, true))
            .ok_or(ProofVerifyError::KeyLengthError(bases.len(), scalars.len()))
    }

    #[tracing::instrument(skip_all)]
    fn msm_s128(bases: &[Self::MulBase], scalars: &[S128]) -> Result<Self, ProofVerifyError> {
        (bases.len() == scalars.len())
            .then(|| msm_s128::<Self>(bases, scalars, true))
            .ok_or(ProofVerifyError::KeyLengthError(bases.len(), scalars.len()))
    }

    fn batch_msm<U>(bases: &[Self::MulBase], polys: &[U]) -> Vec<Self>
    where
        U: Borrow<MultilinearPolynomial<Self::ScalarField>> + Sync,
    {
        polys
            .par_iter()
            .map(|poly| VariableBaseMSM::msm(&bases[..poly.borrow().len()], poly).unwrap())
            .collect()
    }

    fn batch_msm_univariate(
        bases: &[Self::MulBase],
        polys: &[UniPoly<Self::ScalarField>],
    ) -> Vec<Self> {
        polys
            .par_iter()
            .map(|poly| {
                VariableBaseMSM::msm_field_elements(&bases[..poly.coeffs.len()], &poly.coeffs)
                    .unwrap()
            })
            .collect()
    }
}

// Implement VariableBaseMSM For any type G (like G1Projective) that implements the CurveGroup trait.
impl<F: JoltField, G: CurveGroup<ScalarField = F>> VariableBaseMSM for G {}
