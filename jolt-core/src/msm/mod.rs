use std::borrow::Borrow;

use crate::field::JoltField;
use crate::poly::multilinear_polynomial::MultilinearPolynomial;
use crate::poly::unipoly::UniPoly;
use crate::utils::errors::ProofVerifyError;
use ark_ec::scalar_mul::variable_base::VariableBaseMSM as ArkVariableBaseMSM;
use ark_ec::scalar_mul::variable_base::{msm_binary, msm_u16, msm_u32, msm_u64, msm_u8};
use ark_ec::{CurveGroup, ScalarMul};
use rayon::prelude::*;

// A very light wrapper around Ark5.0 VariableBaseMSM
pub trait VariableBaseMSM: ArkVariableBaseMSM
where
    Self: ScalarMul, // technically implied by ArkVariableBaseMSM, but explicitly mentioned to be
    // consistent with current Jolt Msm implementation.
    Self::ScalarField: JoltField,
{
    #[tracing::instrument(skip_all)]
    fn msm<U>(
        bases: &[Self::MulBase],
        poly: &U,
        _max_num_bits: Option<usize>,
    ) -> Result<Self, ProofVerifyError>
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
                        msm_binary::<Self>(bases, &bool_scalars)
                    } else {
                        msm_u8::<Self>(bases, scalars)
                    }
                })
                .ok_or(ProofVerifyError::KeyLengthError(
                    bases.len(),
                    poly.coeffs.len(),
                )),
            MultilinearPolynomial::U16Scalars(poly) => (bases.len() == poly.coeffs.len())
                .then(|| msm_u16::<Self>(bases, &poly.coeffs))
                .ok_or(ProofVerifyError::KeyLengthError(
                    bases.len(),
                    poly.coeffs.len(),
                )),
            MultilinearPolynomial::U32Scalars(poly) => (bases.len() == poly.coeffs.len())
                .then(|| msm_u32::<Self>(bases, &poly.coeffs))
                .ok_or(ProofVerifyError::KeyLengthError(
                    bases.len(),
                    poly.coeffs.len(),
                )),

            MultilinearPolynomial::U64Scalars(poly) => (bases.len() == poly.coeffs.len())
                .then(|| msm_u64::<Self>(bases, &poly.coeffs))
                .ok_or(ProofVerifyError::KeyLengthError(
                    bases.len(),
                    poly.coeffs.len(),
                )),

            // TODO: Check if this is the fastest way forward.
            MultilinearPolynomial::I64Scalars(poly) => {
                let scalars = &poly.coeffs;
                let (pos_scalars, pos_bases, neg_scalars, neg_bases): (
                    Vec<u64>,
                    Vec<_>,
                    Vec<u64>,
                    Vec<_>,
                ) = bases
                    .par_iter()
                    .zip(scalars.par_iter())
                    .fold(
                        || (vec![], vec![], vec![], vec![]),
                        |(mut pos_s, mut pos_b, mut neg_s, mut neg_b), (base, &scalar)| {
                            if scalar > 0 {
                                pos_s.push(scalar as u64);
                                pos_b.push(*base);
                            } else if scalar < 0 {
                                neg_s.push((-scalar) as u64);
                                neg_b.push(*base);
                            }
                            (pos_s, pos_b, neg_s, neg_b)
                        },
                    )
                    .reduce(
                        || (vec![], vec![], vec![], vec![]),
                        |(mut ps1, mut pb1, mut ns1, mut nb1), (ps2, pb2, ns2, nb2)| {
                            ps1.extend(ps2);
                            pb1.extend(pb2);
                            ns1.extend(ns2);
                            nb1.extend(nb2);
                            (ps1, pb1, ns1, nb1)
                        },
                    );
                (bases.len() == poly.coeffs.len())
                    .then(|| {
                        msm_u64::<Self>(&pos_bases, &pos_scalars)
                            - msm_u64::<Self>(&neg_bases, &neg_scalars)
                    })
                    .ok_or(ProofVerifyError::KeyLengthError(
                        bases.len(),
                        poly.coeffs.len(),
                    ))
            }
            _ => unimplemented!("This variant of MultilinearPolynomial is not yet handled"),
        }
    }

    #[tracing::instrument(skip_all)]
    fn msm_field_elements(
        bases: &[Self::MulBase],
        scalars: &[Self::ScalarField],
        _max_num_bits: Option<usize>,
    ) -> Result<Self, ProofVerifyError> {
        ArkVariableBaseMSM::msm(bases, scalars)
            .map_err(|_bad_index| ProofVerifyError::KeyLengthError(bases.len(), scalars.len()))
    }

    #[tracing::instrument(skip_all)]
    fn msm_u8(bases: &[Self::MulBase], scalars: &[u8]) -> Result<Self, ProofVerifyError> {
        (bases.len() == scalars.len())
            .then(|| {
                if scalars.par_iter().all(|&s| s <= 1) {
                    let bool_scalars: Vec<bool> = scalars.par_iter().map(|&s| s == 1).collect();
                    msm_binary::<Self>(bases, &bool_scalars)
                } else {
                    msm_u8::<Self>(bases, scalars)
                }
            })
            .ok_or(ProofVerifyError::KeyLengthError(bases.len(), scalars.len()))
    }

    #[tracing::instrument(skip_all)]
    fn msm_u16(bases: &[Self::MulBase], scalars: &[u16]) -> Result<Self, ProofVerifyError> {
        (bases.len() == scalars.len())
            .then(|| msm_u16::<Self>(bases, scalars))
            .ok_or(ProofVerifyError::KeyLengthError(bases.len(), scalars.len()))
    }

    #[tracing::instrument(skip_all)]
    fn msm_u32(bases: &[Self::MulBase], scalars: &[u32]) -> Result<Self, ProofVerifyError> {
        (bases.len() == scalars.len())
            .then(|| msm_u32::<Self>(bases, scalars))
            .ok_or(ProofVerifyError::KeyLengthError(bases.len(), scalars.len()))
    }

    #[tracing::instrument(skip_all)]
    fn msm_u64(bases: &[Self::MulBase], scalars: &[u64]) -> Result<Self, ProofVerifyError> {
        (bases.len() == scalars.len())
            .then(|| msm_u64::<Self>(bases, scalars))
            .ok_or(ProofVerifyError::KeyLengthError(bases.len(), scalars.len()))
    }

    fn batch_msm<U>(bases: &[Self::MulBase], polys: &[U]) -> Vec<Self>
    where
        U: Borrow<MultilinearPolynomial<Self::ScalarField>> + Sync,
    {
        polys
            .par_iter()
            .map(|poly| VariableBaseMSM::msm(bases, poly, None).unwrap())
            .collect()
    }

    fn batch_msm_univariate(
        bases: &[Self::MulBase],
        polys: &[UniPoly<Self::ScalarField>],
    ) -> Vec<Self> {
        polys
            .par_iter()
            .map(|poly| VariableBaseMSM::msm_field_elements(bases, &poly.coeffs, None).unwrap())
            .collect()
    }
}

// Implement VariableBaseMSM For any type G (like G1Projective) that implements the CurveGroup trait.
impl<F: JoltField, G: CurveGroup<ScalarField = F>> VariableBaseMSM for G {}
