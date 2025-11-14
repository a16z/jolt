//! Type conversions and wrappers between Jolt types and final-dory's arkworks backend

use crate::{
    field::JoltField,
    msm::VariableBaseMSM,
    poly::multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
    transcripts::{AppendToTranscript, Transcript},
};
use ark_bn254::{Fr, G1Affine};
use ark_ec::CurveGroup;
use ark_ff::Zero;
use dory::{
    error::DoryError,
    primitives::{
        arithmetic::{DoryRoutines, Group as DoryGroup, PairingCurve},
        poly::{MultilinearLagrange, Polynomial as DoryPolynomial},
        transcript::Transcript as DoryTranscript,
        DorySerialize,
    },
    setup::ProverSetup,
};
use num_traits::One;
use rayon::prelude::*;

pub use dory::backends::arkworks::{
    ArkDoryProof, ArkFr, ArkG1, ArkG2, ArkGT, ArkworksProverSetup, ArkworksVerifierSetup, BN254,
};

pub type JoltFieldWrapper = ArkFr;

impl AppendToTranscript for ArkGT {
    fn append_to_transcript<S: Transcript>(&self, transcript: &mut S) {
        transcript.append_serializable(self);
    }
}

impl AppendToTranscript for &ArkGT {
    fn append_to_transcript<S: Transcript>(&self, transcript: &mut S) {
        transcript.append_serializable(*self);
    }
}

impl AppendToTranscript for ArkDoryProof {
    fn append_to_transcript<S: Transcript>(&self, transcript: &mut S) {
        transcript.append_serializable(self);
    }
}

impl AppendToTranscript for &ArkDoryProof {
    fn append_to_transcript<S: Transcript>(&self, transcript: &mut S) {
        transcript.append_serializable(*self);
    }
}

#[inline]
pub fn jolt_to_ark(f: &Fr) -> ArkFr {
    // SAFETY: ArkFr and Fr have the same memory layout
    unsafe { std::mem::transmute_copy(f) }
}

#[inline]
pub fn ark_to_jolt(ark: &ArkFr) -> Fr {
    // SAFETY: ArkFr and Fr have the same memory layout
    unsafe { std::mem::transmute_copy(ark) }
}

impl DoryPolynomial<ArkFr> for MultilinearPolynomial<Fr> {
    fn num_vars(&self) -> usize {
        self.get_num_vars()
    }

    fn evaluate(&self, point: &[ArkFr]) -> ArkFr {
        let native_point: Vec<<Fr as JoltField>::Challenge> = point
            .par_iter()
            .map(|p| {
                let f_val: Fr = ark_to_jolt(p);
                unsafe { std::mem::transmute_copy(&f_val) }
            })
            .collect();
        let result = PolynomialEvaluation::evaluate(self, native_point.as_slice());
        jolt_to_ark(&result)
    }

    fn commit<E, _M1>(
        &self,
        _nu: usize,
        sigma: usize,
        setup: &ProverSetup<E>,
    ) -> Result<(E::GT, Vec<E::G1>), DoryError>
    where
        E: PairingCurve,
        _M1: DoryRoutines<E::G1>,
        E::G1: DoryGroup<Scalar = ArkFr>,
    {
        let num_cols = 1 << sigma;

        // Perform an MSM per row of the polynomial's matrix representation
        let row_commitments = commit_tier_1::<E>(self, &setup.g1_vec, num_cols)?;

        let g2_bases = &setup.g2_vec[..row_commitments.len()];
        let commitment = E::multi_pair_g2_setup(&row_commitments, g2_bases);

        Ok((commitment, row_commitments))
    }
}

impl MultilinearLagrange<ArkFr> for MultilinearPolynomial<Fr> {
    fn vector_matrix_product(&self, left_vec: &[ArkFr], nu: usize, sigma: usize) -> Vec<ArkFr> {
        use crate::utils::small_scalar::SmallScalar;

        let num_cols = 1 << sigma;
        let num_rows = 1 << nu;

        let wrapped_left_side: Vec<Fr> = left_vec.iter().map(ark_to_jolt).collect();

        macro_rules! compute_vector_matrix_product {
            ($poly:expr, $field_mul_method:ident) => {
                (0..num_cols)
                    .into_par_iter()
                    .map(|col_idx| {
                        let mut sum = Fr::zero();
                        for row_idx in 0..num_rows.min(wrapped_left_side.len()) {
                            let coeff_idx = row_idx * num_cols + col_idx;
                            if coeff_idx < $poly.len() {
                                sum +=
                                    $poly[coeff_idx].$field_mul_method(wrapped_left_side[row_idx]);
                            }
                        }
                        jolt_to_ark(&sum)
                    })
                    .collect()
            };
        }

        match self {
            MultilinearPolynomial::LargeScalars(poly) => (0..num_cols)
                .into_par_iter()
                .map(|col_idx| {
                    let mut sum = Fr::zero();
                    for row_idx in 0..num_rows.min(wrapped_left_side.len()) {
                        let coeff_idx = row_idx * num_cols + col_idx;
                        if coeff_idx < poly.Z.len() {
                            sum += poly.Z[coeff_idx] * wrapped_left_side[row_idx];
                        }
                    }
                    jolt_to_ark(&sum)
                })
                .collect(),
            MultilinearPolynomial::U8Scalars(poly) => {
                compute_vector_matrix_product!(&poly.coeffs, field_mul)
            }
            MultilinearPolynomial::U16Scalars(poly) => {
                compute_vector_matrix_product!(&poly.coeffs, field_mul)
            }
            MultilinearPolynomial::U32Scalars(poly) => {
                compute_vector_matrix_product!(&poly.coeffs, field_mul)
            }
            MultilinearPolynomial::U64Scalars(poly) => {
                compute_vector_matrix_product!(&poly.coeffs, field_mul)
            }
            MultilinearPolynomial::I64Scalars(poly) => {
                compute_vector_matrix_product!(&poly.coeffs, field_mul)
            }
            MultilinearPolynomial::I128Scalars(poly) => {
                compute_vector_matrix_product!(&poly.coeffs, field_mul)
            }
            MultilinearPolynomial::U128Scalars(poly) => {
                compute_vector_matrix_product!(&poly.coeffs, field_mul)
            }
            MultilinearPolynomial::S128Scalars(poly) => {
                compute_vector_matrix_product!(&poly.coeffs, field_mul)
            }
            MultilinearPolynomial::BoolScalars(poly) => {
                compute_vector_matrix_product!(&poly.coeffs, field_mul)
            }
            MultilinearPolynomial::OneHot(poly) => {
                let mut result = vec![Fr::zero(); num_cols];
                poly.vector_matrix_product(&wrapped_left_side, Fr::one(), &mut result);
                result.into_iter().map(|v| jolt_to_ark(&v)).collect()
            }
            MultilinearPolynomial::RLC(poly) => poly
                .vector_matrix_product(&wrapped_left_side)
                .into_iter()
                .map(|v| jolt_to_ark(&v))
                .collect(),
        }
    }
}

fn commit_tier_1<E>(
    poly: &MultilinearPolynomial<Fr>,
    g1_generators: &[E::G1],
    row_len: usize,
) -> Result<Vec<E::G1>, DoryError>
where
    E: PairingCurve,
    E::G1: DoryGroup<Scalar = ArkFr>,
{
    // SAFETY: E::G1 and ArkG1 have the same memory layout when E = BN254.
    let g1_slice = unsafe {
        std::slice::from_raw_parts(g1_generators.as_ptr() as *const ArkG1, g1_generators.len())
    };

    // All polynomial types should use row_len bases (number of columns).
    // The globals are sized to be >= what any polynomial needs.
    let bases: Vec<G1Affine> = g1_slice
        .iter()
        .take(row_len)
        .map(|g| g.0.into_affine())
        .collect();

    macro_rules! compute_msm {
        ($coeffs:expr, $msm_method:ident) => {
            $coeffs
                .par_chunks(row_len)
                .map(|row| ArkG1(VariableBaseMSM::$msm_method(&bases[..row.len()], row).unwrap()))
                .collect()
        };
    }

    let result: Vec<ArkG1> = match poly {
        MultilinearPolynomial::LargeScalars(poly) => {
            compute_msm!(&poly.Z, msm_field_elements)
        }
        MultilinearPolynomial::U8Scalars(poly) => compute_msm!(&poly.coeffs, msm_u8),
        MultilinearPolynomial::U16Scalars(poly) => compute_msm!(&poly.coeffs, msm_u16),
        MultilinearPolynomial::U32Scalars(poly) => compute_msm!(&poly.coeffs, msm_u32),
        MultilinearPolynomial::U64Scalars(poly) => compute_msm!(&poly.coeffs, msm_u64),
        MultilinearPolynomial::I64Scalars(poly) => compute_msm!(&poly.coeffs, msm_i64),
        MultilinearPolynomial::I128Scalars(poly) => compute_msm!(&poly.coeffs, msm_i128),
        MultilinearPolynomial::U128Scalars(poly) => compute_msm!(&poly.coeffs, msm_u128),
        MultilinearPolynomial::S128Scalars(poly) => compute_msm!(&poly.coeffs, msm_s128),
        MultilinearPolynomial::BoolScalars(poly) => poly
            .coeffs
            .par_chunks(row_len)
            .map(|row| {
                let result = row
                    .iter()
                    .zip(&bases[..row.len()])
                    .filter_map(|(&b, base)| if b { Some(*base) } else { None })
                    .sum();
                ArkG1(result)
            })
            .collect(),
        MultilinearPolynomial::OneHot(poly) => {
            poly.commit_rows(&bases).into_iter().map(ArkG1).collect()
        }
        MultilinearPolynomial::RLC(poly) => {
            poly.commit_rows(&bases).into_iter().map(ArkG1).collect()
        }
    };

    // SAFETY: Vec<ArkG1> and Vec<E::G1> have the same memory layout when E = BN254.
    #[allow(clippy::missing_transmute_annotations)]
    unsafe {
        Ok(std::mem::transmute(result))
    }
}

/// Wrapper to bridge Jolt transcripts to Dory transcript trait
#[derive(Default)]
pub struct JoltToDoryTranscript<'a, T: Transcript> {
    transcript: Option<&'a mut T>,
}

impl<'a, T: Transcript> JoltToDoryTranscript<'a, T> {
    pub fn new(transcript: &'a mut T) -> Self {
        Self {
            transcript: Some(transcript),
        }
    }
}

impl<'a, T: Transcript> DoryTranscript for JoltToDoryTranscript<'a, T> {
    type Curve = BN254;

    fn append_bytes(&mut self, _label: &[u8], bytes: &[u8]) {
        let transcript = self
            .transcript
            .as_mut()
            .expect("Transcript not initialized");
        transcript.append_bytes(bytes);
    }

    fn append_field(&mut self, _label: &[u8], x: &ArkFr) {
        let transcript = self
            .transcript
            .as_mut()
            .expect("Transcript not initialized");
        let jolt_scalar: Fr = ark_to_jolt(x);
        transcript.append_scalar(&jolt_scalar);
    }

    fn append_group<G: DoryGroup>(&mut self, _label: &[u8], g: &G) {
        let transcript = self
            .transcript
            .as_mut()
            .expect("Transcript not initialized");

        let mut buffer = Vec::new();
        g.serialize_compressed(&mut buffer)
            .expect("DorySerialize serialization should not fail");
        transcript.append_bytes(&buffer);
    }

    fn append_serde<S: DorySerialize>(&mut self, _label: &[u8], s: &S) {
        let transcript = self
            .transcript
            .as_mut()
            .expect("Transcript not initialized");

        let mut buffer = Vec::new();
        s.serialize_compressed(&mut buffer)
            .expect("DorySerialize serialization should not fail");
        transcript.append_bytes(&buffer);
    }

    fn challenge_scalar(&mut self, _label: &[u8]) -> ArkFr {
        let transcript = self
            .transcript
            .as_mut()
            .expect("Transcript not initialized");
        jolt_to_ark(&transcript.challenge_scalar::<Fr>())
    }

    fn reset(&mut self, _domain_label: &[u8]) {
        panic!("Reset not supported for JoltToDoryTranscript")
    }
}
