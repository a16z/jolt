//! Type conversions and wrappers between Jolt types and final-dory's arkworks backend

use crate::{
    msm::VariableBaseMSM,
    poly::{
        commitment::dory::{DoryContext, DoryGlobals, DoryLayout},
        multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
    },
    transcripts::Transcript,
};
use ark_bn254::Fr;
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
use rayon::prelude::*;

pub use dory::backends::arkworks::{
    ArkDoryProof, ArkFr, ArkG1, ArkG2, ArkGT, ArkworksProverSetup, ArkworksVerifierSetup, BN254,
};

pub type JoltFieldWrapper = ArkFr;

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
        // Dory uses little-endian variable order; reverse to match Jolt's big-endian order.
        // Use Fr directly (not Challenge type) to avoid type mismatch issues.
        let native_point: Vec<Fr> = point.iter().rev().map(ark_to_jolt).collect();
        let result = PolynomialEvaluation::evaluate(self, native_point.as_slice());
        jolt_to_ark(&result)
    }

    fn commit<E, Mo, _M1>(
        &self,
        _nu: usize,
        sigma: usize,
        setup: &ProverSetup<E>,
    ) -> Result<(E::GT, Vec<E::G1>, ArkFr), DoryError>
    where
        E: PairingCurve,
        Mo: dory::Mode,
        _M1: DoryRoutines<E::G1>,
        E::G1: DoryGroup<Scalar = ArkFr>,
        E::GT: DoryGroup<Scalar = ArkFr>,
    {
        let num_cols = 1 << sigma;

        let row_commitments = commit_tier_1::<E>(self, &setup.g1_vec, num_cols)?;

        let g2_bases = &setup.g2_vec[..row_commitments.len()];
        let commitment = E::multi_pair_g2_setup(&row_commitments, g2_bases);

        // In ZK mode, blind the tier-2 commitment with r_d1 * HT
        let r_d1: ArkFr = Mo::sample();
        let commitment = Mo::mask(commitment, &setup.ht, &r_d1);

        Ok((commitment, row_commitments, r_d1))
    }
}

impl MultilinearLagrange<ArkFr> for MultilinearPolynomial<Fr> {
    fn vector_matrix_product(&self, left_vec: &[ArkFr], nu: usize, sigma: usize) -> Vec<ArkFr> {
        use crate::utils::small_scalar::SmallScalar;
        use ark_ff::One;

        let num_cols = 1usize << sigma;
        let num_rows = 1usize << nu;

        let wrapped_left_side: Vec<Fr> = left_vec.iter().map(ark_to_jolt).collect();

        // Helper for dense scalar vectors stored row-major as coeffs[row*num_cols + col]
        macro_rules! vmp_row_major {
            ($coeffs:expr, $mul:expr) => {{
                let coeffs = $coeffs;
                let mut result = vec![Fr::zero(); num_cols];
                result
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(col_idx, dest)| {
                        let mut sum = Fr::zero();
                        for row_idx in 0..num_rows.min(wrapped_left_side.len()) {
                            let coeff_idx = row_idx * num_cols + col_idx;
                            if coeff_idx < coeffs.len() {
                                sum += $mul(&coeffs[coeff_idx], wrapped_left_side[row_idx]);
                            }
                        }
                        *dest = sum;
                    });
                result
                    .into_iter()
                    .map(|v| jolt_to_ark(&v))
                    .collect::<Vec<_>>()
            }};
        }

        match self {
            MultilinearPolynomial::LargeScalars(poly) => {
                let coeffs = &poly.Z;
                let mut result = vec![Fr::zero(); num_cols];
                result
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(col_idx, dest)| {
                        let mut sum = Fr::zero();
                        for row_idx in 0..num_rows.min(wrapped_left_side.len()) {
                            let coeff_idx = row_idx * num_cols + col_idx;
                            if coeff_idx < coeffs.len() {
                                sum += coeffs[coeff_idx] * wrapped_left_side[row_idx];
                            }
                        }
                        *dest = sum;
                    });
                result.into_iter().map(|v| jolt_to_ark(&v)).collect()
            }
            MultilinearPolynomial::BoolScalars(poly) => {
                vmp_row_major!(&poly.coeffs, |b: &bool, l: Fr| {
                    if *b {
                        l
                    } else {
                        Fr::zero()
                    }
                })
            }
            MultilinearPolynomial::U8Scalars(poly) => {
                vmp_row_major!(&poly.coeffs, |s: &u8, l: Fr| s.field_mul(l))
            }
            MultilinearPolynomial::U16Scalars(poly) => {
                vmp_row_major!(&poly.coeffs, |s: &u16, l: Fr| s.field_mul(l))
            }
            MultilinearPolynomial::U32Scalars(poly) => {
                vmp_row_major!(&poly.coeffs, |s: &u32, l: Fr| s.field_mul(l))
            }
            MultilinearPolynomial::U64Scalars(poly) => {
                vmp_row_major!(&poly.coeffs, |s: &u64, l: Fr| s.field_mul(l))
            }
            MultilinearPolynomial::U128Scalars(poly) => {
                vmp_row_major!(&poly.coeffs, |s: &u128, l: Fr| s.field_mul(l))
            }
            MultilinearPolynomial::I64Scalars(poly) => {
                vmp_row_major!(&poly.coeffs, |s: &i64, l: Fr| s.field_mul(l))
            }
            MultilinearPolynomial::I128Scalars(poly) => {
                vmp_row_major!(&poly.coeffs, |s: &i128, l: Fr| s.field_mul(l))
            }
            MultilinearPolynomial::S128Scalars(poly) => {
                vmp_row_major!(&poly.coeffs, |s: &ark_ff::biginteger::S128, l: Fr| s
                    .field_mul(l))
            }
            MultilinearPolynomial::OneHot(poly) => {
                let mut result = vec![Fr::zero(); num_cols];
                poly.vector_matrix_product(&wrapped_left_side, Fr::one(), &mut result);
                result.into_iter().map(|v| jolt_to_ark(&v)).collect()
            }
            // In Jolt, we always perform the Dory opening proof using an RLCPolynomial
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

    let dory_context = DoryGlobals::current_context();
    let dory_layout = DoryGlobals::get_layout();

    let is_dense_poly = !matches!(
        poly,
        MultilinearPolynomial::OneHot(_) | MultilinearPolynomial::RLC(_)
    );

    let is_trace_dense_addr_major = matches!(dory_context, DoryContext::Main)
        && dory_layout == DoryLayout::AddressMajor
        && is_dense_poly;
    debug_assert!(
        !is_trace_dense_addr_major || poly.original_len() <= DoryGlobals::get_T(),
        "Main+AddressMajor dense polynomial length exceeds trace T"
    );

    let (dense_affine_bases, dense_chunk_size, dense_sparse_row_terms): (
        Vec<_>,
        usize,
        Option<Vec<Vec<(usize, Fr)>>>,
    ) = if is_trace_dense_addr_major {
        let stride = DoryGlobals::dense_stride();
        let cycles_per_row = row_len / stride;
        // This branch is taken when the AddressMajor trace-dense embedding stride exceeds
        // the post-embedded Main row width (`row_len`), i.e. `row_len < stride`.
        //
        // With:
        // - M = DoryGlobals::get_main_log_embedding() = total embedded Main vars
        // - k = log2(main K)
        // - t = log2(execution T)
        // - e = embedding extra vars = M - (k + t)
        //
        // we have:
        // - row_len = 2^sigma_main, where sigma_main = ceil(M/2)
        //          = 2^ceil((e + k + t)/2)
        // - stride  = 2^(main_embedding_extra_vars + k) = 2^(M - t) = 2^(e + k)
        //
        // so `cycles_per_row == 0` exactly when:
        //   ceil(M/2) < (M - t)   <=>   t < floor(M/2).
        if cycles_per_row == 0 {
            let dense_len = poly.original_len();
            let dense_affine_bases: Vec<_> = g1_slice
                .par_iter()
                .take(row_len)
                .map(|g| g.0.into_affine())
                .collect();
            let num_rows = DoryGlobals::get_max_num_rows();
            let sparse_terms: Vec<(usize, usize, Fr)> = (0..dense_len)
                .into_par_iter()
                .filter_map(|cycle| {
                    let coeff = poly.get_coeff(cycle);
                    if coeff.is_zero() {
                        return None;
                    }
                    let scaled_index = cycle.saturating_mul(stride);
                    let row_index = scaled_index / row_len;
                    let col_index = scaled_index % row_len;
                    debug_assert!(row_index < num_rows);
                    Some((row_index, col_index, coeff))
                })
                .collect();
            let mut row_terms: Vec<Vec<(usize, Fr)>> = vec![Vec::new(); num_rows];
            for (row_index, col_index, coeff) in sparse_terms {
                row_terms[row_index].push((col_index, coeff));
            }
            (dense_affine_bases, 1, Some(row_terms))
        } else {
            let dense_affine_bases: Vec<_> = g1_slice
                .par_iter()
                .take(row_len)
                .step_by(stride)
                .map(|g| g.0.into_affine())
                .collect();
            (dense_affine_bases, cycles_per_row, None)
        }
    } else {
        (
            g1_slice
                .par_iter()
                .take(row_len)
                .map(|g| g.0.into_affine())
                .collect(),
            row_len,
            None,
        )
    };

    if let Some(row_terms) = dense_sparse_row_terms {
        let result: Vec<ArkG1> = row_terms
            .into_par_iter()
            .map(|terms| {
                if terms.is_empty() {
                    return ArkG1(ark_bn254::G1Projective::zero());
                }
                let mut bases = Vec::with_capacity(terms.len());
                let mut scalars = Vec::with_capacity(terms.len());
                for (col_index, scalar) in terms {
                    bases.push(dense_affine_bases[col_index]);
                    scalars.push(scalar);
                }
                ArkG1(VariableBaseMSM::msm_field_elements(&bases, &scalars).unwrap())
            })
            .collect();
        // SAFETY: Vec<ArkG1> and Vec<E::G1> have the same memory layout when E = BN254.
        #[allow(clippy::missing_transmute_annotations)]
        unsafe {
            return Ok(std::mem::transmute(result));
        }
    }

    let result: Vec<ArkG1> = match poly {
        MultilinearPolynomial::LargeScalars(poly) => poly
            .Z
            .par_chunks(dense_chunk_size)
            .map(|row| {
                ArkG1(
                    VariableBaseMSM::msm_field_elements(&dense_affine_bases[..row.len()], row)
                        .unwrap(),
                )
            })
            .collect(),
        MultilinearPolynomial::BoolScalars(poly) => poly
            .coeffs
            .par_chunks(dense_chunk_size)
            .map(|row| {
                let result = row
                    .iter()
                    .zip(&dense_affine_bases[..row.len()])
                    .filter_map(|(&b, base)| if b { Some(*base) } else { None })
                    .sum();
                ArkG1(result)
            })
            .collect(),
        MultilinearPolynomial::U8Scalars(poly) => poly
            .coeffs
            .par_chunks(dense_chunk_size)
            .map(|row| {
                ArkG1(VariableBaseMSM::msm_u8(&dense_affine_bases[..row.len()], row).unwrap())
            })
            .collect(),
        MultilinearPolynomial::U16Scalars(poly) => poly
            .coeffs
            .par_chunks(dense_chunk_size)
            .map(|row| {
                ArkG1(VariableBaseMSM::msm_u16(&dense_affine_bases[..row.len()], row).unwrap())
            })
            .collect(),
        MultilinearPolynomial::U32Scalars(poly) => poly
            .coeffs
            .par_chunks(dense_chunk_size)
            .map(|row| {
                ArkG1(VariableBaseMSM::msm_u32(&dense_affine_bases[..row.len()], row).unwrap())
            })
            .collect(),
        MultilinearPolynomial::U64Scalars(poly) => poly
            .coeffs
            .par_chunks(dense_chunk_size)
            .map(|row| {
                ArkG1(VariableBaseMSM::msm_u64(&dense_affine_bases[..row.len()], row).unwrap())
            })
            .collect(),
        MultilinearPolynomial::U128Scalars(poly) => poly
            .coeffs
            .par_chunks(dense_chunk_size)
            .map(|row| {
                ArkG1(VariableBaseMSM::msm_u128(&dense_affine_bases[..row.len()], row).unwrap())
            })
            .collect(),
        MultilinearPolynomial::I64Scalars(poly) => poly
            .coeffs
            .par_chunks(dense_chunk_size)
            .map(|row| {
                ArkG1(VariableBaseMSM::msm_i64(&dense_affine_bases[..row.len()], row).unwrap())
            })
            .collect(),
        MultilinearPolynomial::I128Scalars(poly) => poly
            .coeffs
            .par_chunks(dense_chunk_size)
            .map(|row| {
                ArkG1(VariableBaseMSM::msm_i128(&dense_affine_bases[..row.len()], row).unwrap())
            })
            .collect(),
        MultilinearPolynomial::S128Scalars(poly) => poly
            .coeffs
            .par_chunks(dense_chunk_size)
            .map(|row| {
                ArkG1(VariableBaseMSM::msm_s128(&dense_affine_bases[..row.len()], row).unwrap())
            })
            .collect(),
        // OneHot polynomials have their own commit_rows implementations
        // that respect the DoryLayout setting (CycleMajor vs AddressMajor)
        MultilinearPolynomial::OneHot(poly) => {
            let affine_bases: Vec<_> = g1_slice
                .par_iter()
                .take(row_len)
                .map(|g| g.0.into_affine())
                .collect();
            poly.commit_rows(&affine_bases)
                .into_iter()
                .map(ArkG1)
                .collect()
        }
        MultilinearPolynomial::RLC(_) => {
            panic!("RLC polynomials should not be committed directly via commit_tier_1")
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
        transcript.append_bytes(b"dory_bytes", bytes);
    }

    fn append_field(&mut self, _label: &[u8], x: &ArkFr) {
        let transcript = self
            .transcript
            .as_mut()
            .expect("Transcript not initialized");
        let jolt_scalar: Fr = ark_to_jolt(x);
        transcript.append_scalar(b"dory_field", &jolt_scalar);
    }

    fn append_group<G: DoryGroup>(&mut self, _label: &[u8], g: &G) {
        let transcript = self
            .transcript
            .as_mut()
            .expect("Transcript not initialized");

        let mut buffer = Vec::new();
        g.serialize_compressed(&mut buffer)
            .expect("DorySerialize serialization should not fail");
        transcript.append_bytes(b"dory_group", &buffer);
    }

    fn append_serde<S: DorySerialize>(&mut self, _label: &[u8], s: &S) {
        let transcript = self
            .transcript
            .as_mut()
            .expect("Transcript not initialized");

        let mut buffer = Vec::new();
        s.serialize_compressed(&mut buffer)
            .expect("DorySerialize serialization should not fail");
        transcript.append_bytes(b"dory_serde", &buffer);
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
