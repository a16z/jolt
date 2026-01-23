//! Type conversions and wrappers between Jolt types and final-dory's arkworks backend

use crate::{
    field::JoltField,
    msm::VariableBaseMSM,
    poly::{
        commitment::dory::{DoryContext, DoryGlobals, DoryLayout},
        multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
    },
    transcripts::{AppendToTranscript, Transcript},
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

    // Dense polynomials (all scalar variants except OneHot/RLC) are committed row-wise.
    //
    // In `Main` + `AddressMajor`, we have two *representations* in this repo:
    // - **Trace-dense**: length == T (e.g., `RdInc`, `RamInc`). These are embedded into the
    //   main matrix by occupying evenly-spaced columns, so each row commitment uses
    //   `cycles_per_row` bases (one per occupied column).
    // - **Matrix-dense**: length == K*T (e.g., bytecode chunk polynomials). These occupy the
    //   full matrix and must use the full `row_len` bases.
    let is_trace_dense = match poly {
        MultilinearPolynomial::LargeScalars(p) => p.Z.len() == DoryGlobals::get_T(),
        MultilinearPolynomial::BoolScalars(p) => p.coeffs.len() == DoryGlobals::get_T(),
        MultilinearPolynomial::U8Scalars(p) => p.coeffs.len() == DoryGlobals::get_T(),
        MultilinearPolynomial::U16Scalars(p) => p.coeffs.len() == DoryGlobals::get_T(),
        MultilinearPolynomial::U32Scalars(p) => p.coeffs.len() == DoryGlobals::get_T(),
        MultilinearPolynomial::U64Scalars(p) => p.coeffs.len() == DoryGlobals::get_T(),
        MultilinearPolynomial::U128Scalars(p) => p.coeffs.len() == DoryGlobals::get_T(),
        MultilinearPolynomial::I64Scalars(p) => p.coeffs.len() == DoryGlobals::get_T(),
        MultilinearPolynomial::I128Scalars(p) => p.coeffs.len() == DoryGlobals::get_T(),
        MultilinearPolynomial::S128Scalars(p) => p.coeffs.len() == DoryGlobals::get_T(),
        MultilinearPolynomial::OneHot(_) | MultilinearPolynomial::RLC(_) => false,
    };

    // Treat ProgramImage like Main here when its context is sized to match Main's K.
    // This enables AddressMajor "trace-dense" embedding (stride-by-K columns) for the
    // committed program-image polynomial.
    let is_trace_dense_addr_major =
        matches!(dory_context, DoryContext::Main | DoryContext::ProgramImage)
            && dory_layout == DoryLayout::AddressMajor
            && is_trace_dense;

    let (dense_affine_bases, dense_chunk_size): (Vec<_>, usize) = if is_trace_dense_addr_major {
        let cycles_per_row = DoryGlobals::address_major_cycles_per_row();
        let bases: Vec<_> = g1_slice
            .par_iter()
            .take(row_len)
            .step_by(row_len / cycles_per_row)
            .map(|g| g.0.into_affine())
            .collect();
        (bases, cycles_per_row)
    } else {
        (
            g1_slice
                .par_iter()
                .take(row_len)
                .map(|g| g.0.into_affine())
                .collect(),
            row_len,
        )
    };

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
