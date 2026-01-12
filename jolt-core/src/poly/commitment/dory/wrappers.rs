//! Type conversions and wrappers between Jolt types and final-dory's arkworks backend

use crate::{
    field::JoltField,
    msm::VariableBaseMSM,
    poly::{
        commitment::dory::{DoryGlobals, DoryLayout},
        multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
    },
    transcripts::{AppendToTranscript, Transcript},
};
use ark_bn254::Fr;
use ark_ec::CurveGroup;
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
    fn vector_matrix_product(&self, left_vec: &[ArkFr], _nu: usize, _sigma: usize) -> Vec<ArkFr> {
        let wrapped_left_side: Vec<Fr> = left_vec.iter().map(ark_to_jolt).collect();
        match self {
            // In Jolt, we always perform the Dory opening proof using an RLCPolynomial
            MultilinearPolynomial::RLC(poly) => poly
                .vector_matrix_product(&wrapped_left_side)
                .into_iter()
                .map(|v| jolt_to_ark(&v))
                .collect(),
            _ => unimplemented!("Unexpected MultilinearPolynomial variant"),
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

    let dory_layout = DoryGlobals::get_layout();

    let result: Vec<ArkG1> = match poly {
        MultilinearPolynomial::U64Scalars(poly) => match dory_layout {
            DoryLayout::CycleMajor => {
                let affine_bases: Vec<_> = g1_slice.par_iter().map(|g| g.0.into_affine()).collect();
                poly.coeffs
                    .par_chunks(row_len)
                    .map(|row| ArkG1(VariableBaseMSM::msm_u64(&affine_bases, row).unwrap()))
                    .collect()
            }
            DoryLayout::AddressMajor => {
                // If we're using address-major matrix layout, dense polynomial coefficients occupy (evenly-spaced)
                // columns of the matrix. Get G1 bases corresponding to those columns
                let cycles_per_row = DoryGlobals::address_major_cycles_per_row();
                let affine_bases: Vec<_> = g1_slice
                    .par_iter()
                    .take(row_len)
                    .step_by(row_len / cycles_per_row)
                    .map(|g| g.0.into_affine())
                    .collect();
                poly.coeffs
                    .par_chunks(cycles_per_row)
                    .map(|row| ArkG1(VariableBaseMSM::msm_u64(&affine_bases, row).unwrap()))
                    .collect()
            }
        },
        MultilinearPolynomial::I128Scalars(poly) => match dory_layout {
            DoryLayout::CycleMajor => {
                let affine_bases: Vec<_> = g1_slice.par_iter().map(|g| g.0.into_affine()).collect();
                poly.coeffs
                    .par_chunks(row_len)
                    .map(|row| ArkG1(VariableBaseMSM::msm_i128(&affine_bases, row).unwrap()))
                    .collect()
            }
            DoryLayout::AddressMajor => {
                // If we're using address-major matrix layout, dense polynomial coefficients occupy (evenly-spaced)
                // columns of the matrix. Get G1 bases corresponding to those columns
                let cycles_per_row = DoryGlobals::address_major_cycles_per_row();
                let affine_bases: Vec<_> = g1_slice
                    .par_iter()
                    .take(row_len)
                    .step_by(row_len / cycles_per_row)
                    .map(|g| g.0.into_affine())
                    .collect();
                poly.coeffs
                    .par_chunks(cycles_per_row)
                    .map(|row| ArkG1(VariableBaseMSM::msm_i128(&affine_bases, row).unwrap()))
                    .collect()
            }
        },
        // OneHot polynomials have their own commit_rows implementations
        // that respect the DoryLayout setting (CycleMajor vs AddressMajor)
        MultilinearPolynomial::OneHot(poly) => {
            let affine_bases: Vec<_> = g1_slice.par_iter().map(|g| g.0.into_affine()).collect();
            poly.commit_rows(&affine_bases)
                .into_iter()
                .map(ArkG1)
                .collect()
        }
        _ => unimplemented!("MultilinearPolynomial variant is never committed"),
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
