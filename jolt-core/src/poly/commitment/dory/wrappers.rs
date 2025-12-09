//! Type conversions and wrappers between Jolt types and final-dory's arkworks backend
use std::io::{Read, Write};

use crate::{
    field::JoltField,
    msm::VariableBaseMSM,
    poly::multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
    transcripts::{AppendToTranscript, Transcript},
};
use ark_bn254::{Bn254, CompressedFq12, Fr, G1Affine};
use ark_ec::{pairing::CompressedPairing, CurveGroup};
use ark_ff::Zero;
use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress as ArkCompress,
    SerializationError as ArkSerializationError, Valid as ArkValid,
};
use dory::{
    error::DoryError,
    primitives::{
        arithmetic::{
            CompressedPairingCurve, DoryElement, DoryRoutines, Group as DoryGroup, PairingCurve,
        },
        poly::{MultilinearLagrange, Polynomial as DoryPolynomial},
        transcript::Transcript as DoryTranscript,
        Compress, DoryDeserialize, DorySerialize, SerializationError, Valid, Validate,
    },
    setup::ProverSetup,
    DoryProof, FirstReduceMessage, ScalarProductMessage, SecondReduceMessage, VMVMessage,
};
use num_traits::One;
use rayon::prelude::*;

pub use dory::backends::arkworks::{
    ArkDoryProof, ArkFr, ArkG1, ArkG2, ArkGT, ArkworksProverSetup, ArkworksVerifierSetup,
    BN254 as DoryBN254,
};

#[derive(Default, Clone, Debug)]
pub struct JoltBn254;

impl PairingCurve for JoltBn254 {
    type G1 = ArkG1;
    type G2 = ArkG2;
    type GT = ArkGT;

    fn pair(p: &Self::G1, q: &Self::G2) -> Self::GT {
        DoryBN254::pair(p, q)
    }
}

impl CompressedPairingCurve for JoltBn254 {
    type CompressedGT = ArkGTCompressed;

    fn multi_pair_compressed(ps: &[Self::G1], qs: &[Self::G2]) -> Self::CompressedGT {
        let res = Bn254::compressed_multi_pairing(ps.iter().map(|p| p.0), qs.iter().map(|q| q.0));
        ArkGTCompressed(res)
    }
}

pub type JoltFieldWrapper = ArkFr;

#[derive(Clone, Debug)]
pub struct CompressedArkDoryProof(pub DoryProof<ArkG1, ArkG2, ArkGTCompressed>);

// Adapted from similar implementation in ark_serde.rs in the Dory crate for ArkDoryProof.
// This is a workaround to allow the use of the CompressedArkDoryProof type in the JoltProof struct.
// For future work: consider cleaning up the implementation in Dory by making it more generic.
impl ArkValid for CompressedArkDoryProof {
    fn check(&self) -> Result<(), ArkSerializationError> {
        Ok(())
    }
}

impl CanonicalSerialize for CompressedArkDoryProof {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: ArkCompress,
    ) -> Result<(), ArkSerializationError> {
        // Serialize VMV message
        CanonicalSerialize::serialize_with_mode(&self.0.vmv_message.c, &mut writer, compress)?;
        CanonicalSerialize::serialize_with_mode(&self.0.vmv_message.d2, &mut writer, compress)?;
        CanonicalSerialize::serialize_with_mode(&self.0.vmv_message.e1, &mut writer, compress)?;

        // Serialize number of rounds
        let num_rounds = self.0.first_messages.len() as u32;
        CanonicalSerialize::serialize_with_mode(&num_rounds, &mut writer, compress)?;

        // Serialize first messages
        for msg in &self.0.first_messages {
            CanonicalSerialize::serialize_with_mode(&msg.d1_left, &mut writer, compress)?;
            CanonicalSerialize::serialize_with_mode(&msg.d1_right, &mut writer, compress)?;
            CanonicalSerialize::serialize_with_mode(&msg.d2_left, &mut writer, compress)?;
            CanonicalSerialize::serialize_with_mode(&msg.d2_right, &mut writer, compress)?;
            CanonicalSerialize::serialize_with_mode(&msg.e1_beta, &mut writer, compress)?;
            CanonicalSerialize::serialize_with_mode(&msg.e2_beta, &mut writer, compress)?;
        }

        // Serialize second messages
        for msg in &self.0.second_messages {
            CanonicalSerialize::serialize_with_mode(&msg.c_plus, &mut writer, compress)?;
            CanonicalSerialize::serialize_with_mode(&msg.c_minus, &mut writer, compress)?;
            CanonicalSerialize::serialize_with_mode(&msg.e1_plus, &mut writer, compress)?;
            CanonicalSerialize::serialize_with_mode(&msg.e1_minus, &mut writer, compress)?;
            CanonicalSerialize::serialize_with_mode(&msg.e2_plus, &mut writer, compress)?;
            CanonicalSerialize::serialize_with_mode(&msg.e2_minus, &mut writer, compress)?;
        }

        // Serialize final message
        CanonicalSerialize::serialize_with_mode(&self.0.final_message.e1, &mut writer, compress)?;
        CanonicalSerialize::serialize_with_mode(&self.0.final_message.e2, &mut writer, compress)?;

        // Serialize nu and sigma
        CanonicalSerialize::serialize_with_mode(&(self.0.nu as u32), &mut writer, compress)?;
        CanonicalSerialize::serialize_with_mode(&(self.0.sigma as u32), &mut writer, compress)?;

        Ok(())
    }

    fn serialized_size(&self, compress: ArkCompress) -> usize {
        let mut size = 0;

        // VMV message
        size += CanonicalSerialize::serialized_size(&self.0.vmv_message.c, compress);
        size += CanonicalSerialize::serialized_size(&self.0.vmv_message.d2, compress);
        size += CanonicalSerialize::serialized_size(&self.0.vmv_message.e1, compress);

        // Number of rounds
        size += 4; // u32

        // First messages
        for msg in &self.0.first_messages {
            size += CanonicalSerialize::serialized_size(&msg.d1_left, compress);
            size += CanonicalSerialize::serialized_size(&msg.d1_right, compress);
            size += CanonicalSerialize::serialized_size(&msg.d2_left, compress);
            size += CanonicalSerialize::serialized_size(&msg.d2_right, compress);
            size += CanonicalSerialize::serialized_size(&msg.e1_beta, compress);
            size += CanonicalSerialize::serialized_size(&msg.e2_beta, compress);
        }

        // Second messages
        for msg in &self.0.second_messages {
            size += CanonicalSerialize::serialized_size(&msg.c_plus, compress);
            size += CanonicalSerialize::serialized_size(&msg.c_minus, compress);
            size += CanonicalSerialize::serialized_size(&msg.e1_plus, compress);
            size += CanonicalSerialize::serialized_size(&msg.e1_minus, compress);
            size += CanonicalSerialize::serialized_size(&msg.e2_plus, compress);
            size += CanonicalSerialize::serialized_size(&msg.e2_minus, compress);
        }

        // Final message
        size += CanonicalSerialize::serialized_size(&self.0.final_message.e1, compress);
        size += CanonicalSerialize::serialized_size(&self.0.final_message.e2, compress);

        // nu and sigma
        size += 8; // 2 * u32

        size
    }
}

impl CanonicalDeserialize for CompressedArkDoryProof {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: ArkCompress,
        validate: ark_serialize::Validate,
    ) -> Result<Self, ArkSerializationError> {
        // Deserialize VMV message
        let c = CanonicalDeserialize::deserialize_with_mode(&mut reader, compress, validate)?;
        let d2 = CanonicalDeserialize::deserialize_with_mode(&mut reader, compress, validate)?;
        let e1 = CanonicalDeserialize::deserialize_with_mode(&mut reader, compress, validate)?;
        let vmv_message = VMVMessage { c, d2, e1 };

        // Deserialize number of rounds
        let num_rounds =
            <u32 as CanonicalDeserialize>::deserialize_with_mode(&mut reader, compress, validate)?
                as usize;

        // Deserialize first messages
        let mut first_messages = Vec::with_capacity(num_rounds);
        for _ in 0..num_rounds {
            let d1_left =
                CanonicalDeserialize::deserialize_with_mode(&mut reader, compress, validate)?;
            let d1_right =
                CanonicalDeserialize::deserialize_with_mode(&mut reader, compress, validate)?;
            let d2_left =
                CanonicalDeserialize::deserialize_with_mode(&mut reader, compress, validate)?;
            let d2_right =
                CanonicalDeserialize::deserialize_with_mode(&mut reader, compress, validate)?;
            let e1_beta =
                CanonicalDeserialize::deserialize_with_mode(&mut reader, compress, validate)?;
            let e2_beta =
                CanonicalDeserialize::deserialize_with_mode(&mut reader, compress, validate)?;
            first_messages.push(FirstReduceMessage {
                d1_left,
                d1_right,
                d2_left,
                d2_right,
                e1_beta,
                e2_beta,
            });
        }

        // Deserialize second messages
        let mut second_messages = Vec::with_capacity(num_rounds);
        for _ in 0..num_rounds {
            let c_plus =
                CanonicalDeserialize::deserialize_with_mode(&mut reader, compress, validate)?;
            let c_minus =
                CanonicalDeserialize::deserialize_with_mode(&mut reader, compress, validate)?;
            let e1_plus =
                CanonicalDeserialize::deserialize_with_mode(&mut reader, compress, validate)?;
            let e1_minus =
                CanonicalDeserialize::deserialize_with_mode(&mut reader, compress, validate)?;
            let e2_plus =
                CanonicalDeserialize::deserialize_with_mode(&mut reader, compress, validate)?;
            let e2_minus =
                CanonicalDeserialize::deserialize_with_mode(&mut reader, compress, validate)?;
            second_messages.push(SecondReduceMessage {
                c_plus,
                c_minus,
                e1_plus,
                e1_minus,
                e2_plus,
                e2_minus,
            });
        }

        // Deserialize final message
        let e1 = CanonicalDeserialize::deserialize_with_mode(&mut reader, compress, validate)?;
        let e2 = CanonicalDeserialize::deserialize_with_mode(&mut reader, compress, validate)?;
        let final_message = ScalarProductMessage { e1, e2 };

        // Deserialize nu and sigma
        let nu =
            <u32 as CanonicalDeserialize>::deserialize_with_mode(&mut reader, compress, validate)?
                as usize;
        let sigma =
            <u32 as CanonicalDeserialize>::deserialize_with_mode(&mut reader, compress, validate)?
                as usize;

        Ok(CompressedArkDoryProof(DoryProof {
            vmv_message,
            first_messages,
            second_messages,
            final_message,
            nu,
            sigma,
        }))
    }
}

#[derive(Default, Clone, Copy, PartialEq, Eq, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct ArkGTCompressed(pub CompressedFq12);

impl ArkGTCompressed {
    pub fn mul_compressed(lhs: Self, rhs: Self) -> Self {
        ArkGTCompressed(CompressedFq12::mul_compressed(lhs.0, rhs.0))
    }

    pub fn homomorphic_combine(elements: &[Self]) -> Self {
        let elements_vec: Vec<CompressedFq12> = elements.iter().map(|e| e.0).collect();
        let value = CompressedFq12::homomorphic_combine_pairing_values(elements_vec.as_slice());
        ArkGTCompressed(value)
    }

    pub fn pow<S: AsRef<[u64]>>(&self, exp: S) -> Self {
        let value = CompressedFq12::pow(&self.0, exp);
        ArkGTCompressed(value)
    }
}

impl DoryElement for ArkGTCompressed {}

impl Valid for ArkGTCompressed {
    fn check(&self) -> Result<(), SerializationError> {
        use ark_serialize::Valid as ArkValid;
        self.0
             .0
            .check()
            .map_err(|e| SerializationError::InvalidData(format!("{e:?}")))?;
        Ok(())
    }
}

impl DorySerialize for ArkGTCompressed {
    fn serialize_with_mode<W: Write>(
        &self,
        writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        match compress {
            Compress::Yes => self
                .0
                .serialize_compressed(writer)
                .map_err(|e| SerializationError::InvalidData(format!("{e}"))),
            Compress::No => self
                .0
                .serialize_uncompressed(writer)
                .map_err(|e| SerializationError::InvalidData(format!("{e}"))),
        }
    }

    fn serialized_size(&self, compress: dory::primitives::Compress) -> usize {
        match compress {
            Compress::Yes => self.0.compressed_size(),
            Compress::No => self.0.uncompressed_size(),
        }
    }
}

impl DoryDeserialize for ArkGTCompressed {
    fn deserialize_with_mode<R: Read>(
        reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let inner = match compress {
            Compress::Yes => ark_bn254::CompressedFq12::deserialize_compressed(reader)
                .map_err(|e| SerializationError::InvalidData(format!("{e:?}")))?,
            Compress::No => ark_bn254::CompressedFq12::deserialize_uncompressed(reader)
                .map_err(|e| SerializationError::InvalidData(format!("{e:?}")))?,
        };

        let res = Self(inner);
        if matches!(validate, Validate::Yes) {
            Valid::check(&res).map_err(|e| SerializationError::InvalidData(format!("{e:?}")))?;
        }

        Ok(res)
    }
}

impl From<ArkGTCompressed> for ArkGT {
    fn from(gt: ArkGTCompressed) -> Self {
        let decompressed = CompressedFq12::decompress_to_fq12(gt.0);
        Self(decompressed)
    }
}

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

impl AppendToTranscript for ArkGTCompressed {
    fn append_to_transcript<S: Transcript>(&self, transcript: &mut S) {
        transcript.append_serializable(self);
    }
}

impl AppendToTranscript for &ArkGTCompressed {
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

impl MultilinearPolynomial<Fr> {
    fn tier_1_commit<E: PairingCurve>(
        &self,
        _nu: usize,
        sigma: usize,
        setup: &ProverSetup<E>,
    ) -> Result<Vec<E::G1>, DoryError>
    where
        E::G1: DoryGroup<Scalar = ArkFr>,
    {
        let num_cols = 1 << sigma;

        // Perform an MSM per row of the polynomial's matrix representation
        let row_commitments = commit_tier_1::<E>(self, &setup.g1_vec, num_cols)?;

        Ok(row_commitments)
    }
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
        nu: usize,
        sigma: usize,
        setup: &ProverSetup<E>,
    ) -> Result<(E::GT, Vec<E::G1>), DoryError>
    where
        E: PairingCurve,
        _M1: DoryRoutines<E::G1>,
        E::G1: DoryGroup<Scalar = ArkFr>,
    {
        let row_commitments = self.tier_1_commit::<E>(nu, sigma, setup)?;
        let g2_bases = &setup.g2_vec[..row_commitments.len()];
        let commitment = E::multi_pair_g2_setup(&row_commitments, g2_bases);

        Ok((commitment, row_commitments))
    }

    fn commit_compressed<E, M1>(
        &self,
        nu: usize,
        sigma: usize,
        setup: &ProverSetup<E>,
    ) -> Result<(E::CompressedGT, Vec<E::G1>), DoryError>
    where
        E: CompressedPairingCurve,
        M1: DoryRoutines<E::G1>,
        E::G1: DoryGroup<Scalar = ArkFr>,
    {
        let row_commitments = self.tier_1_commit::<E>(nu, sigma, setup)?;
        let g2_bases = &setup.g2_vec[..row_commitments.len()];
        let commitment = E::multi_pair_g2_setup_compressed(&row_commitments, g2_bases);

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
    type Curve = JoltBn254;

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
