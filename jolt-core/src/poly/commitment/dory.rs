//! Dory polynomial commitment scheme implementation for Jolt.
//!
//! This module provides a wrapper around the Dory commitment scheme,
//! adapting it to work with Jolt's types and traits.

use super::commitment_scheme::CommitmentScheme;
use crate::{
    field::JoltField,
    msm::{Icicle, VariableBaseMSM},
    poly::{
        compact_polynomial::CompactPolynomial, dense_mlpoly::DensePolynomial,
        multilinear_polynomial::MultilinearPolynomial,
    },
    utils::{errors::ProofVerifyError, transcript::Transcript},
};

use std::{borrow::Borrow, marker::PhantomData};

// Arkworks
use ark_bn254::{Bn254, Fr, G1Projective, G2Projective};
use ark_ec::{pairing::{Pairing as ArkPairing, MillerLoopOutput}, CurveGroup};
use ark_ff::{Field, One, PrimeField, UniformRand};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::rand::RngCore;
use ark_std::Zero;
use rayon::prelude::*;

// Dory
use dory::{
    arithmetic::{
        Field as DoryField, Group as DoryGroup, MultiScalarMul as DoryMultiScalarMul,
        MultilinearPolynomial as DoryMultilinearPolynomial, Pairing as DoryPairing,
    },
    commit as dory_commit, evaluate as dory_evaluate, setup as dory_setup,
    transcript::Transcript as DoryTranscript,
    verify as dory_verify, DoryProof, DoryProofBuilder, ProverSetup, VerifierSetup,
};

// ===== Field Wrapper =====

/// NewType Wrapper around JoltField to implement Dory's Field trait
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, CanonicalSerialize, CanonicalDeserialize)]
pub struct JoltFieldWrapper<F: JoltField>(pub F);

impl<F: JoltField> DoryField for JoltFieldWrapper<F> {
    fn zero() -> Self {
        JoltFieldWrapper(F::zero())
    }

    fn one() -> Self {
        JoltFieldWrapper(F::one())
    }

    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }

    fn add(&self, rhs: &Self) -> Self {
        JoltFieldWrapper(self.0 + rhs.0)
    }

    fn sub(&self, rhs: &Self) -> Self {
        JoltFieldWrapper(self.0 - rhs.0)
    }

    fn mul(&self, rhs: &Self) -> Self {
        JoltFieldWrapper(self.0 * rhs.0)
    }

    fn inv(&self) -> Option<Self> {
        self.0.inverse().map(JoltFieldWrapper)
    }

    fn random<R: RngCore>(rng: &mut R) -> Self {
        JoltFieldWrapper(F::random(rng))
    }

    fn from_u64(val: u64) -> Self {
        JoltFieldWrapper(F::from_u64(val))
    }

    fn from_i64(val: i64) -> Self {
        JoltFieldWrapper(F::from_i64(val))
    }
}

// ===== Group Wrappers =====

/// Wrapper for elliptic curve groups (G1, G2)
#[repr(transparent)]
#[derive(Debug, Clone, PartialEq, CanonicalSerialize, CanonicalDeserialize)]
pub struct JoltGroupWrapper<G: CurveGroup> {
    pub inner: G,
}

impl<G> DoryGroup for JoltGroupWrapper<G>
where
    G: CurveGroup + VariableBaseMSM,
    G::ScalarField: JoltField,
{
    type Scalar = JoltFieldWrapper<G::ScalarField>;

    fn identity() -> Self {
        Self { inner: G::zero() }
    }

    fn add(&self, rhs: &Self) -> Self {
        Self {
            inner: self.inner + rhs.inner,
        }
    }

    fn neg(&self) -> Self {
        Self { inner: -self.inner }
    }

    fn scale(&self, k: &Self::Scalar) -> Self {
        Self {
            inner: self.inner * k.0,
        }
    }

    fn random<R: RngCore>(rng: &mut R) -> Self {
        Self {
            inner: G::rand(rng),
        }
    }
}

/// Wrapper for target field group (GT)
#[repr(transparent)]
#[derive(Debug, Clone, PartialEq, CanonicalSerialize, CanonicalDeserialize)]
pub struct JoltGTWrapper<P: ArkPairing> {
    pub inner: P::TargetField,
}

impl<P> DoryGroup for JoltGTWrapper<P>
where
    P: ArkPairing,
    P::ScalarField: JoltField,
{
    type Scalar = JoltFieldWrapper<P::ScalarField>;

    fn identity() -> Self {
        Self {
            inner: P::TargetField::one(),
        }
    }

    fn add(&self, rhs: &Self) -> Self {
        Self {
            inner: self.inner * rhs.inner,
        }
    }

    fn neg(&self) -> Self {
        Self {
            inner: self
                .inner
                .inverse()
                .expect("GT element should be invertible"),
        }
    }

    fn scale(&self, k: &Self::Scalar) -> Self {
        Self {
            inner: self.inner.pow(k.0.into_bigint()),
        }
    }

    fn random<R: RngCore>(rng: &mut R) -> Self {
        Self {
            inner: P::TargetField::rand(rng),
        }
    }
}

impl<P: ArkPairing> Default for JoltGTWrapper<P> {
    fn default() -> Self {
        Self {
            inner: P::TargetField::one(),
        }
    }
}

// ===== MSM Implementations =====

/// Multi-scalar multiplication for (generic) G1, G2 + polynomial
pub struct JoltCustomMSM<G: CurveGroup> {
    _phantom: PhantomData<G>,
}

impl<G> DoryMultiScalarMul<JoltGroupWrapper<G>> for JoltCustomMSM<G>
where
    G: CurveGroup + VariableBaseMSM,
    G::ScalarField: JoltField,
{
    fn msm(
        bases: &[JoltGroupWrapper<G>],
        scalars: &DoryMultilinearPolynomial<JoltFieldWrapper<G::ScalarField>>,
    ) -> JoltGroupWrapper<G> {
        let affines: Vec<G::Affine> = bases.iter().map(|w| w.inner.into_affine()).collect();
        let ml_poly: MultilinearPolynomial<G::ScalarField> = scalars.into();

        let result = <G as VariableBaseMSM>::msm(&affines, None, &ml_poly, None)
            .expect("MSM should not fail");

        JoltGroupWrapper { inner: result }
    }
}

/// Multi-scalar multiplication for GT + polynomial
pub struct JoltGTCustomMSM<P: ArkPairing> {
    _phantom: PhantomData<P>,
}

impl<P> DoryMultiScalarMul<JoltGTWrapper<P>> for JoltGTCustomMSM<P>
where
    P: ArkPairing,
    P::ScalarField: JoltField,
{
    fn msm(
        bases: &[JoltGTWrapper<P>],
        scalars: &DoryMultilinearPolynomial<JoltFieldWrapper<P::ScalarField>>,
    ) -> JoltGTWrapper<P> {
        use rayon::prelude::*;

        match scalars {
            DoryMultilinearPolynomial::LargeScalars(coeffs) => {

                let chunk_size = (coeffs.len() / rayon::current_num_threads()).max(32);

                bases
                    .par_chunks(chunk_size)
                    .zip(coeffs.par_chunks(chunk_size))
                    .map(|(base_chunk, coeff_chunk)| {
                        base_chunk
                            .iter()
                            .zip(coeff_chunk.iter())
                            .filter(|(_, coeff)| !coeff.0.is_zero())
                            .fold(JoltGTWrapper::<P>::identity(), |acc, (base, coeff)| {
                                acc.add(&base.scale(coeff))
                            })
                    })
                    .reduce(JoltGTWrapper::<P>::identity, |a, b| a.add(&b))
            }
            DoryMultilinearPolynomial::U8Scalars(coeffs) => {
                msm_gt_small::<P, u8>(bases, coeffs)
            }
            DoryMultilinearPolynomial::U16Scalars(coeffs) => msm_gt_small::<P, u16>(bases, coeffs),
            DoryMultilinearPolynomial::U32Scalars(coeffs) => msm_gt_small::<P, u32>(bases, coeffs),
            DoryMultilinearPolynomial::U64Scalars(coeffs) => msm_gt_small::<P, u64>(bases, coeffs),
            DoryMultilinearPolynomial::I64Scalars(coeffs) => {

                let field_coeffs: Vec<JoltFieldWrapper<P::ScalarField>> = coeffs
                    .iter()
                    .map(|&x| JoltFieldWrapper(P::ScalarField::from_i64(x)))
                    .collect();

                bases
                    .iter()
                    .zip(field_coeffs.iter())
                    .filter(|(_, coeff)| !coeff.0.is_zero())
                    .fold(JoltGTWrapper::<P>::identity(), |acc, (base, coeff)| {
                        acc.add(&base.scale(coeff))
                    })
            }
        }
    }
}

// ===== Pairing Implementation =====

/// Generic pairing implementation
#[derive(Clone)]
pub struct JoltPairing<E: ArkPairing>(PhantomData<E>);

impl<E> DoryPairing for JoltPairing<E>
where
    E: ArkPairing,
    E::ScalarField: JoltField,
    E::G1: Icicle,
    E::G2: Icicle,
{
    type G1 = JoltGroupWrapper<E::G1>;
    type G2 = JoltGroupWrapper<E::G2>;
    type GT = JoltGTWrapper<E>;

    fn pair(p: &Self::G1, q: &Self::G2) -> Self::GT {
        let gt = E::pairing(p.inner, q.inner).0;
        Self::GT { inner: gt }
    }

    fn multi_pair(ps: &[Self::G1], qs: &[Self::G2]) -> Self::GT {
        assert_eq!(ps.len(), qs.len(), "multi_pair requires equal length vectors");

        if ps.is_empty() {
            return Self::GT::identity();
        }

        let g1_inner: Vec<E::G1> = ps.iter().map(|p| p.inner).collect();
        let g2_inner: Vec<E::G2> = qs.iter().map(|q| q.inner).collect();

        // Normalize to affine coordinates
        let aff_left = E::G1::normalize_batch(&g1_inner);
        let aff_right = E::G2::normalize_batch(&g2_inner);

        let left = aff_left
            .par_iter()
            .map(E::G1Prepared::from)
            .collect::<Vec<_>>();
        let right = aff_right
            .par_iter()
            .map(E::G2Prepared::from)
            .collect::<Vec<_>>();

        // We want to process N chunks in parallel where N is the number of threads available
        let num_chunks = rayon::current_num_threads();

        let chunk_size = if num_chunks <= left.len() {
            left.len() / num_chunks
        } else {
            // More threads than elements. Just do it all in parallel
            1
        };

        let (left_chunks, right_chunks) =
            (left.par_chunks(chunk_size), right.par_chunks(chunk_size));

        // Compute all the (partial) pairings and take the product. We have to take the product over
        // E::TargetField because MillerLoopOutput doesn't impl Product
        let ml_result = left_chunks
            .zip(right_chunks)
            .map(|(aa, bb)| E::multi_miller_loop(aa.iter().cloned(), bb.iter().cloned()).0)
            .product();

        let pairing_result = E::final_exponentiation(MillerLoopOutput(ml_result))
            .expect("Final exponentiation should not fail");
        
        Self::GT {
            inner: pairing_result.0,
        }
    }
}

// ===== Polynomial Conversions =====

impl<'a, F: JoltField> From<&DoryMultilinearPolynomial<'a, JoltFieldWrapper<F>>>
    for MultilinearPolynomial<F>
{
    fn from(dory_poly: &DoryMultilinearPolynomial<'a, JoltFieldWrapper<F>>) -> Self {
        match dory_poly {
            DoryMultilinearPolynomial::LargeScalars(scalars) => {
                // SAFETY: JoltFieldWrapper is #[repr(transparent)] over F, so &[JoltFieldWrapper<F>]
                // has the same memory layout as &[F]. We can transmute the slice reference.
                let field_scalars_slice: &[F] = unsafe {
                    std::slice::from_raw_parts(scalars.as_ptr() as *const F, scalars.len())
                };
                // Note: This still allocates because DensePolynomial owns its data.
                // @TODO(markosg04) For true zero-cost, DensePolynomial needs to be refactored slightly.
                MultilinearPolynomial::LargeScalars(DensePolynomial::new(
                    field_scalars_slice.to_vec(),
                ))
            }
            DoryMultilinearPolynomial::U8Scalars(scalars) => {
                MultilinearPolynomial::U8Scalars(CompactPolynomial::from_coeffs(scalars.to_vec()))
            }
            DoryMultilinearPolynomial::U16Scalars(scalars) => {
                MultilinearPolynomial::U16Scalars(CompactPolynomial::from_coeffs(scalars.to_vec()))
            }
            DoryMultilinearPolynomial::U32Scalars(scalars) => {
                MultilinearPolynomial::U32Scalars(CompactPolynomial::from_coeffs(scalars.to_vec()))
            }
            DoryMultilinearPolynomial::U64Scalars(scalars) => {
                MultilinearPolynomial::U64Scalars(CompactPolynomial::from_coeffs(scalars.to_vec()))
            }
            DoryMultilinearPolynomial::I64Scalars(scalars) => {
                MultilinearPolynomial::I64Scalars(CompactPolynomial::from_coeffs(scalars.to_vec()))
            }
        }
    }
}

impl<'a, F: JoltField> From<&'a MultilinearPolynomial<F>>
    for DoryMultilinearPolynomial<'a, JoltFieldWrapper<F>>
{
    fn from(
        src: &'a MultilinearPolynomial<F>,
    ) -> DoryMultilinearPolynomial<'a, JoltFieldWrapper<F>> {
        use DoryMultilinearPolynomial as Dory;

        match src {
            MultilinearPolynomial::LargeScalars(dense) => {
                // SAFETY: JoltFieldWrapper is #[repr(transparent)] over F,
                // hence always the same memory layout
                let wrapped: &'a [JoltFieldWrapper<F>] =
                    unsafe { std::slice::from_raw_parts(dense.Z.as_ptr().cast(), dense.Z.len()) };
                Dory::LargeScalars(wrapped)
            }
            MultilinearPolynomial::U8Scalars(cp) => Dory::U8Scalars(cp.coeffs.as_slice()),
            MultilinearPolynomial::U16Scalars(cp) => Dory::U16Scalars(cp.coeffs.as_slice()),
            MultilinearPolynomial::U32Scalars(cp) => Dory::U32Scalars(cp.coeffs.as_slice()),
            MultilinearPolynomial::U64Scalars(cp) => Dory::U64Scalars(cp.coeffs.as_slice()),
            MultilinearPolynomial::I64Scalars(cp) => Dory::I64Scalars(cp.coeffs.as_slice()),
        }
    }
}

// ===== Transcript Adapter =====

/// NewType wrapper for Jolt transcript to Dory transcript
#[derive(Default)]
pub struct JoltToDoryTranscriptRef<'a, F: JoltField, T: Transcript> {
    transcript: Option<&'a mut T>,
    _phantom: PhantomData<F>,
}

impl<'a, F: JoltField, T: Transcript> JoltToDoryTranscriptRef<'a, F, T> {
    pub fn new(transcript: &'a mut T) -> Self {
        JoltToDoryTranscriptRef {
            transcript: Some(transcript),
            _phantom: PhantomData,
        }
    }
}

impl<'a, F: JoltField, T: Transcript> DoryTranscript for JoltToDoryTranscriptRef<'a, F, T> {
    type Scalar = JoltFieldWrapper<F>;

    fn append_bytes(&mut self, label: &[u8], bytes: &[u8]) {
        let transcript = self
            .transcript
            .as_mut()
            .expect("Transcript not initialized");
        let label_str: &'static [u8] = Box::leak(label.to_vec().into_boxed_slice());
        transcript.append_message(label_str);
        transcript.append_bytes(bytes);
    }

    fn append_field(&mut self, label: &[u8], x: &Self::Scalar) {
        let transcript = self
            .transcript
            .as_mut()
            .expect("Transcript not initialized");
        let label_str: &'static [u8] = Box::leak(label.to_vec().into_boxed_slice());
        transcript.append_message(label_str);
        transcript.append_scalar(&x.0);
    }

    fn append_group<G: CanonicalSerialize>(&mut self, label: &[u8], g: &G) {
        let transcript = self
            .transcript
            .as_mut()
            .expect("Transcript not initialized");
        let label_str: &'static [u8] = Box::leak(label.to_vec().into_boxed_slice());
        transcript.append_message(label_str);
        transcript.append_serializable(g);
    }

    fn append_serde<S: serde::Serialize>(&mut self, label: &[u8], s: &S) {
        let transcript = self
            .transcript
            .as_mut()
            .expect("Transcript not initialized");
        let label_str: &'static [u8] = Box::leak(label.to_vec().into_boxed_slice());
        transcript.append_message(label_str);
        let bytes = postcard::to_allocvec(s).unwrap_or_default();
        transcript.append_bytes(&bytes);
    }

    fn challenge_scalar(&mut self, label: &[u8]) -> Self::Scalar {
        let transcript = self
            .transcript
            .as_mut()
            .expect("Transcript not initialized");
        let label_str: &'static [u8] = Box::leak(label.to_vec().into_boxed_slice());
        transcript.append_message(label_str);
        JoltFieldWrapper(transcript.challenge_scalar::<F>())
    }

    fn reset(&mut self, _domain_label: &[u8]) {
        panic!("Reset not supported for JoltToDoryTranscript")
    }
}

// ===== Type Aliases =====

pub type JoltG1Wrapper = JoltGroupWrapper<G1Projective>;
pub type JoltG2Wrapper = JoltGroupWrapper<G2Projective>;
pub type JoltGTBn254 = JoltGTWrapper<Bn254>;

pub type JoltG1MSM = JoltCustomMSM<G1Projective>;
pub type JoltG2MSM = JoltCustomMSM<G2Projective>;
pub type JoltGTMSM = JoltGTCustomMSM<Bn254>;

pub type JoltBn254 = JoltPairing<Bn254>;

/// Slightly Optimized MSM for GT group
fn msm_gt_small<P, T>(bases: &[JoltGTWrapper<P>], scalars: &[T]) -> JoltGTWrapper<P>
where
    P: ArkPairing,
    P::ScalarField: JoltField,
    T: Into<u64> + Copy + PartialEq + Eq,
    u64: From<T>,
{
    use std::collections::HashMap;

    let mut buckets: HashMap<u64, JoltGTWrapper<P>> = HashMap::new();

    for (base, &scalar) in bases.iter().zip(scalars.iter()) {
        let scalar_u64: u64 = scalar.into();
        if scalar_u64 != 0 {
            buckets
                .entry(scalar_u64)
                .and_modify(|acc| *acc = acc.add(base))
                .or_insert_with(|| base.clone());
        }
    }

    buckets
        .into_iter()
        .map(|(scalar, base_sum)| {
            let field_scalar = JoltFieldWrapper(<P::ScalarField as JoltField>::from_u64(scalar));
            base_sum.scale(&field_scalar)
        })
        .fold(JoltGTWrapper::<P>::identity(), |acc, x| acc.add(&x))
}

// ===== Commitment Scheme Types =====

/// Dory commitment scheme implementation
#[derive(Clone, Debug)]
pub struct DoryCommitmentScheme<ProofTranscript: Transcript> {
    _phantom_transcript: PhantomData<ProofTranscript>,
}

/// Setup parameters for both prover and verifier
#[derive(Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct DorySetup {
    pub prover_setup: ProverSetup<JoltBn254>,
    pub verifier_setup: VerifierSetup<JoltBn254>,
    pub max_num_vars: usize,
}

/// Commitment to a polynomial
#[derive(Clone, Debug, Default, PartialEq, CanonicalSerialize, CanonicalDeserialize)]
pub struct DoryCommitment {
    pub commitment: JoltGTBn254,
}

/// Proof of polynomial evaluation
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct DoryProofData<F: JoltField> {
    pub evaluation: F,
    pub opening_point: Vec<F>,
    pub sigma: usize,
    pub dory_proof_data: DoryProof<JoltG1Wrapper, JoltG2Wrapper, JoltGTBn254>,
    _phantom_f_marker: PhantomData<F>,
}

/// Batched proof structure
#[derive(Default, Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct DoryBatchedProof<F: JoltField> {
    proofs: Vec<DoryProofData<F>>,
}

// ===== Commitment Scheme Implementation =====

impl<ProofTranscript> CommitmentScheme<ProofTranscript> for DoryCommitmentScheme<ProofTranscript>
where
    ProofTranscript: Transcript,
    P: Pairing + Clone + 'static + Debug + Default + PartialEq,
    P::G1: dory::arithmetic::Group<Scalar = JoltToDoryField<F>> + Debug,
    P::G2: dory::arithmetic::Group<Scalar = JoltToDoryField<F>> + Debug,
    P::GT: dory::arithmetic::Group<Scalar = JoltToDoryField<F>> + Clone + Debug + Default,
    OptimizedMsmG1: MultiScalarMul<P::G1>,
    OptimizedMsmG2: MultiScalarMul<P::G2>,
    DummyMsm<P::GT>: MultiScalarMul<P::GT>,
{
    type Field = Fr;
    type Setup = DorySetup;
    type Commitment = DoryCommitment;
    type Proof = DoryProofData<Self::Field>;
    type BatchedProof = DoryBatchedProof<Self::Field>;

    fn setup(max_num_vars: usize) -> Self::Setup {
        let (prover_setup, verifier_setup) =
            dory_setup::<JoltBn254, _>(ark_std::rand::thread_rng(), max_num_vars);

        DorySetup {
            prover_setup,
            verifier_setup,
            max_num_vars,
        }
    }

    fn commit(poly: &MultilinearPolynomial<Self::Field>, setup: &Self::Setup) -> Self::Commitment {
        let dory_poly: DoryMultilinearPolynomial<'_, JoltFieldWrapper<Self::Field>> = poly.into();

        let num_vars = poly.get_num_vars();
        let sigma = (num_vars + 1) / 2;

        let commitment_val =
            dory_commit::<JoltBn254, JoltG1MSM>(&dory_poly, 0, sigma, &setup.prover_setup);

        DoryCommitment {
            commitment: commitment_val,
        }
    }

    fn batch_commit<U>(_polys: &[U], _setup: &Self::Setup) -> Vec<Self::Commitment>
    where
        U: Borrow<MultilinearPolynomial<Self::Field>> + Sync,
    {
        todo!("Batch commit not yet implemented for Dory")
    }

    fn prove(
        setup: &Self::Setup,
        poly: &MultilinearPolynomial<Self::Field>,
        opening_point: &[Self::Field],
        transcript: &'a mut ProofTranscript,
    ) -> Self::Proof {
        let dory_poly: DoryMultilinearPolynomial<'_, JoltFieldWrapper<Self::Field>> = poly.into();
        let point_dory: Vec<JoltFieldWrapper<Self::Field>> =
            opening_point.iter().map(|&p| JoltFieldWrapper(p)).collect();

        let num_vars = poly.get_num_vars();
        let sigma = (num_vars + 1) / 2;
        let dory_transcript = JoltToDoryTranscriptRef::<Self::Field, _>::new(transcript);

        let (claimed_evaluation, proof_builder) = dory_evaluate::<
            JoltBn254,
            JoltToDoryTranscriptRef<'_, Self::Field, ProofTranscript>,
            JoltG1MSM,
            JoltG2MSM,
        >(
            &dory_poly,
            &point_dory,
            sigma,
            &setup.prover_setup,
            dory_transcript,
        );

        let dory_proof = proof_builder.build();

        DoryProofData {
            evaluation: claimed_evaluation.0,
            opening_point: opening_point.to_vec(),
            sigma,
            dory_proof_data: dory_proof,
            _phantom_f_marker: PhantomData,
        }
    }

    fn verify<'a>(
        proof: &Self::Proof,
        setup: &Self::Setup,
        transcript: &mut ProofTranscript,
        _opening_point: &[Self::Field],
        _opening: &Self::Field,
        commitment: &Self::Commitment,
    ) -> Result<(), ProofVerifyError> {
        let opening_point: Vec<JoltFieldWrapper<Self::Field>> = proof
            .opening_point
            .iter()
            .map(|&p| JoltFieldWrapper(p))
            .collect();

        let claimed_opening = JoltFieldWrapper(proof.evaluation);
        let dory_transcript = JoltToDoryTranscriptRef::<Self::Field, _>::new(transcript);
        let verifier_builder =
            DoryProofBuilder::from_proof_no_transcript(proof.dory_proof_data.clone());

        let verify_result = dory_verify::<
            JoltBn254,
            JoltToDoryTranscriptRef<'_, Self::Field, ProofTranscript>,
            JoltG1MSM,
            JoltG2MSM,
            JoltGTMSM,
        >(
            commitment.commitment.clone(),
            claimed_opening,
            &opening_point,
            verifier_builder,
            proof.sigma,
            &setup.verifier_setup,
            dory_transcript,
        );

        match verify_result {
            Ok(()) => Ok(()),
            Err(_) => Err(ProofVerifyError::InvalidOpeningProof),
        }
    }

    fn protocol_name() -> &'static [u8] {
        b"dory_commitment_scheme"
    }
}

impl crate::utils::transcript::AppendToTranscript for DoryCommitment {
    fn append_to_transcript<PT: Transcript>(&self, transcript: &mut PT) {
        transcript.append_serializable(&self.commitment);
    }
}


// ===== Tests =====

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::transcript::KeccakTranscript;
    use crate::poly::compact_polynomial::CompactPolynomial;
    use ark_std::rand::thread_rng;
    use ark_std::UniformRand;
    use std::time::Instant;

    /// Test setup helper that creates a setup once and reuses it
    fn create_test_setup(max_num_vars: usize) -> DorySetup {
        DoryCommitmentScheme::<KeccakTranscript>::setup(max_num_vars)
    }

    /// Helper function to run the full commitment scheme test
    fn test_commitment_scheme_with_poly(
        poly: MultilinearPolynomial<Fr>,
        poly_type_name: &str,
        setup: &DorySetup,
    ) -> (std::time::Duration, std::time::Duration, std::time::Duration, std::time::Duration) {
        let num_vars = poly.get_num_vars();
        let num_coeffs = match &poly {
            MultilinearPolynomial::LargeScalars(dense) => dense.Z.len(),
            MultilinearPolynomial::U8Scalars(compact) => compact.coeffs.len(),
            MultilinearPolynomial::U16Scalars(compact) => compact.coeffs.len(),
            MultilinearPolynomial::U32Scalars(compact) => compact.coeffs.len(),
            MultilinearPolynomial::U64Scalars(compact) => compact.coeffs.len(),
            MultilinearPolynomial::I64Scalars(compact) => compact.coeffs.len(),
        };

        println!(
            "Testing Dory PCS ({}) with {} variables, {} coefficients",
            poly_type_name, num_vars, num_coeffs
        );

        let mut rng = thread_rng();
        let opening_point: Vec<Fr> = (0..num_vars).map(|_| Fr::rand(&mut rng)).collect();

        let commit_start = Instant::now();
        let commitment = DoryCommitmentScheme::<KeccakTranscript>::commit(&poly, setup);
        let commit_time = commit_start.elapsed();
        println!("  Commit time: {:?}", commit_time);

        let mut prove_transcript = KeccakTranscript::new(b"dory_test");
        let prove_start = Instant::now();
        let proof = DoryCommitmentScheme::<KeccakTranscript>::prove(
            setup,
            &poly,
            &opening_point,
            &mut prove_transcript,
        );
        let prove_time = prove_start.elapsed();
        println!("  Prove time: {:?}", prove_time);

        let mut verify_transcript = KeccakTranscript::new(b"dory_test");
        let verify_start = Instant::now();
        let verification_result = DoryCommitmentScheme::<KeccakTranscript>::verify(
            &proof,
            setup,
            &mut verify_transcript,
            &opening_point,
            &proof.evaluation,
            &commitment,
        );
        let verify_time = verify_start.elapsed();
        println!("  Verify time: {:?}", verify_time);

        let total_time = commit_time + prove_time + verify_time;
        println!("  Total time (without setup): {:?}", total_time);

        assert!(
            verification_result.is_ok(),
            "Dory verification failed for {}: {:?}",
            poly_type_name,
            verification_result
        );

        println!("  ✅ {} test passed!\n", poly_type_name);

        (commit_time, prove_time, verify_time, total_time)
    }

    #[test]
    fn test_dory_commitment_scheme_all_polynomial_types() {
        let max_num_vars = 18;
        let num_vars = 18;
        let num_coeffs = 1 << num_vars;

        println!("Setting up Dory PCS with max_num_vars = {}", max_num_vars);
        let setup_start = Instant::now();
        let setup = create_test_setup(max_num_vars);
        let setup_time = setup_start.elapsed();
        println!("Setup time: {:?}\n", setup_time);

        let mut rng = thread_rng();

        // Test 1: LargeScalars (Field elements)
        let coeffs_large: Vec<Fr> = (0..num_coeffs).map(|_| Fr::rand(&mut rng)).collect();
        let poly_large = MultilinearPolynomial::LargeScalars(DensePolynomial::new(coeffs_large));
        let (commit_large, prove_large, verify_large, total_large) =
            test_commitment_scheme_with_poly(poly_large, "LargeScalars", &setup);

        // Test 2: U8Scalars
        let coeffs_u8: Vec<u8> = (0..num_coeffs).map(|_| rng.next_u32() as u8).collect();
        let poly_u8 = MultilinearPolynomial::U8Scalars(CompactPolynomial::from_coeffs(coeffs_u8));
        let (commit_u8, prove_u8, verify_u8, total_u8) =
            test_commitment_scheme_with_poly(poly_u8, "U8Scalars", &setup);

        // Test 3: U16Scalars
        let coeffs_u16: Vec<u16> = (0..num_coeffs).map(|_| rng.next_u32() as u16).collect();
        let poly_u16 = MultilinearPolynomial::U16Scalars(CompactPolynomial::from_coeffs(coeffs_u16));
        let (commit_u16, prove_u16, verify_u16, total_u16) =
            test_commitment_scheme_with_poly(poly_u16, "U16Scalars", &setup);

        // Test 4: U32Scalars
        let coeffs_u32: Vec<u32> = (0..num_coeffs).map(|_| rng.next_u32()).collect();
        let poly_u32 = MultilinearPolynomial::U32Scalars(CompactPolynomial::from_coeffs(coeffs_u32));
        let (commit_u32, prove_u32, verify_u32, total_u32) =
            test_commitment_scheme_with_poly(poly_u32, "U32Scalars", &setup);

        // Test 5: U64Scalars
        let coeffs_u64: Vec<u64> = (0..num_coeffs).map(|_| rng.next_u64()).collect();
        let poly_u64 = MultilinearPolynomial::U64Scalars(CompactPolynomial::from_coeffs(coeffs_u64));
        let (commit_u64, prove_u64, verify_u64, total_u64) =
            test_commitment_scheme_with_poly(poly_u64, "U64Scalars", &setup);

        // Test 6: I64Scalars
        let coeffs_i64: Vec<i64> = (0..num_coeffs).map(|_| rng.next_u64() as i64).collect();
        let poly_i64 = MultilinearPolynomial::I64Scalars(CompactPolynomial::from_coeffs(coeffs_i64));
        let (commit_i64, prove_i64, verify_i64, total_i64) =
            test_commitment_scheme_with_poly(poly_i64, "I64Scalars", &setup);

        // Summary of results
        println!("========== PERFORMANCE SUMMARY ==========");
        println!("Setup time: {:?}", setup_time);
        println!();
        println!("Polynomial Type | Commit Time | Prove Time  | Verify Time | Total Time");
        println!("----------------|-------------|-------------|-------------|------------");
        println!("LargeScalars    | {:>11?} | {:>11?} | {:>11?} | {:>10?}", commit_large, prove_large, verify_large, total_large);
        println!("U8Scalars       | {:>11?} | {:>11?} | {:>11?} | {:>10?}", commit_u8, prove_u8, verify_u8, total_u8);
        println!("U16Scalars      | {:>11?} | {:>11?} | {:>11?} | {:>10?}", commit_u16, prove_u16, verify_u16, total_u16);
        println!("U32Scalars      | {:>11?} | {:>11?} | {:>11?} | {:>10?}", commit_u32, prove_u32, verify_u32, total_u32);
        println!("U64Scalars      | {:>11?} | {:>11?} | {:>11?} | {:>10?}", commit_u64, prove_u64, verify_u64, total_u64);
        println!("I64Scalars      | {:>11?} | {:>11?} | {:>11?} | {:>10?}", commit_i64, prove_i64, verify_i64, total_i64);
        println!("==========================================");
    }

    #[test]
    fn test_dory_soundness() {
        use ark_std::UniformRand;

        // Test setup
        let num_vars = 10;
        let max_num_vars = 10;
        let num_coeffs = 1 << num_vars;

        let mut rng = thread_rng();
        let coeffs: Vec<Fr> = (0..num_coeffs).map(|_| Fr::rand(&mut rng)).collect();
        let poly = MultilinearPolynomial::LargeScalars(DensePolynomial::new(coeffs.clone()));

        let opening_point: Vec<Fr> = (0..num_vars).map(|_| Fr::rand(&mut rng)).collect();

        let setup = DoryCommitmentScheme::<KeccakTranscript>::setup(max_num_vars);

        // Commit to the polynomial
        let commitment = DoryCommitmentScheme::<KeccakTranscript>::commit(&poly, &setup);

        let mut prove_transcript =
            KeccakTranscript::new(DoryCommitmentScheme::<KeccakTranscript>::protocol_name());

        // Generate the proof
        let proof = DoryCommitmentScheme::<KeccakTranscript>::prove(
            &setup,
            &poly,
            &opening_point,
            &mut prove_transcript,
        );

        // Now we consider 4 cases of tampering, which Dory should reject.

        // Test 1: Tamper with the evaluation
        {
            let mut tampered_proof = proof.clone();
            tampered_proof.evaluation = Fr::rand(&mut rng); // Random wrong evaluation

            let mut verify_transcript = KeccakTranscript::new(DoryCommitmentScheme::<KeccakTranscript>::protocol_name());
            let result = DoryCommitmentScheme::<KeccakTranscript>::verify(
                &tampered_proof,
                &setup,
                &mut verify_transcript,
                &opening_point,
                &proof.evaluation, // Use original evaluation as expected
                &commitment,
            );

            assert!(
                result.is_err(),
                "Verification should fail with tampered evaluation"
            );
            println!("✅ Test 1 passed: Tampered evaluation correctly rejected");
        }

        // Test 2: Tamper with the opening point
        {
            let mut tampered_proof = proof.clone();
            tampered_proof.opening_point = (0..num_vars).map(|_| Fr::rand(&mut rng)).collect();

            let mut verify_transcript = KeccakTranscript::new(DoryCommitmentScheme::<KeccakTranscript>::protocol_name());
            let result = DoryCommitmentScheme::<KeccakTranscript>::verify(
                &tampered_proof,
                &setup,
                &mut verify_transcript,
                &opening_point, // Use original opening point
                &proof.evaluation,
                &commitment,
            );

            assert!(
                result.is_err(),
                "Verification should fail with tampered opening point"
            );
            println!("✅ Test 2 passed: Tampered opening point correctly rejected");
        }

        // Test 3: Use wrong commitment
        {
            // Create a different polynomial and its commitment
            let wrong_coeffs: Vec<Fr> = (0..num_coeffs).map(|_| Fr::rand(&mut rng)).collect();
            let wrong_poly =
                MultilinearPolynomial::LargeScalars(DensePolynomial::new(wrong_coeffs));
            let wrong_commitment =
                DoryCommitmentScheme::<KeccakTranscript>::commit(&wrong_poly, &setup);

            let mut verify_transcript = KeccakTranscript::new(DoryCommitmentScheme::<KeccakTranscript>::protocol_name());
            let result = DoryCommitmentScheme::<KeccakTranscript>::verify(
                &proof,
                &setup,
                &mut verify_transcript,
                &opening_point,
                &proof.evaluation,
                &wrong_commitment, // Wrong commitment
            );

            assert!(
                result.is_err(),
                "Verification should fail with wrong commitment"
            );
            println!("✅ Test 3 passed: Wrong commitment correctly rejected");
        }

        // Test 4: Use wrong domain in transcript
        {
            let mut verify_transcript = KeccakTranscript::new(b"wrong_domain");
            let result = DoryCommitmentScheme::<KeccakTranscript>::verify(
                &proof,
                &setup,
                &mut verify_transcript,
                &opening_point,
                &proof.evaluation,
                &commitment,
            );

            assert!(
                result.is_err(),
                "Verification should fail with wrong transcript domain"
            );
            println!("✅ Test 4 passed: Wrong transcript domain correctly rejected");
        }

        // Test 5: Verify that correct proof still passes
        {
            let mut verify_transcript = KeccakTranscript::new(DoryCommitmentScheme::<KeccakTranscript>::protocol_name());
            let result = DoryCommitmentScheme::<KeccakTranscript>::verify(
                &proof,
                &setup,
                &mut verify_transcript,
                &opening_point,
                &proof.evaluation,
                &commitment,
            );

            assert!(
                result.is_ok(),
                "Verification should succeed with correct proof"
            );
            println!("✅ Test 5 passed: Correct proof indeed verifies successfully");
        }
    }
}
