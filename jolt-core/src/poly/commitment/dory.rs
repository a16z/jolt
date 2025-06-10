use super::commitment_scheme::CommitmentScheme;
use crate::{
    field::JoltField,
    msm::{Icicle, VariableBaseMSM},
    poly::multilinear_polynomial::MultilinearPolynomial,
    utils::{
        errors::ProofVerifyError,
        transcript::{AppendToTranscript, Transcript},
    },
};
use ark_bn254::{Bn254, Fr, G1Projective, G2Projective};
use ark_ec::{
    pairing::{MillerLoopOutput, Pairing as ArkPairing, PairingOutput},
    CurveGroup,
};
use ark_ff::{Field, One, PrimeField, UniformRand};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{rand::RngCore, Zero};
use rayon::prelude::*;
use std::{borrow::Borrow, marker::PhantomData};

use dory::{
    arithmetic::{
        Field as DoryField, Group as DoryGroup, MultiScalarMul as DoryMultiScalarMul,
        Pairing as DoryPairing,
    },
    commit, evaluate, setup as dory_setup,
    transcript::Transcript as DoryTranscript,
    verify, DoryProof, DoryProofBuilder, Polynomial as DoryPolynomial, ProverSetup, VerifierSetup,
};

// NewType wrappers for Jolt + arkworks types to interop with Dory traits
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

#[repr(transparent)]
#[derive(Debug, Clone, PartialEq, CanonicalSerialize, CanonicalDeserialize)]
pub struct JoltGroupWrapper<G: CurveGroup>(pub G);

impl<G> DoryGroup for JoltGroupWrapper<G>
where
    G: CurveGroup + VariableBaseMSM,
    G::ScalarField: JoltField,
{
    type Scalar = JoltFieldWrapper<G::ScalarField>;

    fn identity() -> Self {
        Self(G::zero())
    }

    fn add(&self, rhs: &Self) -> Self {
        Self(self.0 + rhs.0)
    }

    fn neg(&self) -> Self {
        Self(-self.0)
    }

    fn scale(&self, k: &Self::Scalar) -> Self {
        Self(self.0 * k.0)
    }

    fn random<R: RngCore>(rng: &mut R) -> Self {
        Self(G::rand(rng))
    }
}

#[repr(transparent)]
#[derive(Debug, Clone, PartialEq, CanonicalSerialize, CanonicalDeserialize)]
pub struct JoltGTWrapper<P: ArkPairing>(pub P::TargetField);

impl<P: ArkPairing> From<PairingOutput<P>> for JoltGTWrapper<P> {
    fn from(value: PairingOutput<P>) -> Self {
        Self(value.0)
    }
}

impl<P: ArkPairing> Into<PairingOutput<P>> for JoltGTWrapper<P> {
    fn into(self) -> PairingOutput<P> {
        PairingOutput(self.0)
    }
}

impl<P> DoryGroup for JoltGTWrapper<P>
where
    P: ArkPairing,
    P::ScalarField: JoltField,
{
    type Scalar = JoltFieldWrapper<P::ScalarField>;

    fn identity() -> Self {
        Self(P::TargetField::one())
    }

    fn add(&self, rhs: &Self) -> Self {
        Self(self.0 * rhs.0)
    }

    fn neg(&self) -> Self {
        Self(self.0.inverse().expect("GT element should be invertible"))
    }

    fn scale(&self, k: &Self::Scalar) -> Self {
        Self(self.0.pow(k.0.into_bigint()))
    }

    fn random<R: RngCore>(rng: &mut R) -> Self {
        Self(P::TargetField::rand(rng))
    }
}

impl<P: ArkPairing> Default for JoltGTWrapper<P> {
    fn default() -> Self {
        Self(P::TargetField::one())
    }
}

impl<P> std::iter::Sum for JoltGTWrapper<P>
where
    P: ArkPairing,
    P::ScalarField: JoltField,
{
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::identity(), |acc, x| acc.add(&x))
    }
}

// Dory accepts an MSM trait for Dory Group and F, so we provide it with one.
// In the future, we can think about removing this trait in Dory (hence this wrapper)
// and just use our arkworks fork smart MSM (the Dory crate already points to our arkworks fork)
pub struct JoltMSM;

impl<G> DoryMultiScalarMul<JoltGroupWrapper<G>> for JoltMSM
where
    G: CurveGroup + VariableBaseMSM,
    G::ScalarField: JoltField,
{
    fn msm(
        bases: &[JoltGroupWrapper<G>],
        scalars: &[JoltFieldWrapper<G::ScalarField>],
    ) -> JoltGroupWrapper<G> {
        let affines: Vec<_> = bases.iter().map(|w| w.0.into_affine()).collect();

        // # Safety
        // JoltFieldWrapper always has same memory layout as underlying G::ScalarField here.
        let raw_scalars: &[G::ScalarField] = unsafe {
            std::slice::from_raw_parts(scalars.as_ptr() as *const G::ScalarField, scalars.len())
        };

        let result = G::msm_field_elements(&affines, None, raw_scalars, None, false)
            .expect("msm_field_elements should not fail");

        JoltGroupWrapper(result)
    }
}

// We implement MSM for specifically GT (the other case handles G1 and G2)
impl<P> DoryMultiScalarMul<JoltGTWrapper<P>> for JoltMSM
where
    P: ArkPairing,
    P::ScalarField: JoltField,
{
    fn msm(
        bases: &[JoltGTWrapper<P>],
        scalars: &[JoltFieldWrapper<P::ScalarField>],
    ) -> JoltGTWrapper<P> {
        let chunk_size = (scalars.len() / rayon::current_num_threads()).max(32);

        bases
            .par_chunks(chunk_size)
            .zip(scalars.par_chunks(chunk_size))
            .map(|(base_chunk, coeff_chunk)| {
                base_chunk
                    .iter()
                    .zip(coeff_chunk.iter())
                    .filter(|(_, coeff)| !coeff.0.is_zero())
                    .fold(JoltGTWrapper::<P>::identity(), |acc, (base, coeff)| {
                        acc.add(&base.scale(coeff))
                    })
            })
            .sum()
    }
}

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
        let gt = E::pairing(p.0, q.0).0;
        JoltGTWrapper(gt)
    }

    fn multi_pair(ps: &[Self::G1], qs: &[Self::G2]) -> Self::GT {
        assert_eq!(
            ps.len(),
            qs.len(),
            "multi_pair requires equal length vectors"
        );

        if ps.is_empty() {
            return Self::GT::identity();
        }

        let g1_inner: Vec<E::G1> = ps.iter().map(|p| p.0).collect();
        let g2_inner: Vec<E::G2> = qs.iter().map(|q| q.0).collect();

        let aff_left = E::G1::normalize_batch(&g1_inner);
        let aff_right = E::G2::normalize_batch(&g2_inner);

        let left: Vec<_> = aff_left.par_iter().map(E::G1Prepared::from).collect();
        let right: Vec<_> = aff_right.par_iter().map(E::G2Prepared::from).collect();

        let num_chunks = rayon::current_num_threads();
        let chunk_size = (left.len() / num_chunks.max(1)).max(1);

        let ml_result = left
            .par_chunks(chunk_size)
            .zip(right.par_chunks(chunk_size))
            .map(|(aa, bb)| E::multi_miller_loop(aa.iter().cloned(), bb.iter().cloned()).0)
            .product();

        let pairing_result = E::final_exponentiation(MillerLoopOutput(ml_result))
            .expect("Final exponentiation should not fail");

        JoltGTWrapper(pairing_result.0)
    }
}

// Dory's Poly trait: right now it uses default implementations for the poly utilities. We will want to override them
// for the sparse case or for further optimizations.
// We implement get() and len() since they do not have default implementations.
impl<F, G> DoryPolynomial<JoltFieldWrapper<F>, JoltGroupWrapper<G>> for MultilinearPolynomial<F>
where
    F: JoltField + PrimeField,
    G: CurveGroup<ScalarField = F> + VariableBaseMSM,
{
    fn get(&self, index: usize) -> JoltFieldWrapper<F> {
        assert!(
            index < self.len(),
            "Polynomial index out of bounds: {} >= {}",
            index,
            self.len()
        );
        JoltFieldWrapper(self.get_coeff(index))
    }

    fn len(&self) -> usize {
        self.len()
    }

    //     fn commit_rows<M1: DoryMultiScalarMul<JoltGroupWrapper<G>>>(
    //         &self,
    //         g1_generators: &[JoltGroupWrapper<G>],
    //         row_len: usize,
    //     ) -> Vec<JoltGroupWrapper<G>> {
    //         todo!()
    //     }

    //     fn vector_matrix_product(
    //         &self,
    //         left_vec: &[JoltFieldWrapper<F>],
    //         sigma: usize,
    //         nu: usize,
    //     ) -> Vec<JoltFieldWrapper<F>> {
    //         todo!()
    //     }
}

// Note that we have this `Option<&'a mut T>` so that we can derive Default, which is required.
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

    fn append_bytes(&mut self, _label: &[u8], bytes: &[u8]) {
        let transcript = self
            .transcript
            .as_mut()
            .expect("Transcript not initialized");
        transcript.append_bytes(bytes);
    }

    fn append_field(&mut self, _label: &[u8], x: &Self::Scalar) {
        let transcript = self
            .transcript
            .as_mut()
            .expect("Transcript not initialized");
        transcript.append_scalar(&x.0);
    }

    fn append_group<G: CanonicalSerialize>(&mut self, _label: &[u8], g: &G) {
        let transcript = self
            .transcript
            .as_mut()
            .expect("Transcript not initialized");
        transcript.append_serializable(g);
    }

    fn append_serde<S: serde::Serialize>(&mut self, _label: &[u8], s: &S) {
        let transcript = self
            .transcript
            .as_mut()
            .expect("Transcript not initialized");
        let bytes = postcard::to_allocvec(s).unwrap_or_default();
        transcript.append_bytes(&bytes);
    }

    fn challenge_scalar(&mut self, _label: &[u8]) -> Self::Scalar {
        let transcript = self
            .transcript
            .as_mut()
            .expect("Transcript not initialized");
        JoltFieldWrapper(transcript.challenge_scalar::<F>())
    }

    fn reset(&mut self, _domain_label: &[u8]) {
        panic!("Reset not supported for JoltToDoryTranscript")
    }
}

// BN254-specific Aliases
pub type JoltG1Wrapper = JoltGroupWrapper<G1Projective>;
pub type JoltG2Wrapper = JoltGroupWrapper<G2Projective>;
pub type JoltGTBn254 = JoltGTWrapper<Bn254>;

pub type JoltBn254 = JoltPairing<Bn254>;

#[derive(Clone, Debug)]
pub struct DoryCommitmentScheme<ProofTranscript: Transcript>(PhantomData<ProofTranscript>);

#[derive(Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct DorySetup {
    pub prover_setup: ProverSetup<JoltBn254>,
    pub verifier_setup: VerifierSetup<JoltBn254>,
    pub max_num_vars: usize,
}

#[derive(Clone, Debug, Default, PartialEq, CanonicalSerialize, CanonicalDeserialize)]
pub struct DoryCommitment(pub JoltGTBn254);

#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct DoryProofData {
    pub sigma: usize,
    pub dory_proof_data: DoryProof<JoltG1Wrapper, JoltG2Wrapper, JoltGTBn254>,
}

#[derive(Default, Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct DoryBatchedProof {
    proofs: Vec<DoryProofData>,
}

impl<ProofTranscript> CommitmentScheme<ProofTranscript> for DoryCommitmentScheme<ProofTranscript>
where
    ProofTranscript: Transcript,
{
    type Field = Fr;
    type Setup = DorySetup;
    type Commitment = DoryCommitment;
    type Proof = DoryProofData;
    type BatchedProof = DoryBatchedProof;

    fn setup(max_num_vars: usize) -> Self::Setup {
        let (prover_setup, verifier_setup) =
            dory_setup::<JoltBn254, _>(ark_std::rand::thread_rng(), max_num_vars);

        DorySetup {
            prover_setup,
            verifier_setup,
            max_num_vars,
        }
    }

    #[tracing::instrument(skip_all, name = "DoryCommitmentScheme::commit")]
    fn commit(poly: &MultilinearPolynomial<Self::Field>, setup: &Self::Setup) -> Self::Commitment {
        let num_vars = poly.get_num_vars();
        let sigma = (num_vars + 1) / 2;

        let commitment_val = commit::<JoltBn254, JoltMSM, _>(poly, 0, sigma, &setup.prover_setup);
        DoryCommitment(commitment_val)
    }

    fn batch_commit<U>(_polys: &[U], _setup: &Self::Setup) -> Vec<Self::Commitment>
    where
        U: Borrow<MultilinearPolynomial<Self::Field>> + Sync,
    {
        todo!("Batch commit not yet implemented for Dory")
    }

    // Note that Dory implementation sometimes uses the term 'evaluation'/'evaluate' -- this is same as 'opening'/'open'
    #[tracing::instrument(skip_all, name = "DoryCommitmentScheme::prove")]
    fn prove(
        setup: &Self::Setup,
        poly: &MultilinearPolynomial<Self::Field>,
        opening_point: &[Self::Field],
        transcript: &mut ProofTranscript,
    ) -> Self::Proof {
        let point_dory: Vec<JoltFieldWrapper<Self::Field>> =
            opening_point.iter().map(|&p| JoltFieldWrapper(p)).collect();

        let num_vars = poly.get_num_vars();
        let sigma = (num_vars + 1) / 2;
        let dory_transcript = JoltToDoryTranscriptRef::<Self::Field, _>::new(transcript);

        // dory evaluate returns the opening but in this case we don't use it, we pass directly the opening to verify()
        let (_claimed_evaluation, proof_builder) = evaluate::<
            JoltBn254,
            JoltToDoryTranscriptRef<'_, Self::Field, ProofTranscript>,
            JoltMSM,
            JoltMSM,
            _,
        >(
            poly,
            &point_dory,
            sigma,
            &setup.prover_setup,
            dory_transcript,
        );

        let dory_proof = proof_builder.build();

        DoryProofData {
            sigma,
            dory_proof_data: dory_proof,
        }
    }

    fn verify(
        proof: &Self::Proof,
        setup: &Self::Setup,
        transcript: &mut ProofTranscript,
        opening_point: &[Self::Field],
        opening: &Self::Field,
        commitment: &Self::Commitment,
    ) -> Result<(), ProofVerifyError> {
        let opening_point_dory: Vec<JoltFieldWrapper<Self::Field>> =
            opening_point.iter().map(|&p| JoltFieldWrapper(p)).collect();

        let claimed_opening = JoltFieldWrapper(*opening);
        let dory_transcript = JoltToDoryTranscriptRef::<Self::Field, _>::new(transcript);
        let verifier_builder =
            DoryProofBuilder::from_proof_no_transcript(proof.dory_proof_data.clone());

        let verify_result = verify::<
            JoltBn254,
            JoltToDoryTranscriptRef<'_, Self::Field, ProofTranscript>,
            JoltMSM,
            JoltMSM,
            JoltMSM,
        >(
            commitment.0.clone(),
            claimed_opening,
            &opening_point_dory,
            verifier_builder,
            proof.sigma,
            &setup.verifier_setup,
            dory_transcript,
        );

        match verify_result {
            Ok(()) => Ok(()),
            Err(e) => Err(ProofVerifyError::DoryError(format!("{:?}", e))),
        }
    }

    fn combine_commitments(
        commitments: &[&Self::Commitment],
        coeffs: &[Self::Field],
    ) -> Self::Commitment {
        let combined_commitment: PairingOutput<_> = commitments
            .iter()
            .zip(coeffs.iter())
            .map(|(commitment, coeff)| {
                let g: PairingOutput<_> = commitment.0.clone().into();
                g * coeff
            })
            .sum();
        DoryCommitment(JoltGTWrapper::from(combined_commitment))
    }

    fn protocol_name() -> &'static [u8] {
        b"dory_commitment_scheme"
    }
}

impl AppendToTranscript for DoryCommitment {
    fn append_to_transcript<PT: Transcript>(&self, transcript: &mut PT) {
        transcript.append_serializable(&self.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poly::compact_polynomial::CompactPolynomial;
    use crate::poly::dense_mlpoly::DensePolynomial;
    use crate::utils::transcript::KeccakTranscript;
    use ark_std::rand::thread_rng;
    use ark_std::UniformRand;
    use std::time::Instant;

    fn create_test_setup(max_num_vars: usize) -> DorySetup {
        DoryCommitmentScheme::<KeccakTranscript>::setup(max_num_vars)
    }

    fn test_commitment_scheme_with_poly(
        poly: MultilinearPolynomial<Fr>,
        poly_type_name: &str,
        setup: &DorySetup,
    ) -> (
        std::time::Duration,
        std::time::Duration,
        std::time::Duration,
        std::time::Duration,
    ) {
        let num_vars = poly.get_num_vars();
        let num_coeffs = match &poly {
            MultilinearPolynomial::LargeScalars(dense) => dense.Z.len(),
            MultilinearPolynomial::U8Scalars(compact) => compact.coeffs.len(),
            MultilinearPolynomial::U16Scalars(compact) => compact.coeffs.len(),
            MultilinearPolynomial::U32Scalars(compact) => compact.coeffs.len(),
            MultilinearPolynomial::U64Scalars(compact) => compact.coeffs.len(),
            MultilinearPolynomial::I64Scalars(compact) => compact.coeffs.len(),
            MultilinearPolynomial::Sparse(_) => todo!(),
            MultilinearPolynomial::OneHot(_) => todo!(),
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

        println!(" Commit time: {:?}", commit_time);

        // Compute the evaluation using the DoryPolynomial trait's evaluate method
        let opening_point_dory: Vec<JoltFieldWrapper<Fr>> =
            opening_point.iter().map(|&p| JoltFieldWrapper(p)).collect();
        let evaluation = <MultilinearPolynomial<Fr> as DoryPolynomial<
            JoltFieldWrapper<Fr>,
            JoltGroupWrapper<G1Projective>,
        >>::evaluate(&poly, &opening_point_dory)
        .0;

        let mut prove_transcript = KeccakTranscript::new(b"dory_test");
        let prove_start = Instant::now();
        let proof = DoryCommitmentScheme::<KeccakTranscript>::prove(
            setup,
            &poly,
            &opening_point,
            &mut prove_transcript,
        );
        let prove_time = prove_start.elapsed();

        println!(" Prove time: {:?}", prove_time);

        let mut verify_transcript = KeccakTranscript::new(b"dory_test");
        let verify_start = Instant::now();
        let verification_result = DoryCommitmentScheme::<KeccakTranscript>::verify(
            &proof,
            setup,
            &mut verify_transcript,
            &opening_point,
            &evaluation,
            &commitment,
        );
        let verify_time = verify_start.elapsed();

        println!(" Verify time: {:?}", verify_time);
        let total_time = commit_time + prove_time + verify_time;
        println!(" Total time (without setup): {:?}", total_time);

        assert!(
            verification_result.is_ok(),
            "Dory verification failed for {}: {:?}",
            poly_type_name,
            verification_result
        );
        println!(" ✅ {} test passed!\n", poly_type_name);

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
        let poly_u16 =
            MultilinearPolynomial::U16Scalars(CompactPolynomial::from_coeffs(coeffs_u16));
        let (commit_u16, prove_u16, verify_u16, total_u16) =
            test_commitment_scheme_with_poly(poly_u16, "U16Scalars", &setup);

        // Test 4: U32Scalars
        let coeffs_u32: Vec<u32> = (0..num_coeffs).map(|_| rng.next_u32()).collect();
        let poly_u32 =
            MultilinearPolynomial::U32Scalars(CompactPolynomial::from_coeffs(coeffs_u32));
        let (commit_u32, prove_u32, verify_u32, total_u32) =
            test_commitment_scheme_with_poly(poly_u32, "U32Scalars", &setup);

        // Test 5: U64Scalars
        let coeffs_u64: Vec<u64> = (0..num_coeffs).map(|_| rng.next_u64()).collect();
        let poly_u64 =
            MultilinearPolynomial::U64Scalars(CompactPolynomial::from_coeffs(coeffs_u64));
        let (commit_u64, prove_u64, verify_u64, total_u64) =
            test_commitment_scheme_with_poly(poly_u64, "U64Scalars", &setup);

        // Test 6: I64Scalars
        let coeffs_i64: Vec<i64> = (0..num_coeffs).map(|_| rng.next_u64() as i64).collect();
        let poly_i64 =
            MultilinearPolynomial::I64Scalars(CompactPolynomial::from_coeffs(coeffs_i64));
        let (commit_i64, prove_i64, verify_i64, total_i64) =
            test_commitment_scheme_with_poly(poly_i64, "I64Scalars", &setup);

        println!("========== PERFORMANCE SUMMARY ==========");

        println!("Setup time: {:?}\n", setup_time);

        println!("Polynomial Type | Commit Time | Prove Time | Verify Time | Total Time");

        println!("----------------|-------------|-------------|-------------|------------");
        println!(
            "LargeScalars | {:>11?} | {:>11?} | {:>11?} | {:>10?}",
            commit_large, prove_large, verify_large, total_large
        );
        println!(
            "U8Scalars | {:>11?} | {:>11?} | {:>11?} | {:>10?}",
            commit_u8, prove_u8, verify_u8, total_u8
        );
        println!(
            "U16Scalars | {:>11?} | {:>11?} | {:>11?} | {:>10?}",
            commit_u16, prove_u16, verify_u16, total_u16
        );
        println!(
            "U32Scalars | {:>11?} | {:>11?} | {:>11?} | {:>10?}",
            commit_u32, prove_u32, verify_u32, total_u32
        );
        println!(
            "U64Scalars | {:>11?} | {:>11?} | {:>11?} | {:>10?}",
            commit_u64, prove_u64, verify_u64, total_u64
        );
        println!(
            "I64Scalars | {:>11?} | {:>11?} | {:>11?} | {:>10?}",
            commit_i64, prove_i64, verify_i64, total_i64
        );
        println!("==========================================");
    }

    #[test]
    fn test_dory_soundness() {
        use ark_std::UniformRand;

        let num_vars = 10;
        let max_num_vars = 10;
        let num_coeffs = 1 << num_vars;

        let mut rng = thread_rng();
        let coeffs: Vec<Fr> = (0..num_coeffs).map(|_| Fr::rand(&mut rng)).collect();
        let poly = MultilinearPolynomial::LargeScalars(DensePolynomial::new(coeffs.clone()));

        let opening_point: Vec<Fr> = (0..num_vars).map(|_| Fr::rand(&mut rng)).collect();

        let setup = DoryCommitmentScheme::<KeccakTranscript>::setup(max_num_vars);

        let commitment = DoryCommitmentScheme::<KeccakTranscript>::commit(&poly, &setup);

        let mut prove_transcript =
            KeccakTranscript::new(DoryCommitmentScheme::<KeccakTranscript>::protocol_name());

        // Compute the correct evaluation
        let opening_point_dory: Vec<JoltFieldWrapper<Fr>> =
            opening_point.iter().map(|&p| JoltFieldWrapper(p)).collect();
        let correct_evaluation = <MultilinearPolynomial<Fr> as DoryPolynomial<
            JoltFieldWrapper<Fr>,
            JoltGroupWrapper<G1Projective>,
        >>::evaluate(&poly, &opening_point_dory)
        .0;

        let proof = DoryCommitmentScheme::<KeccakTranscript>::prove(
            &setup,
            &poly,
            &opening_point,
            &mut prove_transcript,
        );

        // Test 1: Tamper with the evaluation
        {
            let tampered_evaluation = Fr::rand(&mut rng);

            let mut verify_transcript =
                KeccakTranscript::new(DoryCommitmentScheme::<KeccakTranscript>::protocol_name());
            let result = DoryCommitmentScheme::<KeccakTranscript>::verify(
                &proof,
                &setup,
                &mut verify_transcript,
                &opening_point,
                &tampered_evaluation,
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
            let tampered_opening_point: Vec<Fr> =
                (0..num_vars).map(|_| Fr::rand(&mut rng)).collect();

            let mut verify_transcript =
                KeccakTranscript::new(DoryCommitmentScheme::<KeccakTranscript>::protocol_name());
            let result = DoryCommitmentScheme::<KeccakTranscript>::verify(
                &proof,
                &setup,
                &mut verify_transcript,
                &tampered_opening_point,
                &correct_evaluation,
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

            let mut verify_transcript =
                KeccakTranscript::new(DoryCommitmentScheme::<KeccakTranscript>::protocol_name());
            let result = DoryCommitmentScheme::<KeccakTranscript>::verify(
                &proof,
                &setup,
                &mut verify_transcript,
                &opening_point,
                &correct_evaluation,
                &wrong_commitment,
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
                &correct_evaluation,
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
            let mut verify_transcript =
                KeccakTranscript::new(DoryCommitmentScheme::<KeccakTranscript>::protocol_name());
            let result = DoryCommitmentScheme::<KeccakTranscript>::verify(
                &proof,
                &setup,
                &mut verify_transcript,
                &opening_point,
                &correct_evaluation,
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
