//! Implements the Jolt CommitmentScheme trait for Dory as a light wrapper
use super::commitment_scheme::CommitmentScheme;
use crate::{
    field::JoltField,
    poly::multilinear_polynomial::MultilinearPolynomial,
    utils::{errors::ProofVerifyError, transcript::Transcript},
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::rand::RngCore;
use std::borrow::Borrow;
use std::fmt::Debug;
use std::marker::PhantomData;

use ark_bn254::Fr;
use ark_std::rand::thread_rng;
use dory::{
    arithmetic::{Field, MultiScalarMul, Pairing},
    commit,
    curve::{ArkBn254Pairing, DummyMsm, OptimizedMsmG1, OptimizedMsmG2},
    evaluate, setup,
    transcript::Transcript as DoryTranscript,
    verify, DoryProof as DoryProofData, DoryProofBuilder, ProverSetup, VerifierSetup,
};

/// Newtype wrapper that adapts any JoltField to Dory's Field trait
#[derive(Clone, Copy, PartialEq, Eq, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct JoltToDoryField<F: JoltField>(pub F);
impl<F: JoltField> JoltToDoryField<F> {
    pub fn new(value: F) -> Self {
        Self(value)
    }

    pub fn inner(&self) -> &F {
        &self.0
    }

    pub fn into_inner(self) -> F {
        self.0
    }
}

impl<F: JoltField> Field for JoltToDoryField<F> {
    fn zero() -> Self {
        JoltToDoryField(F::zero())
    }

    fn one() -> Self {
        JoltToDoryField(F::one())
    }

    fn add(&self, rhs: &Self) -> Self {
        JoltToDoryField(self.0 + rhs.0)
    }

    fn sub(&self, rhs: &Self) -> Self {
        JoltToDoryField(self.0 - rhs.0)
    }

    fn mul(&self, rhs: &Self) -> Self {
        JoltToDoryField(self.0 * rhs.0)
    }

    fn inv(&self) -> Option<Self> {
        self.0.inverse().map(JoltToDoryField)
    }

    fn random<R: RngCore>(rng: &mut R) -> Self {
        JoltToDoryField(F::random(rng))
    }
    
    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}


// Helper functions to convert collections
impl<F: JoltField> JoltToDoryField<F> {
    /// Convert a slice of JoltField elements to wrapped elements
    pub fn wrap_slice(slice: &[F]) -> Vec<Self> {
        slice.iter().copied().map(Self).collect()
    }

    /// Convert a slice of wrapped elements back to JoltField elements
    pub fn unwrap_slice(slice: &[Self]) -> Vec<F> {
        slice.iter().map(|w| w.0).collect()
    }
}

// Conversion from Fr to JoltToDoryField<Fr> 
impl From<Fr> for JoltToDoryField<Fr> {
    fn from(value: Fr) -> Self {
        JoltToDoryField(value)
    }
}

// Conversion from JoltToDoryField<Fr> to Fr
impl From<JoltToDoryField<Fr>> for Fr {
    fn from(value: JoltToDoryField<Fr>) -> Self {
        value.0
    }
}


/// Newtype wrapper adapting Jolt Transcript to Dory Transcript
/// We use this Option mut ref thing so that we can avoid cloning in Prove and Verify.
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
    type Scalar = JoltToDoryField<F>;

    fn append_bytes(&mut self, label: &[u8], bytes: &[u8]) {
        let transcript = self.transcript.as_mut().expect("Transcript not initialized");
        let label_str: &'static [u8] = Box::leak(label.to_vec().into_boxed_slice());
        transcript.append_message(label_str);
        transcript.append_bytes(bytes);
    }

    fn append_field(&mut self, label: &[u8], x: &Self::Scalar) {
        let transcript = self.transcript.as_mut().expect("Transcript not initialized");
        let label_str: &'static [u8] = Box::leak(label.to_vec().into_boxed_slice());
        transcript.append_message(label_str);
        transcript.append_serializable(&x.0);
    }

    fn append_group<G: CanonicalSerialize>(&mut self, label: &[u8], g: &G) {
        let transcript = self.transcript.as_mut().expect("Transcript not initialized");
        let label_str: &'static [u8] = Box::leak(label.to_vec().into_boxed_slice());
        transcript.append_message(label_str);
        transcript.append_serializable(g);
    }

    fn append_serde<S: serde::Serialize>(&mut self, label: &[u8], s: &S) {
        let transcript = self.transcript.as_mut().expect("Transcript not initialized");
        let label_str: &'static [u8] = Box::leak(label.to_vec().into_boxed_slice());
        transcript.append_message(label_str);
        let bytes = postcard::to_allocvec(s).unwrap_or_default();
        transcript.append_bytes(&bytes);
    }

    fn challenge_scalar(&mut self, label: &[u8]) -> Self::Scalar {
        let transcript = self.transcript.as_mut().expect("Transcript not initialized");
        let label_str: &'static [u8] = Box::leak(label.to_vec().into_boxed_slice());
        transcript.append_message(label_str);
        let challenge: F = transcript.challenge_scalar::<F>();
        JoltToDoryField(challenge)
    }

    fn reset(&mut self, domain_label: &[u8]) {
        let _ = domain_label;
        panic!("We do not want to ever reset JoltToDoryTranscript.")
        }
}

// Helper function to convert multilinear polynomial to field coefficients
// This can be non-zero cost for large polys so will want a better solution, eventually
fn extract_field_coefficients<F: JoltField>(poly: &MultilinearPolynomial<F>) -> Vec<F> {
    match poly {
        MultilinearPolynomial::LargeScalars(dense_poly) => dense_poly.Z.clone(),
        MultilinearPolynomial::U8Scalars(compact_poly) => {
            compact_poly.coeffs.iter().map(|&c| F::from_u8(c)).collect()
        }
        MultilinearPolynomial::U16Scalars(compact_poly) => compact_poly
            .coeffs
            .iter()
            .map(|&c| F::from_u16(c))
            .collect(),
        MultilinearPolynomial::U32Scalars(compact_poly) => compact_poly
            .coeffs
            .iter()
            .map(|&c| F::from_u32(c))
            .collect(),
        MultilinearPolynomial::U64Scalars(compact_poly) => compact_poly
            .coeffs
            .iter()
            .map(|&c| F::from_u64(c))
            .collect(),
        MultilinearPolynomial::I64Scalars(compact_poly) => compact_poly
            .coeffs
            .iter()
            .map(|&c| F::from_i64(c))
            .collect(),
    }
}

/// Dory commitment scheme implementation
#[derive(Clone, Debug)]
pub struct DoryCommitmentScheme<F: JoltField, ProofTranscript: Transcript, P: Pairing> {
    _phantom_field: PhantomData<F>,
    _phantom_transcript: PhantomData<ProofTranscript>,
    _phantom_pairing: PhantomData<P>,
}

/// Setup structure containing both prover and verifier parameters.
/// Dory API exposes an SRS to disk option, but is not (yet) utilized here.
#[derive(Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct DorySetup<P>
where
    P: Pairing + Debug,
{
    pub prover_setup: ProverSetup<P>,
    pub verifier_setup: VerifierSetup<P>,
    pub max_log_n: usize,
}

/// Commitment structure
#[derive(Clone, Debug, Default, PartialEq, CanonicalSerialize, CanonicalDeserialize)]
pub struct DoryCommitment<P>
where
    P: Pairing,
    P::GT: Clone + Debug + Default + PartialEq,
{
    pub commitment: P::GT,
}

/// Proof structure storing the serializable DoryProof data and other data
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct DoryProof<F: JoltField, P>
where
    P: Pairing,
    P::G1: Debug,
    P::G2: Debug,
    P::GT: Debug,
{
    pub evaluation: F,
    pub opening_point: Vec<F>,
    pub sigma: usize,
    // Store the serializable DoryProof from the DoryProofBuilder
    pub dory_proof_data: DoryProofData<P::G1, P::G2, P::GT>,
    _phantom: PhantomData<F>,
}

/// Batched proof structure
/// @TODO: this is not yet fully implemented.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct DoryBatchedProof<F: JoltField, P>
where
    P: Pairing,
    P::G1: Debug,
    P::G2: Debug,
    P::GT: Debug,
{
    proofs: Vec<DoryProof<F, P>>,
}

impl<F, ProofTranscript, P> CommitmentScheme<ProofTranscript>
    for DoryCommitmentScheme<F, ProofTranscript, P>
where
    F: JoltField,
    ProofTranscript: Transcript,
    P: Pairing + Clone + 'static + Debug + Default + PartialEq,
    P::G1: dory::arithmetic::Group<Scalar = JoltToDoryField<F>> + Debug,
    P::G2: dory::arithmetic::Group<Scalar = JoltToDoryField<F>> + Debug,
    P::GT: dory::arithmetic::Group<Scalar = JoltToDoryField<F>> + Clone + Debug + Default,
    OptimizedMsmG1: MultiScalarMul<P::G1>,
    OptimizedMsmG2: MultiScalarMul<P::G2>,
    DummyMsm<P::GT>: MultiScalarMul<P::GT>,
{
    type Field = F;
    type Setup = DorySetup<P>;
    type Commitment = DoryCommitment<P>;
    type Proof = DoryProof<F, P>;
    type BatchedProof = DoryBatchedProof<F, P>;

    // @TODO: we use `max_num_vars`, but others might not. We want other PCS to use this instead of `max_len`.
    fn setup(max_num_vars: usize) -> Self::Setup {
        let (prover_setup, verifier_setup) = setup::<P, _>(thread_rng(), max_num_vars);

        DorySetup {
            prover_setup,
            verifier_setup,
            max_log_n: max_num_vars,
        }
    }

    fn commit(poly: &MultilinearPolynomial<Self::Field>, setup: &Self::Setup) -> Self::Commitment {
        // Extract polynomial coefficients
        let field_coeffs = extract_field_coefficients(poly);

        // Convert JoltField coefficients to JoltToDoryField
        let coeffs = JoltToDoryField::wrap_slice(&field_coeffs);

        // For Dory, sigma is the number of columns in the matrix representation
        // we use sigma = ceil(num_vars / 2) so we have a nice square matrix.
        let num_vars = poly.get_num_vars();
        let sigma = (num_vars + 1) / 2;
        let commitment =
            commit::<P, OptimizedMsmG1>(&coeffs, 0, sigma, &setup.prover_setup);

        DoryCommitment { commitment }
    }

    // @TODO: Implement linear combination batching
    fn batch_commit<U>(polys: &[U], setup: &Self::Setup) -> Vec<Self::Commitment>
    where
        U: Borrow<MultilinearPolynomial<Self::Field>> + Sync,
    {
        todo!()
    }

    // Note that Dory implementation sometimes uses the term 'evaluation'/'evaluate' -- this is same as 'opening'/'open'
    fn prove<'a>(
        setup: &Self::Setup,
        poly: &MultilinearPolynomial<Self::Field>,
        opening_point: &[Self::Field],
        transcript: &'a mut ProofTranscript,
    ) -> Self::Proof {
        // Extract polynomial coefficients and convert to JoltToDoryField
        let field_coeffs = extract_field_coefficients(poly);
        let coeffs = JoltToDoryField::wrap_slice(&field_coeffs);

        // Convert opening point to JoltToDoryField
        let point = JoltToDoryField::wrap_slice(opening_point);

        let num_vars = poly.get_num_vars();
        let sigma = (num_vars + 1) / 2;

        let dory_transcript_prover = JoltToDoryTranscriptRef::<Self::Field, ProofTranscript>::new(transcript);

        let (evaluation, proof_builder) =
            evaluate::<P, _, OptimizedMsmG1, OptimizedMsmG2>(
                &coeffs,
                &point,
                sigma,
                &setup.prover_setup,
                dory_transcript_prover,
            );

        // Extract from the builder into a serializable form
        let dory_proof = proof_builder.build();

        DoryProof {
            evaluation: evaluation.into_inner(),
            opening_point: JoltToDoryField::unwrap_slice(&point),
            sigma,
            dory_proof_data: dory_proof,
            _phantom: PhantomData,
        }
    }

    fn verify<'a>(
        proof: &Self::Proof,
        setup: &Self::Setup,
        transcript: &'a mut ProofTranscript,
        opening_point: &[Self::Field],
        _opening: &Self::Field, // we use the opening directly from Dory's Proof object.
        commitment: &Self::Commitment,
    ) -> Result<(), ProofVerifyError> {
        // Convert opening point to JoltToDoryField
        let point = JoltToDoryField::wrap_slice(opening_point);

        // Create JoltToDoryTranscript wrapper around the provided transcript
        let dory_transcript = JoltToDoryTranscriptRef::<Self::Field, ProofTranscript>::new(transcript);

        // Create a proof builder from the DoryProof
        let verifier_proof_builder =
            DoryProofBuilder::from_proof_no_transcript(proof.dory_proof_data.clone());

        // Use Dory's verify function with the proof builder
        let verify_result =
            verify::<P, _, OptimizedMsmG1, OptimizedMsmG2, DummyMsm<P::GT>>(
                commitment.commitment.clone(),
                JoltToDoryField::new(proof.evaluation),
                &point,
                verifier_proof_builder,
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

// Implement AppendToTranscript for DoryCommitment
impl<P> crate::utils::transcript::AppendToTranscript for DoryCommitment<P>
where
    P: Pairing,
    P::GT: CanonicalSerialize + Clone + Debug + Default + PartialEq,
{
    fn append_to_transcript<ProofTranscript: Transcript>(&self, transcript: &mut ProofTranscript) {
        transcript.append_serializable(&self.commitment);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poly::dense_mlpoly::DensePolynomial;
    use crate::utils::transcript::KeccakTranscript;
    use ark_bn254::Fr;
    use ark_std::rand::thread_rng;

    #[test]
    fn test_dory_commitment_scheme() {
        use ark_std::UniformRand;
        use std::time::Instant;

        // Create a random polynomial and other related preliminaries
        let max_log_n = 18; // This will support polynomials up to 2^18 coefficients
        let num_vars = 18;
        let num_coeffs = 1 << num_vars;
        let sigma = (num_vars + 1) / 2;

        println!(
            "Testing Dory PCS with {} variables, {} coefficients, sigma = {}",
            num_vars, num_coeffs, sigma
        );

        let mut rng = thread_rng();
        let coeffs: Vec<Fr> = (0..num_coeffs).map(|_| (Fr::rand(&mut rng))).collect();

        let poly = MultilinearPolynomial::LargeScalars(DensePolynomial::new(coeffs.clone()));

        // Generate random opening point
        let opening_point: Vec<JoltToDoryField<Fr>> = (0..num_vars).map(|_| JoltToDoryField::new(Fr::rand(&mut rng))).collect();

        // Setup timing
        let setup_start = Instant::now();

        // the commitment key is actually size sqrt(max_log_n) = 9 here
        let setup = DoryCommitmentScheme::<Fr, KeccakTranscript, ArkBn254Pairing>::setup(max_log_n);
        let setup_time = setup_start.elapsed();
        println!("Setup time: {:?}", setup_time);

        // Commit timing
        let commit_start = Instant::now();
        let commitment = DoryCommitmentScheme::<Fr, KeccakTranscript, ArkBn254Pairing>::commit(&poly, &setup);
        let commit_time = commit_start.elapsed();
        println!("Commit time: {:?}", commit_time);

        // Note that domains initially need to be the same for transcripts.
        let mut prove_transcript = KeccakTranscript::new(b"dory_test");

        // Prove timing
        let prove_start = Instant::now();
        let proof = DoryCommitmentScheme::<Fr, KeccakTranscript, ArkBn254Pairing>::prove(
            &setup,
            &poly,
            &opening_point,
            &mut prove_transcript,
        );
        let prove_time = prove_start.elapsed();
        println!("Prove time: {:?}", prove_time);

        let mut verify_transcript = KeccakTranscript::new(b"dory_test");

        // Verify timing
        let verify_start = Instant::now();
        let verification_result = DoryCommitmentScheme::<Fr, KeccakTranscript, ArkBn254Pairing>::verify(
            &proof,
            &setup,
            &mut verify_transcript,
            &opening_point,
            &proof.evaluation,
            &commitment,
        );
        let verify_time = verify_start.elapsed();
        println!("Verify time: {:?}", verify_time);

        let total_time = setup_time + commit_time + prove_time + verify_time;
        println!("Total time: {:?}", total_time);

        // The verification should succeed
        assert!(
            verification_result.is_ok(),
            "Dory verification failed: {:?}",
            verification_result
        );

        println!("✅ Dory commitment scheme test passed!");
        println!(
            "   - Polynomial size: 2^{} = {} coefficients",
            num_vars, num_coeffs
        );
        println!("   - Commitment: {:?}", commitment.commitment);
        println!("   - Evaluation: {:?}", proof.evaluation);
    }

    #[test]
    fn test_dory_soundness() {
        use ark_std::UniformRand;

        // Test setup
        let num_vars = 10;
        let max_log_n = 10;
        let num_coeffs = 1 << num_vars;

        let mut rng = thread_rng();
        let coeffs: Vec<Fr> = (0..num_coeffs).map(|_| Fr::rand(&mut rng)).collect();
        let poly = MultilinearPolynomial::LargeScalars(DensePolynomial::new(coeffs.clone()));

        let opening_point: Vec<Fr> = (0..num_vars).map(|_| Fr::rand(&mut rng)).collect();

        let setup = DoryCommitmentScheme::<JoltToDoryField<Fr>, KeccakTranscript, ArkBn254Pairing>::setup(max_log_n);

        // Commit to the polynomial
        let commitment = DoryCommitmentScheme::<JoltToDoryField<Fr>, KeccakTranscript, ArkBn254Pairing>::commit(&poly, &setup);

        let mut prove_transcript =
            KeccakTranscript::new(DoryCommitmentScheme::<Fr, KeccakTranscript, ArkBn254Pairing>::protocol_name());

        // Generate the proof
        let proof = DoryCommitmentScheme::<Fr, KeccakTranscript, ArkBn254Pairing>::prove(
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

            let mut verify_transcript = KeccakTranscript::new(DoryCommitmentScheme::<
                Fr,
                KeccakTranscript,
                ArkBn254Pairing,
            >::protocol_name());
            let result = DoryCommitmentScheme::<Fr, KeccakTranscript, ArkBn254Pairing>::verify(
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
            let tampered_opening_point: Vec<Fr> =
                (0..num_vars).map(|_| Fr::rand(&mut rng)).collect();

            let mut verify_transcript = KeccakTranscript::new(DoryCommitmentScheme::<
                Fr,
                KeccakTranscript,
                ArkBn254Pairing,
            >::protocol_name());
            let result = DoryCommitmentScheme::<Fr, KeccakTranscript, ArkBn254Pairing>::verify(
                &proof,
                &setup,
                &mut verify_transcript,
                &tampered_opening_point, // Wrong opening point
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
                DoryCommitmentScheme::<Fr, KeccakTranscript>::commit(&wrong_poly, &setup);

            let mut verify_transcript = KeccakTranscript::new(DoryCommitmentScheme::<
                Fr,
                KeccakTranscript,
                ArkBn254Pairing,
            >::protocol_name());
            let result = DoryCommitmentScheme::<Fr, KeccakTranscript, ArkBn254Pairing>::verify(
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
            let result = DoryCommitmentScheme::<Fr, KeccakTranscript, ArkBn254Pairing>::verify(
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
            let mut verify_transcript = KeccakTranscript::new(DoryCommitmentScheme::<
                Fr,
                KeccakTranscript,
                ArkBn254Pairing,
            >::protocol_name());
            let result = DoryCommitmentScheme::<Fr, KeccakTranscript, ArkBn254Pairing>::verify(
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
