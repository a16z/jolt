//! Implements the Jolt CommitmentScheme trait for Dory as a light wrapper
//! This is (currently) coupled to the use of arkworks bn254
//! That is, the underlying wrappers expect Fr, G1, G2 from arkworks -- Undefined Behavior otherwise.
//! This can be changed in the future with some NewType refactoring.
//!
//! TODOs:
//! 1. think about  Fr <> poly coefficient optimizations
//! 2. batching (batching here means multiple poly evals over single point)
use super::commitment_scheme::CommitmentScheme;
use crate::{
    field::JoltField,
    msm::VariableBaseMSM,
    poly::multilinear_polynomial::MultilinearPolynomial,
    utils::{errors::ProofVerifyError, transcript::Transcript},
};
use ark_ec::{pairing::PairingOutput, PrimeGroup};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use std::borrow::Borrow;
use std::marker::PhantomData;

use ark_bn254::{Bn254, Fq12, Fr, G1Affine, G1Projective};
use ark_std::rand::thread_rng;
use dory::arithmetic::{Field as DoryField, Group as DoryGroup};
use dory::{
    commit,
    curve::{ArkBn254Pairing, DummyMsm, OptimizedMsmG1, OptimizedMsmG2},
    evaluate, setup,
    transcript::Transcript as DoryTranscript,
    verify, DoryProof as DoryProofData, DoryProofBuilder, Polynomial as DoryPolynomial,
    ProverSetup, VerifierSetup,
};

/// Newtype wrapper adapting Jolt Transcript to Dory Transcript
///
/// # Safety Requirements
///
/// This implementation uses unsafe transmutes to convert `&[u8]` to `&'static [u8]`.
/// @TODO(markosg04): better solution than unsafe usage?
/// @TODO(markosg04): We also don't need labels appended, but for now it doesn't hurt.
/// This is only safe if:
/// 1. All labels passed to this transcript outlive the transcript itself
/// 2. The transcript is not used after the labels are deallocated
/// 3. The labels are typically string literals or other actually-static data
///
/// Using this with dynamically allocated labels is UNDEFINED BEHAVIOR.

#[derive(Clone)]
pub struct JoltToDoryTranscript<T: Transcript>(pub T);

impl<T: Transcript> JoltToDoryTranscript<T> {
    pub fn new(transcript: T) -> Self {
        Self(transcript)
    }

    /// # Safety
    /// The caller must ensure that `label` outlives this transcript
    unsafe fn transmute_label(label: &[u8]) -> &'static [u8] {
        std::mem::transmute(label)
    }
}

impl<T: Transcript> DoryTranscript for JoltToDoryTranscript<T> {
    type Scalar = Fr;

    fn append_bytes(&mut self, label: &[u8], bytes: &[u8]) {
        let label_str = unsafe { Self::transmute_label(label) };
        self.0.append_message(label_str);
        self.0.append_bytes(bytes);
    }

    fn append_field(&mut self, label: &[u8], x: &Self::Scalar) {
        let label_str = unsafe { Self::transmute_label(label) };
        self.0.append_message(label_str);
        self.0.append_serializable(x);
    }

    fn append_group<G: CanonicalSerialize>(&mut self, label: &[u8], g: &G) {
        let label_str = unsafe { Self::transmute_label(label) };
        self.0.append_message(label_str);
        self.0.append_serializable(g);
    }

    fn append_serde<S: serde::Serialize>(&mut self, label: &[u8], s: &S) {
        let label_str = unsafe { Self::transmute_label(label) };
        self.0.append_message(label_str);
        let bytes = postcard::to_allocvec(s).unwrap_or_default();
        self.0.append_bytes(&bytes);
    }

    fn challenge_scalar(&mut self, label: &[u8]) -> Self::Scalar {
        let label_str = unsafe { Self::transmute_label(label) };
        self.0.append_message(label_str);
        self.0.challenge_scalar::<Fr>()
    }

    fn reset(&mut self, domain_label: &[u8]) {
        let label_str = unsafe { Self::transmute_label(domain_label) };
        self.0 = T::new(label_str);
    }
}

// Helper function to convert multilinear polynomial to field coefficients
// This can be non-zero cost for large polys so will want a better solution, eventually
// For now this is needed since we want arkworks Fr
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
pub struct DoryCommitmentScheme<F: JoltField, ProofTranscript: Transcript> {
    _phantom_field: PhantomData<F>,
    _phantom_transcript: PhantomData<ProofTranscript>,
}

/// Setup structure containing both prover and verifier parameters.
/// Dory API exposes an SRS to disk option, but is not (yet) utilized here.
#[derive(Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct DorySetup {
    pub prover_setup: ProverSetup<ArkBn254Pairing>,
    pub verifier_setup: VerifierSetup<ArkBn254Pairing>,
    pub max_log_n: usize,
}

/// Commitment structure
#[derive(Clone, Debug, Default, PartialEq, CanonicalSerialize, CanonicalDeserialize)]
pub struct DoryCommitment {
    pub commitment: PairingOutput<Bn254>, // this is GT
}

/// Proof structure storing the serializable DoryProof data and other data
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct DoryProof<F: JoltField> {
    pub evaluation: Fr,
    pub opening_point: Vec<Fr>,
    pub sigma: usize,
    // Store the serializable DoryProof from the DoryProofBuilder
    pub dory_proof_data: DoryProofData<ark_bn254::G1Affine, dory::curve::G2AffineWrapper, Fq12>,
    _phantom: PhantomData<F>,
}

/// Batched proof structure
/// @TODO: this is not yet fully implemented.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct DoryBatchedProof<F: JoltField> {
    proofs: Vec<DoryProof<F>>,
}

impl<F, ProofTranscript> CommitmentScheme<ProofTranscript>
    for DoryCommitmentScheme<F, ProofTranscript>
where
    F: JoltField,
    ProofTranscript: Transcript,
{
    type Field = Fr;
    type Setup = DorySetup;
    type Commitment = DoryCommitment;
    type Proof = DoryProof<F>;
    type BatchedProof = DoryBatchedProof<F>;

    fn setup(max_log_n: usize) -> Self::Setup {
        let (prover_setup, verifier_setup) = setup::<ArkBn254Pairing, _>(thread_rng(), max_log_n);

        DorySetup {
            prover_setup,
            verifier_setup,
            max_log_n,
        }
    }

    #[tracing::instrument(skip_all, name = "DoryCommitmentScheme::commit")]
    fn commit(poly: &MultilinearPolynomial<Self::Field>, setup: &Self::Setup) -> Self::Commitment {
        // // Extract polynomial coefficients
        // let field_coeffs = extract_field_coefficients(poly);

        // // Convert JoltField coefficients to Fr
        // // This assumes F is Fr or can be safely transmuted to Fr
        // let coeffs: Vec<Fr> = field_coeffs
        //     .iter()
        //     .map(|&coeff| unsafe { std::mem::transmute_copy(&coeff) })
        //     .collect();

        // For Dory, sigma is the number of columns in the matrix representation
        // we use sigma = ceil(num_vars / 2) so we have a nice square matrix.
        let num_vars = poly.get_num_vars();
        let sigma = (num_vars + 1) / 2;
        let commitment = commit::<ArkBn254Pairing, _, _, Fr, OptimizedMsmG1, _>(
            poly,
            0,
            sigma,
            &setup.prover_setup,
        );

        DoryCommitment {
            commitment: PairingOutput(commitment),
        }
    }

    // @TODO: Implement linear combination batching
    fn batch_commit<U>(_polys: &[U], _setup: &Self::Setup) -> Vec<Self::Commitment>
    where
        U: Borrow<MultilinearPolynomial<Self::Field>> + Sync,
    {
        todo!()
    }

    // Note that Dory implementation sometimes uses the term 'evaluation'/'evaluate' -- this is same as 'opening'/'open'
    #[tracing::instrument(skip_all, name = "DoryCommitmentScheme::prove")]
    fn prove(
        setup: &Self::Setup,
        poly: &MultilinearPolynomial<Self::Field>,
        opening_point: &[Self::Field],
        transcript: &mut ProofTranscript,
    ) -> Self::Proof {
        // extract as JoltField transmutable to arkworks Fr
        let field_coeffs = extract_field_coefficients(poly);

        // Convert to arkworks Fr
        let coeffs: Vec<Fr> = field_coeffs
            .iter()
            .map(|&coeff| unsafe { std::mem::transmute_copy(&coeff) })
            .collect();

        // Convert opening point to arkworks Fr
        let point: Vec<Fr> = opening_point
            .iter()
            .map(|&p| unsafe { std::mem::transmute_copy(&p) })
            .collect();

        let num_vars = poly.get_num_vars();
        let sigma = (num_vars + 1) / 2;

        let dory_transcript_prover = JoltToDoryTranscript::new(transcript.clone());

        let (evaluation, proof_builder) =
            evaluate::<ArkBn254Pairing, _, OptimizedMsmG1, OptimizedMsmG2>(
                &coeffs,
                &point,
                sigma,
                &setup.prover_setup,
                dory_transcript_prover,
            );

        // Extract from the builder into a serializable form
        let dory_proof = proof_builder.build();

        DoryProof {
            evaluation,
            opening_point: point,
            sigma,
            dory_proof_data: dory_proof,
            _phantom: PhantomData,
        }
    }

    fn verify(
        proof: &Self::Proof,
        setup: &Self::Setup,
        transcript: &mut ProofTranscript,
        opening_point: &[Self::Field],
        _opening: &Self::Field, // we use the opening directly from Dory's Proof object.
        commitment: &Self::Commitment,
    ) -> Result<(), ProofVerifyError> {
        // Convert opening point to arkworks Fr
        let point: Vec<Fr> = opening_point
            .iter()
            .map(|&p| unsafe { std::mem::transmute_copy(&p) })
            .collect();

        // Create JoltToDoryTranscript wrapper around the provided transcript
        let dory_transcript = JoltToDoryTranscript::new(transcript.clone());

        // Create a proof builder from the DoryProof
        let verifier_builder =
            DoryProofBuilder::from_proof(proof.dory_proof_data.clone(), dory_transcript.clone());

        // Use Dory's verify function with the proof builder
        let verify_result =
            verify::<ArkBn254Pairing, _, OptimizedMsmG1, OptimizedMsmG2, DummyMsm<Fq12>>(
                commitment.commitment.0,
                proof.evaluation,
                &point,
                verifier_builder,
                proof.sigma,
                &setup.verifier_setup,
                dory_transcript,
            );

        match verify_result {
            Ok(()) => Ok(()),
            Err(_) => Err(ProofVerifyError::InternalError),
        }
    }

    fn combine_commitments(
        commitments: &[&Self::Commitment],
        coeffs: &[Self::Field],
    ) -> Self::Commitment {
        let combined_commitment: PairingOutput<Bn254> = commitments
            .iter()
            .zip(coeffs.iter())
            .map(|(commitment, coeff)| commitment.commitment * coeff)
            .sum();
        DoryCommitment {
            commitment: combined_commitment,
        }
    }

    fn protocol_name() -> &'static [u8] {
        b"dory_commitment_scheme"
    }
}

impl<F, G1> DoryPolynomial<F, G1> for MultilinearPolynomial<F>
where
    F: JoltField + DoryField,
    G1: DoryGroup<Scalar = F> + VariableBaseMSM,
    <G1 as PrimeGroup>::ScalarField: JoltField,
{
    fn len(&self) -> usize {
        todo!()
    }

    fn evaluate(&self, point: &[F]) -> F {
        todo!()
    }

    fn commit_rows(&self, g1_generators: &[G1], row_len: usize) -> Vec<G1> {
        todo!()
    }

    fn vector_matrix_product(&self, l_vec: &[F]) -> Vec<F> {
        todo!()
    }
}

// Implement AppendToTranscript for DoryCommitment
impl crate::utils::transcript::AppendToTranscript for DoryCommitment {
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
        let max_log_n = 22; // This will support polynomials up to 2^22 coefficients
        let num_vars = 22;
        let num_coeffs = 1 << num_vars;
        let sigma = (num_vars + 1) / 2;

        println!(
            "Testing Dory PCS with {} variables, {} coefficients, sigma = {}",
            num_vars, num_coeffs, sigma
        );

        let mut rng = thread_rng();
        let coeffs: Vec<Fr> = (0..num_coeffs).map(|_| Fr::rand(&mut rng)).collect();

        let poly = MultilinearPolynomial::LargeScalars(DensePolynomial::new(coeffs.clone()));

        // Generate random opening point
        let opening_point: Vec<Fr> = (0..num_vars).map(|_| Fr::rand(&mut rng)).collect();

        // Setup timing
        let setup_start = Instant::now();

        // the commitment key is actually size sqrt(max_log_n) = 11 here
        let setup = DoryCommitmentScheme::<Fr, KeccakTranscript>::setup(max_log_n);
        let setup_time = setup_start.elapsed();
        println!("Setup time: {:?}", setup_time);

        // Commit timing
        let commit_start = Instant::now();
        let commitment = DoryCommitmentScheme::<Fr, KeccakTranscript>::commit(&poly, &setup);
        let commit_time = commit_start.elapsed();
        println!("Commit time: {:?}", commit_time);

        // Note that domains initially need to be the same for transcripts.
        let mut prove_transcript = KeccakTranscript::new(b"dory_test");

        // Prove timing
        let prove_start = Instant::now();
        let proof = DoryCommitmentScheme::<Fr, KeccakTranscript>::prove(
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
        let verification_result = DoryCommitmentScheme::<Fr, KeccakTranscript>::verify(
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

        let setup = DoryCommitmentScheme::<Fr, KeccakTranscript>::setup(max_log_n);

        // Commit to the polynomial
        let commitment = DoryCommitmentScheme::<Fr, KeccakTranscript>::commit(&poly, &setup);

        let mut prove_transcript =
            KeccakTranscript::new(DoryCommitmentScheme::<Fr, KeccakTranscript>::protocol_name());

        // Generate the proof
        let proof = DoryCommitmentScheme::<Fr, KeccakTranscript>::prove(
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
            >::protocol_name());
            let result = DoryCommitmentScheme::<Fr, KeccakTranscript>::verify(
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
            >::protocol_name());
            let result = DoryCommitmentScheme::<Fr, KeccakTranscript>::verify(
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
            >::protocol_name());
            let result = DoryCommitmentScheme::<Fr, KeccakTranscript>::verify(
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
            let result = DoryCommitmentScheme::<Fr, KeccakTranscript>::verify(
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
            >::protocol_name());
            let result = DoryCommitmentScheme::<Fr, KeccakTranscript>::verify(
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
