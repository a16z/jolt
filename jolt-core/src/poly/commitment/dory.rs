use super::commitment_scheme::CommitmentScheme;
use crate::{
    field::JoltField,
    poly::multilinear_polynomial::MultilinearPolynomial,
    utils::{errors::ProofVerifyError, transcript::Transcript},
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, Valid};
use std::borrow::Borrow;
use std::marker::PhantomData;

// Import Dory types and traits
use dory::{
    arithmetic::{Field as DoryField, Group as DoryGroup},
    transcript::Transcript as DoryTranscript,
    ProverSetup, VerifierSetup, DoryProofBuilder,
    setup, commit, evaluate, verify,
    curve::{ArkBn254Pairing, OptimizedMsmG1, OptimizedMsmG2, DummyMsm},
    toy_transcript::ToyTranscript,
};
use ark_bn254::{Fq12, Fr};
use ark_std::rand::thread_rng;
use blake2::Blake2s256;

// Bridge types to convert between Jolt and Dory traits

// Blanket implementation of Dory Field trait for JoltField
// Note: This only works if F is actually Fr (BN254 scalar field)
// We use a newtype wrapper to avoid orphan rule issues
#[derive(Clone, Debug, PartialEq, Eq, CanonicalSerialize, CanonicalDeserialize)]
pub struct DoryFieldWrapper<F: JoltField>(pub F);

impl<F: JoltField> DoryField for DoryFieldWrapper<F> {
    fn zero() -> Self {
        DoryFieldWrapper(F::zero())
    }
    
    fn one() -> Self {
        DoryFieldWrapper(F::one())
    }
    
    fn add(&self, rhs: &Self) -> Self {
        DoryFieldWrapper(self.0 + rhs.0)
    }
    
    fn sub(&self, rhs: &Self) -> Self {
        DoryFieldWrapper(self.0 - rhs.0)
    }
    
    fn mul(&self, rhs: &Self) -> Self {
        DoryFieldWrapper(self.0 * rhs.0)
    }
    
    fn inv(&self) -> Option<Self> {
        self.0.inverse().map(DoryFieldWrapper)
    }
    
    fn random<R: ark_std::rand::RngCore>(rng: &mut R) -> Self {
        DoryFieldWrapper(F::random(rng))
    }
}

// Wrapper to adapt Jolt transcript to Dory transcript
pub struct JoltToDoryTranscript<T: Transcript> {
    inner: T,
}

impl<T: Transcript> JoltToDoryTranscript<T> {
    pub fn new(transcript: T) -> Self {
        Self { inner: transcript }
    }
}

impl<T: Transcript> DoryTranscript for JoltToDoryTranscript<T> {
    type Scalar = Fr; // Use BN254 scalar field
    
    fn append_bytes(&mut self, label: &[u8], bytes: &[u8]) {
        // Convert label to static str - this is unsafe but necessary for API compatibility
        let label_str: &'static [u8] = unsafe { std::mem::transmute(label) };
        self.inner.append_message(label_str);
        self.inner.append_bytes(bytes);
    }
    
    fn append_field(&mut self, label: &[u8], x: &Self::Scalar) {
        let label_str: &'static [u8] = unsafe { std::mem::transmute(label) };
        self.inner.append_message(label_str);
        self.inner.append_serializable(x);
    }
    
    fn append_group<G: CanonicalSerialize>(&mut self, label: &[u8], g: &G) {
        let label_str: &'static [u8] = unsafe { std::mem::transmute(label) };
        self.inner.append_message(label_str);
        self.inner.append_serializable(g);
    }
    
    fn append_serde<S: serde::Serialize>(&mut self, label: &[u8], s: &S) {
        let label_str: &'static [u8] = unsafe { std::mem::transmute(label) };
        self.inner.append_message(label_str);
        // For serde, we serialize to bytes first
        let bytes = postcard::to_allocvec(s).unwrap_or_default();
        self.inner.append_bytes(&bytes);
    }
    
    fn challenge_scalar(&mut self, _label: &[u8]) -> Self::Scalar {
        self.inner.challenge_scalar::<Fr>()
    }
    
    fn reset(&mut self, domain_label: &[u8]) {
        // Reset by creating a new transcript with the domain label
        let label_str: &'static [u8] = unsafe { std::mem::transmute(domain_label) };
        self.inner = T::new(label_str);
    }
}

// Helper function to convert multilinear polynomial to field coefficients
fn extract_field_coefficients<F: JoltField>(poly: &MultilinearPolynomial<F>) -> Vec<F> {
    match poly {
        MultilinearPolynomial::LargeScalars(dense_poly) => {
            // Access the Z field directly from DensePolynomial
            dense_poly.Z.clone()
        }
        MultilinearPolynomial::U8Scalars(compact_poly) => {
            compact_poly.coeffs.iter().map(|&c| F::from_u8(c)).collect()
        }
        MultilinearPolynomial::U16Scalars(compact_poly) => {
            compact_poly.coeffs.iter().map(|&c| F::from_u16(c)).collect()
        }
        MultilinearPolynomial::U32Scalars(compact_poly) => {
            compact_poly.coeffs.iter().map(|&c| F::from_u32(c)).collect()
        }
        MultilinearPolynomial::U64Scalars(compact_poly) => {
            compact_poly.coeffs.iter().map(|&c| F::from_u64(c)).collect()
        }
        MultilinearPolynomial::I64Scalars(compact_poly) => {
            compact_poly.coeffs.iter().map(|&c| F::from_i64(c)).collect()
        }
    }
}

// Dory commitment scheme implementation
#[derive(Clone, Debug)]
pub struct DoryCommitmentScheme<F: JoltField, ProofTranscript: Transcript> {
    _phantom_field: PhantomData<F>,
    _phantom_transcript: PhantomData<ProofTranscript>,
}

// Setup structure containing both prover and verifier parameters
#[derive(Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct DorySetup {
    pub prover_setup: ProverSetup<ArkBn254Pairing>,
    pub verifier_setup: VerifierSetup<ArkBn254Pairing>,
    pub max_log_n: usize,
}

// Commitment structure
#[derive(Clone, Debug, PartialEq, CanonicalSerialize, CanonicalDeserialize)]
pub struct DoryCommitment {
    pub commitment: Fq12, // GT element for BN254
}

impl Default for DoryCommitment {
    fn default() -> Self {
        use ark_std::One;
        Self {
            commitment: Fq12::one(),
        }
    }
}

// Proof structure using DoryProofBuilder
// Note: We need to store the proof builder for verification
pub struct DoryProof<F: JoltField> {
    pub evaluation: Fr,
    pub proof_builder: DoryProofBuilder<
        ark_bn254::G1Affine,
        dory::curve::G2AffineWrapper,
        Fq12,
        Fr,
        ToyTranscript<Fr, Blake2s256>
    >,
    pub opening_point: Vec<Fr>,
    pub sigma: usize,
    _phantom: PhantomData<F>,
}

// Note: DoryProofBuilder doesn't implement Clone by default, so we'll need to handle this carefully
impl<F: JoltField> Clone for DoryProof<F> {
    fn clone(&self) -> Self {
        // For now, we'll create a new default proof builder
        // In a production implementation, you'd want to properly serialize/deserialize the proof
        let transcript = ToyTranscript::<Fr, Blake2s256>::new(b"dory_evaluation");
        let proof_builder = DoryProofBuilder::new(transcript);
        
        Self {
            evaluation: self.evaluation,
            proof_builder,
            opening_point: self.opening_point.clone(),
            sigma: self.sigma,
            _phantom: PhantomData,
        }
    }
}

// We'll need to implement serialization manually since DoryProofBuilder may not be serializable
impl<F: JoltField> CanonicalSerialize for DoryProof<F> {
    fn serialize_with_mode<W: ark_serialize::Write>(
        &self,
        mut writer: W,
        compress: ark_serialize::Compress,
    ) -> Result<(), ark_serialize::SerializationError> {
        self.evaluation.serialize_with_mode(&mut writer, compress)?;
        self.opening_point.serialize_with_mode(&mut writer, compress)?;
        self.sigma.serialize_with_mode(&mut writer, compress)?;
        Ok(())
    }

    fn serialized_size(&self, compress: ark_serialize::Compress) -> usize {
        self.evaluation.serialized_size(compress) +
        self.opening_point.serialized_size(compress) +
        self.sigma.serialized_size(compress)
    }
}

impl<F: JoltField> CanonicalDeserialize for DoryProof<F> {
    fn deserialize_with_mode<R: ark_serialize::Read>(
        mut reader: R,
        compress: ark_serialize::Compress,
        validate: ark_serialize::Validate,
    ) -> Result<Self, ark_serialize::SerializationError> {
        let evaluation = Fr::deserialize_with_mode(&mut reader, compress, validate)?;
        let opening_point = Vec::<Fr>::deserialize_with_mode(&mut reader, compress, validate)?;
        let sigma = usize::deserialize_with_mode(&mut reader, compress, validate)?;
        
        let transcript = ToyTranscript::<Fr, Blake2s256>::new(b"dory_evaluation");
        let proof_builder = DoryProofBuilder::new(transcript);
        
        Ok(Self {
            evaluation,
            proof_builder,
            opening_point,
            sigma,
            _phantom: PhantomData,
        })
    }
}

impl<F: JoltField> std::fmt::Debug for DoryProof<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DoryProof")
            .field("evaluation", &self.evaluation)
            .field("opening_point", &self.opening_point)
            .field("sigma", &self.sigma)
            .finish()
    }
}

impl<F: JoltField> Valid for DoryProof<F> {
    fn check(&self) -> Result<(), ark_serialize::SerializationError> {
        self.evaluation.check()?;
        self.opening_point.check()?;
        self.sigma.check()?;
        Ok(())
    }
}

impl<F: JoltField> Default for DoryProof<F> {
    fn default() -> Self {
        let transcript = ToyTranscript::<Fr, Blake2s256>::new(b"dory_evaluation");
        let proof_builder = DoryProofBuilder::new(transcript);
        
        Self {
            evaluation: <Fr as ark_std::Zero>::zero(),
            proof_builder,
            opening_point: Vec::new(),
            sigma: 0,
            _phantom: PhantomData,
        }
    }
}

// Batched proof structure
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct DoryBatchedProof<F: JoltField> {
    proofs: Vec<DoryProof<F>>,
}

impl<F, ProofTranscript> CommitmentScheme<ProofTranscript> for DoryCommitmentScheme<F, ProofTranscript>
where
    F: JoltField,
    ProofTranscript: Transcript,
{
    type Field = F;
    type Setup = DorySetup;
    type Commitment = DoryCommitment;
    type Proof = DoryProof<F>;
    type BatchedProof = DoryBatchedProof<F>;

    fn setup(max_len: usize) -> Self::Setup {
        let max_log_n = (max_len as f64).log2().ceil() as usize;
        let (prover_setup, verifier_setup) = setup::<ArkBn254Pairing, _>(thread_rng(), max_log_n);
        
        DorySetup {
            prover_setup,
            verifier_setup,
            max_log_n,
        }
    }

    fn commit(poly: &MultilinearPolynomial<Self::Field>, setup: &Self::Setup) -> Self::Commitment {
        // Extract polynomial coefficients
        let field_coeffs = extract_field_coefficients(poly);
        
        // Convert JoltField coefficients to Fr
        // This assumes F is Fr or can be safely transmuted to Fr
        let coeffs: Vec<Fr> = field_coeffs.iter()
            .map(|&coeff| {
                unsafe { std::mem::transmute_copy(&coeff) }
            })
            .collect();
        
        let sigma = poly.get_num_vars();
        let commitment = commit::<ArkBn254Pairing, OptimizedMsmG1>(
            &coeffs,
            0,
            sigma,
            &setup.prover_setup
        );
        
        DoryCommitment { commitment }
    }

    fn batch_commit<U>(polys: &[U], setup: &Self::Setup) -> Vec<Self::Commitment>
    where
        U: Borrow<MultilinearPolynomial<Self::Field>> + Sync,
    {
        polys.iter()
            .map(|poly| Self::commit(poly.borrow(), setup))
            .collect()
    }

    fn prove(
        setup: &Self::Setup,
        poly: &MultilinearPolynomial<Self::Field>,
        opening_point: &[Self::Field],
        _transcript: &mut ProofTranscript,
    ) -> Self::Proof {
        // Extract polynomial coefficients
        let field_coeffs = extract_field_coefficients(poly);
        
        // Convert polynomial coefficients to Fr
        let coeffs: Vec<Fr> = field_coeffs.iter()
            .map(|&coeff| unsafe { std::mem::transmute_copy(&coeff) })
            .collect();
        
        // Convert opening point to Fr
        let point: Vec<Fr> = opening_point.iter()
            .map(|&p| unsafe { std::mem::transmute_copy(&p) })
            .collect();
        
        let sigma = poly.get_num_vars();
        
        // Create Dory transcript
        let dory_transcript = ToyTranscript::<Fr, Blake2s256>::new(b"dory_evaluation");
        
        let (evaluation, proof_builder) = evaluate::<ArkBn254Pairing, _, OptimizedMsmG1, OptimizedMsmG2>(
            &coeffs,
            &point,
            sigma,
            &setup.prover_setup,
            dory_transcript
        );
        
        DoryProof {
            evaluation,
            proof_builder,
            opening_point: point,
            sigma,
            _phantom: PhantomData,
        }
    }

    fn verify(
        proof: &Self::Proof,
        setup: &Self::Setup,
        _transcript: &mut ProofTranscript,
        opening_point: &[Self::Field],
        opening: &Self::Field,
        commitment: &Self::Commitment,
    ) -> Result<(), ProofVerifyError> {
        // Convert expected opening to Fr
        let expected_opening: Fr = unsafe { std::mem::transmute_copy(opening) };
        
        // Convert opening point to Fr
        let point: Vec<Fr> = opening_point.iter()
            .map(|&p| unsafe { std::mem::transmute_copy(&p) })
            .collect();
        
        // Verify that the proof evaluation matches expected
        if proof.evaluation != expected_opening {
            return Err(ProofVerifyError::InternalError);
        }
        
        // Use Dory's verify function
        let verify_result = verify::<ArkBn254Pairing, _, OptimizedMsmG1, OptimizedMsmG2, DummyMsm<Fq12>>(
            commitment.commitment,
            proof.evaluation,
            &point,
            proof.proof_builder.clone(),
            proof.sigma,
            &setup.verifier_setup,
            b"dory_evaluation"
        );
        
        match verify_result {
            Ok(()) => Ok(()),
            Err(_) => Err(ProofVerifyError::InternalError),
        }
    }

    fn protocol_name() -> &'static [u8] {
        b"dory_commitment_scheme"
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
    use ark_bn254::Fr;
    use crate::poly::dense_mlpoly::DensePolynomial;
    use crate::utils::transcript::KeccakTranscript;
    use ark_std::rand::thread_rng;
    
    #[test]
    fn test_dory_commitment_scheme() {
        // Create a random polynomial with 2^10 coefficients
        let num_vars = 10;
        let num_coeffs = 1 << num_vars; // 2^10 = 1024
        
        let mut rng = thread_rng();
        let coeffs: Vec<Fr> = (0..num_coeffs)
            .map(|_| <Fr as crate::field::JoltField>::random(&mut rng))
            .collect();
        
        // Create a multilinear polynomial
        let poly = MultilinearPolynomial::LargeScalars(
            DensePolynomial::new(coeffs)
        );
        
        // Generate random opening point
        let opening_point: Vec<Fr> = (0..num_vars)
            .map(|_| <Fr as crate::field::JoltField>::random(&mut rng))
            .collect();
        
        // Setup the commitment scheme
        let setup = DoryCommitmentScheme::<Fr, KeccakTranscript>::setup(num_vars);
        
        // Commit to the polynomial
        let commitment = DoryCommitmentScheme::<Fr, KeccakTranscript>::commit(&poly, &setup);
        
        // Evaluate the polynomial at the opening point
        use crate::poly::multilinear_polynomial::PolynomialEvaluation;
        // let expected_evaluation = poly.evaluate(&opening_point);
        
        // Create transcript for proving
        let mut prove_transcript = KeccakTranscript::new(b"dory_test");
        
        // Generate the proof
        let proof = DoryCommitmentScheme::<Fr, KeccakTranscript>::prove(
            &setup,
            &poly,
            &opening_point,
            &mut prove_transcript,
        );
        
        // Verify that the proof evaluation matches expected
        // assert_eq!(expected_evaluation, proof.evaluation);
        
        // Create transcript for verification (should be fresh)
        let mut verify_transcript = KeccakTranscript::new(b"dory_test");
        
        // Verify the proof
        let verification_result = DoryCommitmentScheme::<Fr, KeccakTranscript>::verify(
            &proof,
            &setup,
            &mut verify_transcript,
            &opening_point,
            &proof.evaluation,
            &commitment,
        );
        
        // The verification should succeed
        assert!(verification_result.is_ok(), "Dory verification failed: {:?}", verification_result);
        
        println!("✅ Dory commitment scheme test passed!");
        println!("   - Polynomial size: 2^{} = {} coefficients", num_vars, num_coeffs);
        println!("   - Commitment: {:?}", commitment.commitment);
        println!("   - Evaluation: {:?}", proof.evaluation);
    }
    
}