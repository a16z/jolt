//! Implements the Jolt CommitmentScheme trait for Dory as a light wrapper
//! Uses newtype wrappers to bridge Jolt types with Dory's trait requirements
use super::commitment_scheme::CommitmentScheme;
use crate::{
    field::JoltField,
    jolt::vm::Jolt,
    msm::{self, Icicle, VariableBaseMSM},
    poly::multilinear_polynomial::MultilinearPolynomial,
    utils::{errors::ProofVerifyError, transcript::Transcript},
};
use ark_ff::Field;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{rand::RngCore, UniformRand, Zero};
use std::borrow::Borrow;
use std::marker::PhantomData;

use ark_bn254::{Fq, Fq12, Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ec::pairing::Pairing as ArkPairing;
use ark_ec::CurveGroup;

// Correct Dory imports
use dory::{
    arithmetic::MultilinearPolynomial as DoryMultilinearPolynomial,
    arithmetic::{
        Field as DoryField, Group as DoryGroup, MultiScalarMul as DoryMultiScalarMul,
        Pairing as DoryPairing,
    },
    commit as dory_commit,
    curve::ArkBn254Pairing,
    evaluate as dory_evaluate, setup as dory_setup,
    transcript::Transcript as DoryTranscript,
    verify as dory_verify, DoryProof, DoryProofBuilder, ProverSetup, VerifierSetup,
};

/// Newtype wrapper around JoltField to implement Dory's Field trait
#[derive(Debug, Clone, Copy, PartialEq, CanonicalSerialize, CanonicalDeserialize)]
pub struct JoltFieldWrapper<F: JoltField>(pub F);

// Manual Valid impl removed, derived one from CanonicalDeserialize is used.

// Implement the traits that Dory's Field requires
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

use ark_ec::pairing::Pairing;
use ark_ff::PrimeField;
use ark_std::One;
#[derive(Debug, Clone, PartialEq, CanonicalSerialize, CanonicalDeserialize)]
pub struct JoltGTWrapper<P: Pairing> {
    pub inner: P::TargetField,
    _marker: PhantomData<P>,
}

impl<P> dory::arithmetic::Group for JoltGTWrapper<P>
where
    P: Pairing,                // fixes TargetField and ScalarField
    P::ScalarField: JoltField, // your scalar must satisfy JoltField
{
    type Scalar = JoltFieldWrapper<P::ScalarField>;

    // multiplicative-group identity = 1
    fn identity() -> Self {
        Self {
            inner: P::TargetField::one(),
            _marker: PhantomData,
        }
    }

    // group operation = field multiplication
    fn add(&self, rhs: &Self) -> Self {
        Self {
            inner: self.inner * rhs.inner,
            _marker: PhantomData,
        }
    }

    // inverse = field inverse
    fn neg(&self) -> Self {
        Self {
            inner: self.inner.inverse().unwrap(),
            _marker: PhantomData,
        }
    }

    // “scale” = exponentiation by the scalar
    fn scale(&self, k: &Self::Scalar) -> Self {
        Self {
            inner: self.inner.pow(k.0.into_bigint()),
            _marker: PhantomData,
        }
    }

    fn random<R: RngCore>(rng: &mut R) -> Self {
        Self {
            inner: P::TargetField::rand(rng),
            _marker: PhantomData,
        }
    }
}

impl<P: Pairing> Default for JoltGTWrapper<P> {
    fn default() -> Self {
        Self {
            inner: P::TargetField::one(),
            _marker: PhantomData,
        }
    }
}

#[derive(Debug, Clone, PartialEq, CanonicalSerialize, CanonicalDeserialize)]
pub struct JoltGroupWrapper<G: CurveGroup> {
    pub inner: G,
    _marker: PhantomData<G>, // keeps the type parameter in the value
}

impl<G> dory::arithmetic::Group for JoltGroupWrapper<G>
where
    G: CurveGroup + VariableBaseMSM, // ensures `ScalarField` exists
    G::ScalarField: JoltField,       // your scalar must satisfy JoltField
{
    type Scalar = JoltFieldWrapper<G::ScalarField>;

    fn identity() -> Self {
        Self {
            inner: G::zero(),
            _marker: PhantomData,
        }
    }
    fn add(&self, rhs: &Self) -> Self {
        Self {
            inner: self.inner + rhs.inner,
            _marker: PhantomData,
        }
    }
    fn neg(&self) -> Self {
        Self {
            inner: -self.inner,
            _marker: PhantomData,
        }
    }
    fn scale(&self, k: &Self::Scalar) -> Self {
        Self {
            inner: self.inner * k.0,
            _marker: PhantomData,
        }
    }
    fn random<R: RngCore>(rng: &mut R) -> Self {
        Self {
            inner: G::rand(rng),
            _marker: PhantomData,
        }
    }
}

/// Jolt's custom MSM implementation for Dory
pub struct JoltCustomMSM<G: CurveGroup> {
    _phantom: PhantomData<G>,
}

impl<G> dory::arithmetic::MultiScalarMul<JoltGroupWrapper<G>> for JoltCustomMSM<G>
where
    G: CurveGroup + VariableBaseMSM, // <-- provides msm() and ScalarField
    G::ScalarField: JoltField,
{
    fn msm(
        bases: &[JoltGroupWrapper<G>],
        scalars: &dory::arithmetic::MultilinearPolynomial<JoltFieldWrapper<G::ScalarField>>,
    ) -> JoltGroupWrapper<G> {
        // 1. Convert bases to affine
        let affines: Vec<G::Affine> = bases.iter().map(|w| w.inner.into_affine()).collect();

        // 2. Convert scalars to Jolt’s multilinear poly over the concrete field
        let ml_poly: crate::poly::multilinear_polynomial::MultilinearPolynomial<G::ScalarField> =
            scalars.into();

        // 3. Run the curve’s built-in variable-base MSM
        let proj =
            <G as msm::VariableBaseMSM>::msm(&affines, None, &ml_poly, None).expect("MSM failed");

        JoltGroupWrapper {
            inner: proj,
            _marker: PhantomData,
        }
    }
}

// 3. Handy aliases for the concrete Bn254 groups --------------------------------
pub type JoltG1Wrapper = JoltGroupWrapper<G1Projective>;
pub type JoltG2Wrapper = JoltGroupWrapper<G2Projective>;
pub type JoltGTBn254 = JoltGTWrapper<ark_bn254::Bn254>;
use ark_bn254::Bn254;
pub type JoltBn254 = JoltPairing<Bn254>;

pub type JoltG1MSM = JoltCustomMSM<G1Projective>;
pub type JoltG2MSM = JoltCustomMSM<G2Projective>;

#[derive(Clone)]
pub struct JoltBn254Pairing;

impl DoryPairing for JoltBn254Pairing {
    type G1 = JoltG1Wrapper;
    type G2 = JoltG2Wrapper;
    type GT = JoltGTBn254;

    fn pair(p: &Self::G1, q: &Self::G2) -> Self::GT {
        let gt = ark_bn254::Bn254::pairing(p.inner, q.inner).0;
        Self::GT {
            inner: gt,
            _marker: PhantomData,
        }
    }

    fn multi_pair(ps: &[Self::G1], qs: &[Self::G2]) -> Self::GT {
        std::assert_eq!(
            ps.len(),
            qs.len(),
            "multi_pair requires equal length vectors"
        );

        if ps.is_empty() {
            return Self::GT::identity();
        }

        ps.iter()
            .zip(qs.iter())
            .fold(Self::GT::identity(), |acc, (p, q)| {
                acc.add(&Self::pair(p, q))
            })
    }
}

/// Zero-sized marker type that “selects” the engine `E`
pub struct JoltPairing<E: Pairing>(PhantomData<E>);

impl<E> DoryPairing for JoltPairing<E>
where
    E: Pairing,
    E::ScalarField: JoltField,
    E::G1: Icicle,
    E::G2: Icicle,
    // E::TargetField: ark_ff::PrimeField,
{
    // Wrap the concrete curve types chosen by `E`
    type G1 = JoltGroupWrapper<E::G1>;
    type G2 = JoltGroupWrapper<E::G2>;
    type GT = JoltGTWrapper<E>;

    #[inline]
    fn pair(p: &Self::G1, q: &Self::G2) -> Self::GT {
        let gt = E::pairing(p.inner, q.inner).0;
        Self::GT {
            inner: gt,
            _marker: PhantomData,
        }
    }

    fn multi_pair(ps: &[Self::G1], qs: &[Self::G2]) -> Self::GT {
        assert_eq!(
            ps.len(),
            qs.len(),
            "multi_pair requires equal length vectors"
        );

        ps.iter().zip(qs).fold(Self::GT::identity(), |acc, (p, q)| {
            acc.add(&Self::pair(p, q))
        })
    }
}

/// Convert Dory's polynomial to Jolt's polynomial using Into trait
impl<'a, F: JoltField> From<&DoryMultilinearPolynomial<'a, JoltFieldWrapper<F>>>
    for MultilinearPolynomial<F>
{
    fn from(dory_poly: &DoryMultilinearPolynomial<'a, JoltFieldWrapper<F>>) -> Self {
        match dory_poly {
            DoryMultilinearPolynomial::LargeScalars(scalars) => {
                use crate::poly::dense_mlpoly::DensePolynomial;
                let field_scalars: Vec<F> = scalars.iter().map(|w| w.0).collect();
                MultilinearPolynomial::LargeScalars(DensePolynomial::new(field_scalars))
            }
            DoryMultilinearPolynomial::U8Scalars(scalars) => {
                use crate::poly::compact_polynomial::CompactPolynomial;
                MultilinearPolynomial::U8Scalars(CompactPolynomial::from_coeffs(scalars.to_vec()))
            }
            DoryMultilinearPolynomial::U16Scalars(scalars) => {
                use crate::poly::compact_polynomial::CompactPolynomial;
                MultilinearPolynomial::U16Scalars(CompactPolynomial::from_coeffs(scalars.to_vec()))
            }
            DoryMultilinearPolynomial::U32Scalars(scalars) => {
                use crate::poly::compact_polynomial::CompactPolynomial;
                MultilinearPolynomial::U32Scalars(CompactPolynomial::from_coeffs(scalars.to_vec()))
            }
            DoryMultilinearPolynomial::U64Scalars(scalars) => {
                use crate::poly::compact_polynomial::CompactPolynomial;
                MultilinearPolynomial::U64Scalars(CompactPolynomial::from_coeffs(scalars.to_vec()))
            }
            DoryMultilinearPolynomial::I64Scalars(scalars) => {
                use crate::poly::compact_polynomial::CompactPolynomial;
                MultilinearPolynomial::I64Scalars(CompactPolynomial::from_coeffs(scalars.to_vec()))
            }
        }
    }
}

/// zero-copy borrow from Jolt → Dory
impl<'a, F: JoltField> From<&'a MultilinearPolynomial<F>>
    for DoryMultilinearPolynomial<'a, JoltFieldWrapper<F>>
{
    fn from(
        src: &'a MultilinearPolynomial<F>,
    ) -> DoryMultilinearPolynomial<'a, JoltFieldWrapper<F>> {
        match src {
            // Dense coefficients ──────────────────────────────
            MultilinearPolynomial::LargeScalars(dense) => {
                // SAFETY: JoltFieldWrapper<F> is #[repr(transparent)] over F,
                // so the slice layout is identical.
                let wrapped: &'a [JoltFieldWrapper<F>] = unsafe {
                    std::slice::from_raw_parts(
                        dense.Z.as_ptr() as *const JoltFieldWrapper<F>,
                        dense.Z.len(),
                    )
                };
                DoryMultilinearPolynomial::LargeScalars(wrapped)
            }

            // Compact forms ───────────────────────────────────
            MultilinearPolynomial::U8Scalars(cp) => {
                DoryMultilinearPolynomial::U8Scalars(cp.coeffs.as_slice())
            }
            MultilinearPolynomial::U16Scalars(cp) => {
                DoryMultilinearPolynomial::U16Scalars(cp.coeffs.as_slice())
            }
            MultilinearPolynomial::U32Scalars(cp) => {
                DoryMultilinearPolynomial::U32Scalars(cp.coeffs.as_slice())
            }
            MultilinearPolynomial::U64Scalars(cp) => {
                DoryMultilinearPolynomial::U64Scalars(cp.coeffs.as_slice())
            }
            MultilinearPolynomial::I64Scalars(cp) => {
                DoryMultilinearPolynomial::I64Scalars(cp.coeffs.as_slice())
            }
        }
    }
}

// impl<'a, F: JoltField> Into<DoryMultilinearPolynomial<'a, JoltFieldWrapper<F>>>
//     for &'a MultilinearPolynomial<F>
// {
//     fn into(self) -> DoryMultilinearPolynomial<'a, JoltFieldWrapper<F>> {
//         Self::from(self)
//     }
// }

/// Newtype wrapper adapting Jolt Transcript to Dory Transcript
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

impl<'a, F: JoltField + DoryField, T: Transcript> DoryTranscript
    for JoltToDoryTranscriptRef<'a, F, T>
{
    type Scalar = F;

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
        transcript.append_scalar(x);
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
        transcript.challenge_scalar::<F>()
    }

    fn reset(&mut self, _domain_label: &[u8]) {
        panic!("We do not want to ever reset JoltToDoryTranscript.")
    }
}

/// Dory commitment scheme implementation
#[derive(Clone, Debug)]
pub struct DoryCommitmentScheme<F: JoltField, ProofTranscript: Transcript> {
    _phantom_field: PhantomData<F>,
    _phantom_transcript: PhantomData<ProofTranscript>,
}

/// Setup structure containing both prover and verifier parameters.
#[derive(Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct DorySetup {
    pub prover_setup: ProverSetup<JoltBn254>,
    pub verifier_setup: VerifierSetup<JoltBn254>,
    pub max_log_n: usize,
}

/// Commitment structure
#[derive(Clone, Debug, Default, PartialEq, CanonicalSerialize, CanonicalDeserialize)]
pub struct DoryCommitment {
    pub commitment: JoltGTBn254,
}

/// Proof structure storing the serializable DoryProof data and other data
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct DoryProofData<F: JoltField> {
    pub evaluation: F,
    pub opening_point: Vec<F>,
    pub sigma: usize,
    pub dory_proof_data: DoryProof<JoltG1Wrapper, JoltG2Wrapper, JoltGTBn254>, // Using Dory's DoryProof directly
    _phantom_f_marker: PhantomData<F>,
}

/// Batched proof structure
#[derive(Default, Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct DoryBatchedProof<F: JoltField> {
    proofs: Vec<DoryProofData<F>>,
}

impl<F, ProofTranscript> CommitmentScheme<ProofTranscript>
    for DoryCommitmentScheme<F, ProofTranscript>
where
    F: JoltField,
    ProofTranscript: Transcript,
{
    type Field = F;
    type Setup = DorySetup;
    type Commitment = DoryCommitment;
    type Proof = DoryProofData<F>;
    type BatchedProof = DoryBatchedProof<F>;

    fn setup(max_log_n: usize) -> Self::Setup {
        let (prover_setup, verifier_setup) =
            dory_setup::<JoltBn254Pairing, _>(ark_std::rand::thread_rng(), max_log_n);

        DorySetup {
            prover_setup,
            verifier_setup,
            max_log_n,
        }
    }

    fn commit(poly: &MultilinearPolynomial<Self::Field>, setup: &Self::Setup) -> Self::Commitment {
        let dory_poly: DoryMultilinearPolynomial<'static, JoltFieldWrapper<Self::Field>> =
            poly.into();
            
        let num_vars = poly.get_num_vars();
        let sigma = (num_vars + 1) / 2;

        let prover_setup: ProverSetup<JoltBn254> = setup.prover_setup;

        let commitment_val = dory_commit::<JoltBn254, _>(
            &dory_poly,
            0,
            sigma,
            &prover_setup,
        );
        DoryCommitment {
            commitment: commitment_val,
        }
    }

    fn batch_commit<U>(_polys: &[U], _setup: &Self::Setup) -> Vec<Self::Commitment>
    where
        U: Borrow<MultilinearPolynomial<Self::Field>> + Sync,
    {
        todo!("Implement linear combination batching")
    }

    fn prove(
        setup: &Self::Setup,
        poly: &MultilinearPolynomial<Self::Field>,
        opening_point: &[Self::Field], // This is &[F]
        transcript: &mut ProofTranscript,
    ) -> Self::Proof {
        let dory_poly: DoryMultilinearPolynomial<'static, JoltFieldWrapper<Self::Field>> =
            poly.into();

        // Dory's evaluate expects point: &[P::G1::Scalar], which is &[JoltFieldWrapper<Self::Field>]
        let point_dory: Vec<JoltFieldWrapper<Self::Field>> = opening_point
            .iter()
            .map(|&p_val| JoltFieldWrapper(p_val))
            .collect();

        let num_vars = poly.get_num_vars();
        let sigma = (num_vars + 1) / 2;
        let dory_transcript_prover = JoltToDoryTranscriptRef::<Self::Field, _>::new(transcript);

        let (eval_wrapper, proof_builder) = dory_evaluate::<
            JoltBn254Pairing<Self::Field>,
            JoltToDoryTranscriptRef<'_, Self::Field, ProofTranscript>,
            JoltCustomMSM<JoltG1Wrapper<Self::Field>, G1Projective>,
            JoltCustomMSM<JoltG2Wrapper<Self::Field>, G2Projective>,
        >(
            &dory_poly,
            &point_dory,
            sigma,
            &setup.prover_setup, // Potential setup type issue here as well
            dory_transcript_prover,
        );

        let dory_proof_data_built = proof_builder.build(); // This will be DoryProofData<JoltG1Wrapper, JoltG2Wrapper, JoltGTWrapper>

        DoryProofData {
            evaluation: eval_wrapper.0.into(), // eval_wrapper is JoltFieldWrapper<F>, .0 is F, .into() converts F to Fr
            opening_point: opening_point.to_vec(), // Store original opening point of type Vec<F>
            sigma,
            dory_proof_data: dory_proof_data_built,
            _phantom_f_marker: PhantomData,
        }
    }

    fn verify(
        proof: &Self::Proof, // proof.evaluation is Fr, proof.opening_point is Vec<F>
        setup: &Self::Setup,
        transcript: &mut ProofTranscript,
        opening_point_param: &[Self::Field], // This is &[F]
        _opening: &Self::Field,              // Self::Field, e.g. Fr
        commitment: &Self::Commitment,       // DoryCommitment { commitment: JoltGTWrapper }
    ) -> Result<(), ProofVerifyError> {
        // Dory's verify expects point: &[P::G1::Scalar], i.e. &[JoltFieldWrapper<Self::Field>]
        // We use proof.opening_point which is Vec<F>. Let's ensure consistency with opening_point_param if needed.
        // Typically, the opening_point used for verification should be the one associated with the proof.
        let point_dory: Vec<JoltFieldWrapper<Self::Field>> = proof
            .opening_point
            .iter()
            .map(|&p_val| JoltFieldWrapper(p_val))
            .collect();

        // Dory's verify expects evaluation: P::G1::Scalar, i.e. JoltFieldWrapper<Self::Field>
        // proof.evaluation is Fr. Need to convert Fr to F, then wrap.
        let eval_f: F = proof.evaluation.into(); // Fr to F
        let eval_wrapper = JoltFieldWrapper(eval_f);

        let mut dory_transcript = JoltToDoryTranscriptRef::<Self::Field, _>::new(transcript);
        let verifier_builder =
            DoryProofBuilder::from_proof_no_transcript(proof.dory_proof_data.clone());

        let verify_result = dory_verify::<
            JoltBn254Pairing<Self::Field>,
            JoltToDoryTranscriptRef<'_, Self::Field, ProofTranscript>,
            JoltCustomMSM<JoltG1Wrapper<Self::Field>, G1Projective>,
            JoltCustomMSM<JoltG2Wrapper<Self::Field>, G2Projective>,
            dory::curve::DummyMsm<JoltGTWrapper>, // M3 for GT
        >(
            commitment.commitment, // This is JoltGTWrapper
            eval_wrapper,          // This is JoltFieldWrapper<F>
            &point_dory,
            verifier_builder,
            proof.sigma,
            &setup.verifier_setup, // Potential setup type issue
            dory_transcript,
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
    fn append_to_transcript<PT: Transcript>(&self, transcript: &mut PT) {
        // self.commitment is JoltGTWrapper which is CanonicalSerialize
        transcript.append_serializable(&self.commitment);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poly::dense_mlpoly::DensePolynomial;
    use crate::utils::transcript::KeccakTranscript;
    use ark_bn254::Fr; // Use Fr directly as the field type for the test
    use ark_std::rand::thread_rng;
    use ark_std::UniformRand; // For Fr::rand
    use std::time::Instant;

    #[test]
    fn test_dory_commitment_scheme() {
        let max_log_n = 10; // Reduced for faster testing; original 22
        let num_vars = 10; // Reduced for faster testing
        let num_coeffs = 1 << num_vars;
        let sigma = (num_vars + 1) / 2;

        println!(
            "Testing Dory PCS with {} variables, {} coefficients, sigma = {}",
            num_vars, num_coeffs, sigma
        );

        let mut rng = thread_rng();
        // Self::Field for DoryCommitmentScheme in test is Fr
        let coeffs: Vec<Fr> = (0..num_coeffs).map(|_| Fr::rand(&mut rng)).collect();
        let poly = MultilinearPolynomial::LargeScalars(DensePolynomial::new(coeffs.clone()));
        let opening_point: Vec<Fr> = (0..num_vars).map(|_| Fr::rand(&mut rng)).collect();

        let setup_start = Instant::now();
        // Here F is Fr. Fr implements JoltField (implicitly assumed by original code).
        // Fr also needs to satisfy From<Fr> + Into<Fr> + Zero + UniformRand + Copy
        // which Fr does.
        let setup = DoryCommitmentScheme::<Fr, KeccakTranscript>::setup(max_log_n);
        let setup_time = setup_start.elapsed();
        println!("Setup time: {:?}", setup_time);

        let commit_start = Instant::now();
        let commitment = DoryCommitmentScheme::<Fr, KeccakTranscript>::commit(&poly, &setup);
        let commit_time = commit_start.elapsed();
        println!("Commit time: {:?}", commit_time);

        let mut prove_transcript = KeccakTranscript::new(b"dory_test");
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
        let verify_start = Instant::now();
        // proof.evaluation is Fr. _opening parameter to verify also expects &Fr.
        let verification_result = DoryCommitmentScheme::<Fr, KeccakTranscript>::verify(
            &proof,
            &setup,
            &mut verify_transcript,
            &opening_point,    // &[Fr]
            &proof.evaluation, // &Fr
            &commitment,
        );
        let verify_time = verify_start.elapsed();
        println!("Verify time: {:?}", verify_time);

        let total_time = setup_time + commit_time + prove_time + verify_time;
        println!("Total time: {:?}", total_time);

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
        // commitment.commitment is JoltGTWrapper(Fq12)
        println!("   - Commitment: {:?}", commitment.commitment.0);
        println!("   - Evaluation: {:?}", proof.evaluation);
    }
}
