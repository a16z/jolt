//! Dory follows an interactive model. Hence, a "proof" consists of some messages
//! between P and V. We use Prover and Verifier "builders" to manage these messages
//! and the fiat-shamir actions throughout the implementation.
use crate::transcript::Transcript;
use std::marker::PhantomData;

use ark_ff::PrimeField;
use ark_serialize::CanonicalSerialize;
use blake2::Blake2s256;

use crate::{
    arithmetic::{Field, Group},
    messages::{
        FirstReduceChallenge, FirstReduceMessage, FoldScalarsChallenge, ScalarProductMessage,
        SecondReduceChallenge, SecondReduceMessage, VMVMessage,
    },
    toy_transcript::ToyTranscript,
};

/// Trait that defines the structure of the Dory proof.
///
/// A type implementing this trait acts as both the transcript and the proof serializer.
/// This is because these two concepts are closely related, and likely should use the same
/// underlying serialization.
pub trait ProofBuilder {
    /// G1 x G2 -> GT
    type Pairing;
    /// The $\mathbb{G}_1$ group
    type G1;
    /// The $\mathbb{G}_2$ group
    type G2;
    /// The target group, $\mathbb{G}_T$
    type GT;
    /// The scalar field, $\mathbb{F}$, of the groups
    type Scalar;

    /// Append a [`FirstReduceMessage`] to the proof and transcript and return a [`FirstReduceChallenge`] drawn from the transcript.
    #[must_use]
    fn append_first_reduce_message(
        self,
        message: FirstReduceMessage<Self::G1, Self::G2, Self::GT>,
    ) -> (FirstReduceChallenge<Self::Scalar>, Self);
    /// Append a [`SecondReduceMessage`] to the proof and transcript and return a [`SecondReduceChallenge`] drawn from the transcript.
    #[must_use]
    fn append_second_reduce_message(
        self,
        message: SecondReduceMessage<Self::G1, Self::G2, Self::GT>,
    ) -> (SecondReduceChallenge<Self::Scalar>, Self);
    /// Draw a [`FoldScalarsChallenge`] from the transcript.
    #[must_use]
    fn challenge_fold_scalars(self) -> (FoldScalarsChallenge<Self::Scalar>, Self);
    /// Append a [`ScalarProductMessage`] to the proof and transcript.
    #[must_use]
    fn append_scalar_product_message(
        self,
        message: ScalarProductMessage<Self::G1, Self::G2>,
    ) -> Self;
    #[must_use]
    /// Append a [`VMVMessage`] to the proof and transcript.
    fn append_vmv_message(self, message: VMVMessage<Self::G1, Self::GT>) -> Self;
}

/// Concrete ProofBuilder to collect messages and perform transcript tasks
pub struct DoryProofBuilder<G1, G2, GT, Scalar, T>
where
    G1: Group<Scalar = Scalar> + CanonicalSerialize,
    G2: Group<Scalar = Scalar> + CanonicalSerialize,
    GT: Group<Scalar = Scalar> + CanonicalSerialize,
    Scalar: Field + PrimeField,
    T: Transcript<Scalar = Scalar>,
{
    /// First prover message for round i
    pub first_messages: Vec<FirstReduceMessage<G1, G2, GT>>,
    /// Second prover message for round i
    pub second_messages: Vec<SecondReduceMessage<G1, G2, GT>>,
    /// Last Scalar product message at end of protocol
    pub final_message: Option<ScalarProductMessage<G1, G2>>,

    /// vector-matrix-vector message, used to transform general dory into PCS
    pub vmv_message: Option<VMVMessage<G1, GT>>,
    /// Fiat shamir
    pub transcript: T,
    /// Phantom
    pub _phantom: PhantomData<(G1, G2, GT, Scalar)>,
}

impl<G1, G2, GT, Scalar, T> DoryProofBuilder<G1, G2, GT, Scalar, T>
where
    G1: Group<Scalar = Scalar> + CanonicalSerialize,
    G2: Group<Scalar = Scalar> + CanonicalSerialize,
    GT: Group<Scalar = Scalar> + CanonicalSerialize,
    Scalar: Field + PrimeField,
    T: Transcript<Scalar = Scalar>,
{
    /// Constructor from new transcript
    pub fn new(transcript: T) -> Self {
        Self {
            first_messages: Vec::new(),
            second_messages: Vec::new(),
            final_message: None,
            vmv_message: None,
            transcript,
            _phantom: PhantomData,
        }
    }

    /// Constructor to create with ToyTranscript for testing
    pub fn new_with_toy_transcript(
        domain: &[u8],
    ) -> DoryProofBuilder<G1, G2, GT, Scalar, ToyTranscript<Scalar, Blake2s256>>
    where
        ToyTranscript<Scalar, Blake2s256>: Transcript<Scalar = Scalar>,
    {
        let transcript = ToyTranscript::<Scalar, Blake2s256>::new(domain);
        DoryProofBuilder {
            first_messages: Vec::new(),
            second_messages: Vec::new(),
            final_message: None,
            vmv_message: None,
            transcript,
            _phantom: PhantomData,
        }
    }
}

impl<G1Arg, G2Arg, GTArg, ScalarArg, T> ProofBuilder
    for DoryProofBuilder<G1Arg, G2Arg, GTArg, ScalarArg, T>
where
    G1Arg: Group<Scalar = ScalarArg> + CanonicalSerialize,
    G2Arg: Group<Scalar = ScalarArg> + CanonicalSerialize,
    GTArg: Group<Scalar = ScalarArg> + CanonicalSerialize,
    ScalarArg: Field + PrimeField,
    T: Transcript<Scalar = ScalarArg>,
{
    type G1 = G1Arg;
    type G2 = G2Arg;
    type GT = GTArg;
    type Scalar = ScalarArg;
    type Pairing = ();

    fn append_first_reduce_message(
        mut self,
        message: FirstReduceMessage<Self::G1, Self::G2, Self::GT>,
    ) -> (FirstReduceChallenge<Self::Scalar>, Self) {
        self.transcript.append_group(b"d1_left", &message.d1_left);
        self.transcript.append_group(b"d1_right", &message.d1_right);
        self.transcript.append_group(b"d2_left", &message.d2_left);
        self.transcript.append_group(b"d2_right", &message.d2_right);
        self.transcript.append_group(b"e1_beta", &message.e1_beta);
        self.transcript.append_group(b"e2_beta", &message.e2_beta);

        let beta = self.transcript.challenge_scalar(b"first_reduce_beta");
        let beta_inverse = beta.inv().expect("Inverse for beta must exist");
        let challenge = FirstReduceChallenge { beta, beta_inverse };

        self.first_messages.push(message);
        (challenge, self)
    }

    fn append_second_reduce_message(
        mut self,
        message: SecondReduceMessage<Self::G1, Self::G2, Self::GT>,
    ) -> (SecondReduceChallenge<Self::Scalar>, Self) {
        self.transcript.append_group(b"c_plus", &message.c_plus);
        self.transcript.append_group(b"c_minus", &message.c_minus);
        self.transcript.append_group(b"e1_plus", &message.e1_plus);
        self.transcript.append_group(b"e1_minus", &message.e1_minus);
        self.transcript.append_group(b"e2_plus", &message.e2_plus);
        self.transcript.append_group(b"e2_minus", &message.e2_minus);

        let alpha = self.transcript.challenge_scalar(b"second_reduce_alpha");
        let alpha_inverse = alpha.inv().expect("Inverse for alpha must exist");
        let challenge = SecondReduceChallenge {
            alpha,
            alpha_inverse,
        };

        self.second_messages.push(message);
        (challenge, self)
    }

    fn challenge_fold_scalars(mut self) -> (FoldScalarsChallenge<Self::Scalar>, Self) {
        let gamma = self.transcript.challenge_scalar(b"fold_scalars_gamma");
        let gamma_inverse = gamma.inv().expect("Inverse for gamma must exist");
        let challenge = FoldScalarsChallenge {
            gamma,
            gamma_inverse,
        };
        (challenge, self)
    }

    fn append_scalar_product_message(
        mut self,
        message: ScalarProductMessage<Self::G1, Self::G2>,
    ) -> Self {
        self.transcript.append_group(b"e1", &message.e1);
        self.transcript.append_group(b"e2", &message.e2);
        self.final_message = Some(message);
        self
    }

    fn append_vmv_message(mut self, message: VMVMessage<Self::G1, Self::GT>) -> Self {
        self.transcript.append_group(b"c_eval_vmv", &message.c);
        self.transcript.append_group(b"d2_eval_vmv", &message.d2);
        self.transcript.append_group(b"e1_eval_vmv", &message.e1);
        self.vmv_message = Some(message);
        self
    }
}

/// Verification analogue of `ProofBuilder`.
pub trait VerificationBuilder {
    /// G1
    type G1: Group;
    /// G2
    type G2: Group;
    /// GT
    type GT: Group;
    /// F_r
    type Scalar: Field + PrimeField;

    /// Number of rounds (nu)
    fn rounds(&mut self) -> usize;

    /// Returns the messages for round[idx]
    fn take_round(
        &mut self,
        idx: usize,
    ) -> (
        FirstReduceMessage<Self::G1, Self::G2, Self::GT>,
        SecondReduceMessage<Self::G1, Self::G2, Self::GT>,
    );

    /// Getter for first msg
    fn first_message(&mut self, idx: usize) -> &FirstReduceMessage<Self::G1, Self::G2, Self::GT>;

    /// Getter for second msg
    fn second_message(&mut self, idx: usize) -> &SecondReduceMessage<Self::G1, Self::G2, Self::GT>;

    /// Consume a FirstReduceMessage, append it to the transcript,
    /// and return β, β⁻¹.
    fn process_first_reduce_message(
        &mut self,
        msg: &FirstReduceMessage<Self::G1, Self::G2, Self::GT>,
    ) -> FirstReduceChallenge<Self::Scalar>;

    /// Consume a SecondReduceMessage, append, and return α, α⁻¹.
    fn process_second_reduce_message(
        &mut self,
        msg: &SecondReduceMessage<Self::G1, Self::G2, Self::GT>,
    ) -> SecondReduceChallenge<Self::Scalar>;

    /// Derive γ, γ⁻¹ after all rounds are ingested.
    fn challenge_fold_scalars(&mut self) -> FoldScalarsChallenge<Self::Scalar>;

    /// Provide the final scalar-product message that the prover sent.
    fn process_scalar_product_message(&self) -> &ScalarProductMessage<Self::G1, Self::G2>;

    /// Process a [`VMVMessage`].
    fn process_vmv_message(&mut self) -> VMVMessage<Self::G1, Self::GT>;
}

/// Concrete Dory verify builder
pub struct DoryVerifyBuilder<G1, G2, GT, Scalar, T>
where
    G1: Group<Scalar = Scalar> + CanonicalSerialize,
    G2: Group<Scalar = Scalar> + CanonicalSerialize,
    GT: Group<Scalar = Scalar> + CanonicalSerialize,
    Scalar: Field + PrimeField,
    T: Transcript<Scalar = Scalar>,
{
    transcript: T,
    first_messages: Vec<FirstReduceMessage<G1, G2, GT>>,
    second_messages: Vec<SecondReduceMessage<G1, G2, GT>>,
    scalar_msg: ScalarProductMessage<G1, G2>,
    vmv_msg: Option<VMVMessage<G1, GT>>,

    _phantom: PhantomData<(G1, G2, GT, Scalar)>,
}

impl<G1, G2, GT, Scalar, T> DoryVerifyBuilder<G1, G2, GT, Scalar, T>
where
    G1: Group<Scalar = Scalar> + CanonicalSerialize,
    G2: Group<Scalar = Scalar> + CanonicalSerialize,
    GT: Group<Scalar = Scalar> + CanonicalSerialize,
    Scalar: Field + PrimeField,
    T: Transcript<Scalar = Scalar>,
{
    /// Build from a *proof* (any concrete `DoryProofBuilder`) and a transcript.
    pub fn new_from_proof(domain: &[u8], proof: DoryProofBuilder<G1, G2, GT, Scalar, T>) -> Self {
        // destructure
        let DoryProofBuilder {
            mut transcript,
            first_messages,
            second_messages,
            final_message,
            vmv_message,
            ..
        } = proof;

        // reset for the verifier run
        transcript.reset(domain);

        let scalar_msg = final_message.expect("proof must contain the scalar-product message");

        Self {
            transcript,
            first_messages,
            second_messages,
            scalar_msg,
            vmv_msg: vmv_message,
            _phantom: PhantomData,
        }
    }
}

impl<G1, G2, GT, Scalar, T> VerificationBuilder for DoryVerifyBuilder<G1, G2, GT, Scalar, T>
where
    G1: Group<Scalar = Scalar> + CanonicalSerialize,
    G2: Group<Scalar = Scalar> + CanonicalSerialize,
    GT: Group<Scalar = Scalar> + CanonicalSerialize,
    Scalar: Field + PrimeField,
    T: Transcript<Scalar = Scalar>,
{
    type G1 = G1;
    type G2 = G2;
    type GT = GT;
    type Scalar = Scalar;

    fn rounds(&mut self) -> usize {
        self.first_messages.len()
    }

    fn take_round(
        &mut self,
        idx: usize,
    ) -> (
        FirstReduceMessage<G1, G2, GT>,
        SecondReduceMessage<G1, G2, GT>,
    ) {
        let m1 = self.first_messages[idx].clone();
        let m2 = self.second_messages[idx].clone();
        (m1, m2)
    }

    fn first_message(&mut self, idx: usize) -> &FirstReduceMessage<G1, G2, GT> {
        &self.first_messages[idx]
    }
    fn second_message(&mut self, idx: usize) -> &SecondReduceMessage<G1, G2, GT> {
        &self.second_messages[idx]
    }

    fn process_first_reduce_message(
        &mut self,
        m: &FirstReduceMessage<G1, G2, GT>,
    ) -> FirstReduceChallenge<Scalar> {
        self.transcript.append_group(b"d1_left", &m.d1_left);
        self.transcript.append_group(b"d1_right", &m.d1_right);
        self.transcript.append_group(b"d2_left", &m.d2_left);
        self.transcript.append_group(b"d2_right", &m.d2_right);
        self.transcript.append_group(b"e1_beta", &m.e1_beta);
        self.transcript.append_group(b"e2_beta", &m.e2_beta);

        let beta = self.transcript.challenge_scalar(b"first_reduce_beta");
        let beta_inv = beta.inv().unwrap();
        FirstReduceChallenge {
            beta,
            beta_inverse: beta_inv,
        }
    }

    fn process_second_reduce_message(
        &mut self,
        m: &SecondReduceMessage<G1, G2, GT>,
    ) -> SecondReduceChallenge<Scalar> {
        self.transcript.append_group(b"c_plus", &m.c_plus);
        self.transcript.append_group(b"c_minus", &m.c_minus);
        self.transcript.append_group(b"e1_plus", &m.e1_plus);
        self.transcript.append_group(b"e1_minus", &m.e1_minus);
        self.transcript.append_group(b"e2_plus", &m.e2_plus);
        self.transcript.append_group(b"e2_minus", &m.e2_minus);

        let alpha = self.transcript.challenge_scalar(b"second_reduce_alpha");
        let alpha_inv = alpha.inv().unwrap();
        SecondReduceChallenge {
            alpha,
            alpha_inverse: alpha_inv,
        }
    }

    fn challenge_fold_scalars(&mut self) -> FoldScalarsChallenge<Scalar> {
        let gamma = self.transcript.challenge_scalar(b"fold_scalars_gamma");
        let gamma_inv = gamma.inv().unwrap();
        FoldScalarsChallenge {
            gamma,
            gamma_inverse: gamma_inv,
        }
    }

    fn process_scalar_product_message(&self) -> &ScalarProductMessage<G1, G2> {
        &self.scalar_msg
    }

    fn process_vmv_message(&mut self) -> VMVMessage<G1, GT> {
        let message = self
            .vmv_msg
            .as_ref()
            .expect("VMV message must be present in verify builder");
        self.transcript.append_group(b"c_eval_vmv", &message.c);
        self.transcript.append_group(b"d2_eval_vmv", &message.d2);
        self.transcript.append_group(b"e1_eval_vmv", &message.e1);
        message.clone()
    }
}

/// Additional debug helpers:
impl<G1, G2, GT, Scalar, T> DoryProofBuilder<G1, G2, GT, Scalar, T>
where
    G1: Group<Scalar = Scalar> + CanonicalSerialize,
    G2: Group<Scalar = Scalar> + CanonicalSerialize,
    GT: Group<Scalar = Scalar> + CanonicalSerialize,
    Scalar: Field + PrimeField,
    T: Transcript<Scalar = Scalar>,
{
    /// Print statistics about the proof structure
    pub fn print_proof_stats(&self) {
        println!("\n=== PROOF STATISTICS ===");
        println!("Number of rounds: {}", self.first_messages.len());
        println!("First reduce messages: {}", self.first_messages.len());
        println!("Second reduce messages: {}", self.second_messages.len());
        println!("Has final message: {}", self.final_message.is_some());
        println!("Has VMV message: {}", self.vmv_message.is_some());

        // Calculate total proof elements
        let total_g1_elements = self.first_messages.iter().map(|_m| 1).sum::<usize>() + // e1_beta per round
                               self.second_messages.iter().map(|_m| 2).sum::<usize>() + // e1_plus + e1_minus per round
                               if self.final_message.is_some() { 1 } else { 0 } + // final e1
                               if self.vmv_message.is_some() { 1 } else { 0 }; // vmv e1

        let total_g2_elements = self.first_messages.iter().map(|_m| 1).sum::<usize>() + // e2_beta per round
                               self.second_messages.iter().map(|_m| 2).sum::<usize>() + // e2_plus + e2_minus per round
                               if self.final_message.is_some() { 1 } else { 0 }; // final e2

        let total_gt_elements = self.first_messages.iter().map(|_m| 4).sum::<usize>() + // d1_left/right + d2_left/right per round
                               self.second_messages.iter().map(|_m| 2).sum::<usize>() + // c_plus + c_minus per round
                               if self.vmv_message.is_some() { 2 } else { 0 }; // vmv c + d2

        println!("Total G1 elements in proof: {}", total_g1_elements);
        println!("Total G2 elements in proof: {}", total_g2_elements);
        println!("Total GT elements in proof: {}", total_gt_elements);
        println!(
            "Total proof elements: {}",
            total_g1_elements + total_g2_elements + total_gt_elements
        );
        println!("========================\n");
    }
}
