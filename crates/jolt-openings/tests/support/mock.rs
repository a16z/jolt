use std::marker::PhantomData;

use jolt_crypto::{Commitment, HomomorphicCommitment};
use jolt_field::Field;
use jolt_openings::{AdditivelyHomomorphic, CommitmentScheme, OpeningsError, ZkOpeningScheme};
use jolt_poly::Polynomial;
use jolt_transcript::{AppendToTranscript, Transcript};
use serde::{de::DeserializeOwned, Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct MockCommitmentScheme<F: Field>(PhantomData<F>);

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: DeserializeOwned"))]
pub struct MockCommitment<F: Field> {
    evaluations: Vec<F>,
}

impl<F: Field> Default for MockCommitment<F> {
    fn default() -> Self {
        Self {
            evaluations: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: DeserializeOwned"))]
pub struct MockProof<F: Field> {
    evaluations: Vec<F>,
}

impl<F: Field> AppendToTranscript for MockCommitment<F> {
    fn append_to_transcript<T: Transcript>(&self, transcript: &mut T) {
        for evaluation in &self.evaluations {
            evaluation.append_to_transcript(transcript);
        }
    }
}

impl<F> Commitment for MockCommitmentScheme<F>
where
    F: Field + Serialize + DeserializeOwned,
{
    type Output = MockCommitment<F>;
}

impl<F> CommitmentScheme for MockCommitmentScheme<F>
where
    F: Field + Serialize + DeserializeOwned,
{
    type Field = F;
    type Proof = MockProof<F>;
    type ProverSetup = ();
    type VerifierSetup = ();
    type Polynomial = Polynomial<F>;
    type OpeningHint = ();
    type SetupParams = ();

    fn setup(_params: Self::SetupParams) -> ((), ()) {
        ((), ())
    }

    fn verifier_setup(_prover_setup: &()) {}

    fn commit<P: jolt_poly::MultilinearPoly<Self::Field> + ?Sized>(
        poly: &P,
        _setup: &Self::ProverSetup,
    ) -> (Self::Output, ()) {
        let mut evaluations = Vec::with_capacity(1 << poly.num_vars());
        poly.for_each_row(poly.num_vars(), &mut |_, row| {
            evaluations.extend_from_slice(row);
        });
        (MockCommitment { evaluations }, ())
    }

    fn open(
        poly: &Self::Polynomial,
        _point: &[Self::Field],
        _eval: Self::Field,
        _setup: &Self::ProverSetup,
        _hint: Option<()>,
        _transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Self::Proof {
        MockProof {
            evaluations: poly.evaluations().to_vec(),
        }
    }

    fn verify(
        commitment: &Self::Output,
        point: &[Self::Field],
        eval: Self::Field,
        proof: &Self::Proof,
        _setup: &Self::VerifierSetup,
        _transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Result<(), OpeningsError> {
        if commitment.evaluations != proof.evaluations {
            return Err(OpeningsError::VerificationFailed);
        }
        let poly = Polynomial::new(proof.evaluations.clone());
        if poly.evaluate(point) != eval {
            return Err(OpeningsError::VerificationFailed);
        }
        Ok(())
    }
}

impl<F: Field> HomomorphicCommitment<F> for MockCommitment<F> {
    fn add(c1: &Self, c2: &Self) -> Self {
        Self::linear_combine(c1, c2, &F::one())
    }

    fn linear_combine(c1: &Self, c2: &Self, scalar: &F) -> Self {
        let len = c1.evaluations.len().max(c2.evaluations.len());
        let mut result = vec![F::zero(); len];
        for (i, r) in result.iter_mut().enumerate() {
            let a = c1.evaluations.get(i).copied().unwrap_or_else(F::zero);
            let b = c2.evaluations.get(i).copied().unwrap_or_else(F::zero);
            *r = a + *scalar * b;
        }
        Self {
            evaluations: result,
        }
    }
}

impl<F> AdditivelyHomomorphic for MockCommitmentScheme<F>
where
    F: Field + Serialize + DeserializeOwned,
{
    fn combine(commitments: &[Self::Output], scalars: &[Self::Field]) -> Self::Output {
        assert_eq!(commitments.len(), scalars.len());
        let len = commitments.first().map_or(0, |c| c.evaluations.len());
        let mut result = vec![F::zero(); len];
        for (commitment, scalar) in commitments.iter().zip(scalars.iter()) {
            for (result, evaluation) in result.iter_mut().zip(commitment.evaluations.iter()) {
                *result += *scalar * *evaluation;
            }
        }
        MockCommitment {
            evaluations: result,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: DeserializeOwned"))]
pub struct MockHidingCommitment<F: Field> {
    pub eval: F,
}

impl<F: Field> AppendToTranscript for MockHidingCommitment<F> {
    fn append_to_transcript<T: Transcript>(&self, transcript: &mut T) {
        self.eval.append_to_transcript(transcript);
    }
}

impl<F> ZkOpeningScheme for MockCommitmentScheme<F>
where
    F: Field + Serialize + DeserializeOwned,
{
    type HidingCommitment = MockHidingCommitment<F>;
    type Blind = ();

    fn commit_zk<P: jolt_poly::MultilinearPoly<Self::Field> + ?Sized>(
        poly: &P,
        setup: &Self::ProverSetup,
    ) -> (Self::Output, Self::OpeningHint) {
        Self::commit(poly, setup)
    }

    fn open_zk(
        poly: &Self::Polynomial,
        _point: &[Self::Field],
        eval: Self::Field,
        _setup: &Self::ProverSetup,
        _hint: Self::OpeningHint,
        _transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> (Self::Proof, Self::HidingCommitment, Self::Blind) {
        (
            MockProof {
                evaluations: poly.evaluations().to_vec(),
            },
            MockHidingCommitment { eval },
            (),
        )
    }

    fn verify_zk(
        commitment: &Self::Output,
        point: &[Self::Field],
        proof: &Self::Proof,
        _setup: &Self::VerifierSetup,
        _transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Result<Self::HidingCommitment, OpeningsError> {
        if commitment.evaluations != proof.evaluations {
            return Err(OpeningsError::VerificationFailed);
        }
        let poly = Polynomial::new(proof.evaluations.clone());
        Ok(MockHidingCommitment {
            eval: poly.evaluate(point),
        })
    }
}
