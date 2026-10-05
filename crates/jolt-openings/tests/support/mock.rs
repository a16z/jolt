use std::marker::PhantomData;

use jolt_crypto::{Commitment, HomomorphicCommitment};
use jolt_field::{CanonicalBytes, CanonicalDecode, JoltField};
use jolt_openings::{AdditivelyHomomorphic, CommitmentScheme, OpeningsError, ZkOpeningScheme};
use jolt_poly::{MultilinearPoly, Polynomial};
use jolt_transcript::{Channel, ProverTranscript, Sponge, VerifierTranscript};
use serde::{de::DeserializeOwned, Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct MockCommitmentScheme<F: JoltField>(PhantomData<F>);

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: DeserializeOwned"))]
pub struct MockCommitment<F: JoltField> {
    evaluations: Vec<F>,
}

impl<F: JoltField> Default for MockCommitment<F> {
    fn default() -> Self {
        Self {
            evaluations: Vec::new(),
        }
    }
}

/// The mock "opening proof" is the polynomial itself, sent in the clear.
fn send_evaluations<F: JoltField, H: Sponge>(
    evaluations: &[F],
    transcript: &mut ProverTranscript<H>,
) {
    transcript.send(&(evaluations.len() as u64).to_le_bytes());
    transcript.send_all(evaluations);
}

fn receive_evaluations<F: JoltField, H: Sponge>(
    transcript: &mut VerifierTranscript<'_, H>,
) -> Result<Vec<F>, OpeningsError> {
    let len = usize::try_from(u64::from_le_bytes(transcript.receive()?))
        .map_err(|_| OpeningsError::VerificationFailed)?;
    Ok(transcript.receive_n(len)?)
}

impl<F> Commitment for MockCommitmentScheme<F>
where
    F: JoltField + Serialize + DeserializeOwned,
{
    type Output = MockCommitment<F>;
}

impl<F> CommitmentScheme for MockCommitmentScheme<F>
where
    F: JoltField + Serialize + DeserializeOwned,
{
    type Field = F;
    type ProverSetup = ();
    type VerifierSetup = ();
    type OpeningHint = ();
    type SetupParams = ();

    fn setup(_params: Self::SetupParams) -> Result<((), ()), OpeningsError> {
        Ok(((), ()))
    }

    fn verifier_setup(_prover_setup: &()) {}

    fn commit<P: MultilinearPoly<Self::Field> + ?Sized>(
        poly: &P,
        _setup: &Self::ProverSetup,
    ) -> Result<(Self::Output, ()), OpeningsError> {
        let evaluations = poly.to_dense().into_owned();
        Ok((MockCommitment { evaluations }, ()))
    }

    fn send_commitment<H: Sponge>(commitment: &Self::Output, transcript: &mut ProverTranscript<H>) {
        send_evaluations(&commitment.evaluations, transcript);
    }

    fn receive_commitment<H: Sponge>(
        _setup: &Self::VerifierSetup,
        transcript: &mut VerifierTranscript<'_, H>,
    ) -> Result<Self::Output, OpeningsError> {
        Ok(MockCommitment {
            evaluations: receive_evaluations(transcript)?,
        })
    }

    fn absorb_commitment<C: Channel>(commitment: &Self::Output, channel: &mut C) {
        channel.public_all(&commitment.evaluations);
    }

    fn open<P: MultilinearPoly<Self::Field> + ?Sized, H: Sponge>(
        poly: &P,
        _point: &[Self::Field],
        _eval: Self::Field,
        _setup: &Self::ProverSetup,
        _hint: Option<()>,
        transcript: &mut ProverTranscript<H>,
    ) -> Result<(), OpeningsError> {
        send_evaluations(&poly.to_dense(), transcript);
        Ok(())
    }

    fn verify<H: Sponge>(
        commitment: &Self::Output,
        point: &[Self::Field],
        eval: Self::Field,
        _setup: &Self::VerifierSetup,
        transcript: &mut VerifierTranscript<'_, H>,
    ) -> Result<(), OpeningsError> {
        let evaluations = receive_evaluations(transcript)?;
        if commitment.evaluations != evaluations {
            return Err(OpeningsError::VerificationFailed);
        }
        if Polynomial::new(evaluations).evaluate(point) != eval {
            return Err(OpeningsError::VerificationFailed);
        }
        Ok(())
    }
}

impl<F: JoltField> HomomorphicCommitment<F> for MockCommitment<F> {
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
    F: JoltField + Serialize + DeserializeOwned,
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
pub struct MockHidingCommitment<F: JoltField> {
    pub eval: F,
}

impl<F: JoltField> CanonicalBytes for MockHidingCommitment<F> {
    const NUM_BYTES: usize = F::NUM_BYTES;

    fn to_bytes_le(&self, out: &mut [u8]) {
        self.eval.to_bytes_le(out);
    }
}

impl<F: JoltField> spongefish::Encoding<[u8]> for MockHidingCommitment<F> {
    fn encode(&self) -> impl AsRef<[u8]> {
        jolt_field::narg::encode(self)
    }
}

impl<F: JoltField> CanonicalDecode for MockHidingCommitment<F> {
    fn from_bytes_le_checked(bytes: &[u8]) -> Option<Self> {
        F::from_bytes_le_checked(bytes).map(|eval| Self { eval })
    }
}

impl<F: JoltField> spongefish::NargDeserialize for MockHidingCommitment<F> {
    fn deserialize_from_narg(buf: &mut &[u8]) -> spongefish::VerificationResult<Self> {
        jolt_field::narg::deserialize(buf)
    }
}

impl<F> ZkOpeningScheme for MockCommitmentScheme<F>
where
    F: JoltField + Serialize + DeserializeOwned,
{
    type HidingCommitment = MockHidingCommitment<F>;
    type Blind = ();

    fn commit_zk<P: MultilinearPoly<Self::Field> + ?Sized>(
        poly: &P,
        setup: &Self::ProverSetup,
    ) -> Result<(Self::Output, Self::OpeningHint), OpeningsError> {
        Self::commit(poly, setup)
    }

    fn open_zk<P: MultilinearPoly<Self::Field> + ?Sized, H: Sponge>(
        poly: &P,
        _point: &[Self::Field],
        eval: Self::Field,
        _setup: &Self::ProverSetup,
        _hint: Self::OpeningHint,
        transcript: &mut ProverTranscript<H>,
    ) -> Result<(Self::HidingCommitment, Self::Blind), OpeningsError> {
        send_evaluations(&poly.to_dense(), transcript);
        Ok((MockHidingCommitment { eval }, ()))
    }

    fn verify_zk<H: Sponge>(
        commitment: &Self::Output,
        point: &[Self::Field],
        _setup: &Self::VerifierSetup,
        transcript: &mut VerifierTranscript<'_, H>,
    ) -> Result<Self::HidingCommitment, OpeningsError> {
        let evaluations = receive_evaluations(transcript)?;
        if commitment.evaluations != evaluations {
            return Err(OpeningsError::VerificationFailed);
        }
        Ok(MockHidingCommitment {
            eval: Polynomial::new(evaluations).evaluate(point),
        })
    }
}
