use jolt_field::Field;
use jolt_openings::CommitmentScheme;
use jolt_witness::{WitnessNamespace, WitnessProvider};

use crate::{
    BackendError, CommitmentRequest, CommitmentResult, OpeningRequest, OpeningResult,
    SumcheckRequest, SumcheckResult,
};

#[cfg(feature = "zk")]
use crate::{BlindFoldRequest, BlindFoldResult};

pub trait Backend {
    fn name(&self) -> &'static str;
}

pub trait CommitmentBackend<F, N, PCS>: Backend
where
    N: WitnessNamespace,
    PCS: CommitmentScheme<Field = F>,
{
    fn commit<W>(
        &mut self,
        request: &CommitmentRequest<N>,
        witness: &W,
        setup: &PCS::ProverSetup,
    ) -> Result<CommitmentResult<N, PCS>, BackendError>
    where
        W: WitnessProvider<F, N>;
}

pub trait SumcheckBackend<F, N>: Backend
where
    F: Field,
    N: WitnessNamespace,
{
    type Proof;

    fn prove_sumcheck<W>(
        &mut self,
        request: &SumcheckRequest<N>,
        witness: &W,
    ) -> Result<SumcheckResult<F, Self::Proof>, BackendError>
    where
        W: WitnessProvider<F, N>,
    {
        let _ = request;
        let _ = witness;
        Err(BackendError::UnsupportedTask {
            backend: self.name(),
            task: "sumcheck",
        })
    }
}

pub trait OpeningBackend<F, N, PCS>: Backend
where
    F: Field,
    N: WitnessNamespace,
    PCS: CommitmentScheme<Field = F>,
{
    fn open<W>(
        &mut self,
        request: &OpeningRequest<F, N>,
        witness: &W,
        setup: &PCS::ProverSetup,
    ) -> Result<OpeningResult<F, PCS::Proof>, BackendError>
    where
        W: WitnessProvider<F, N>,
    {
        let _ = request;
        let _ = witness;
        let _ = setup;
        Err(BackendError::UnsupportedTask {
            backend: self.name(),
            task: "opening",
        })
    }
}

#[cfg(feature = "zk")]
pub trait BlindFoldBackend<F>: Backend
where
    F: Field,
{
    type Proof;

    fn prove_blindfold(
        &mut self,
        request: &BlindFoldRequest<F>,
    ) -> Result<BlindFoldResult<F, Self::Proof>, BackendError> {
        let _ = request;
        Err(BackendError::UnsupportedTask {
            backend: self.name(),
            task: "blindfold",
        })
    }
}
