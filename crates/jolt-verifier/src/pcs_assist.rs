//! Generic PCS-assist verifier boundary.

use std::fmt::Debug;

use jolt_openings::{CommitmentScheme, ZkOpeningScheme};
use jolt_transcript::Transcript;
use serde::{de::DeserializeOwned, Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct NoPcsAssistConfig;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct NoPcsAssistProof;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct NoPcsAssist;

#[derive(Debug, thiserror::Error)]
#[error("PCS assist is disabled for this verifier build")]
pub struct NoPcsAssistError;

pub struct PcsAssistClearInput<'a, PCS: CommitmentScheme> {
    pub setup: &'a PCS::VerifierSetup,
    pub pcs_proof: &'a PCS::Proof,
    pub commitment: &'a PCS::Output,
    pub point: &'a [PCS::Field],
    pub eval: PCS::Field,
}

pub struct PcsAssistZkInput<'a, PCS: CommitmentScheme> {
    pub setup: &'a PCS::VerifierSetup,
    pub pcs_proof: &'a PCS::Proof,
    pub commitment: &'a PCS::Output,
    pub point: &'a [PCS::Field],
}

pub trait PcsProofAssist<PCS: CommitmentScheme> {
    type Proof: Clone + Debug + Eq + Send + Sync + 'static + Serialize + DeserializeOwned;
    type Config: Clone + Debug + Eq + Send + Sync + 'static + Serialize + DeserializeOwned;
    type Error: std::error::Error + Send + Sync + 'static;

    fn selected_config() -> Self::Config;

    fn verify_clear<T>(
        config: &Self::Config,
        input: PcsAssistClearInput<'_, PCS>,
        proof: &Self::Proof,
        transcript: &mut T,
    ) -> Result<(), Self::Error>
    where
        T: Transcript<Challenge = PCS::Field>;

    fn verify_zk<T>(
        config: &Self::Config,
        input: PcsAssistZkInput<'_, PCS>,
        proof: &Self::Proof,
        transcript: &mut T,
    ) -> Result<<PCS as ZkOpeningScheme>::HidingCommitment, Self::Error>
    where
        PCS: ZkOpeningScheme,
        T: Transcript<Challenge = PCS::Field>;
}

impl<PCS> PcsProofAssist<PCS> for NoPcsAssist
where
    PCS: CommitmentScheme,
{
    type Proof = NoPcsAssistProof;
    type Config = NoPcsAssistConfig;
    type Error = NoPcsAssistError;

    fn selected_config() -> Self::Config {
        NoPcsAssistConfig
    }

    fn verify_clear<T>(
        _config: &Self::Config,
        _input: PcsAssistClearInput<'_, PCS>,
        _proof: &Self::Proof,
        _transcript: &mut T,
    ) -> Result<(), Self::Error>
    where
        T: Transcript<Challenge = PCS::Field>,
    {
        Err(NoPcsAssistError)
    }

    fn verify_zk<T>(
        _config: &Self::Config,
        _input: PcsAssistZkInput<'_, PCS>,
        _proof: &Self::Proof,
        _transcript: &mut T,
    ) -> Result<<PCS as ZkOpeningScheme>::HidingCommitment, Self::Error>
    where
        PCS: ZkOpeningScheme,
        T: Transcript<Challenge = PCS::Field>,
    {
        Err(NoPcsAssistError)
    }
}
