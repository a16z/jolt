use thiserror::Error;

#[derive(Debug, Error)]
pub enum ProverError {
    #[error("prover frontier is not implemented yet: {frontier}")]
    FrontierNotImplemented { frontier: &'static str },
    #[error("invalid prover configuration: {reason}")]
    InvalidProverConfig { reason: String },
    #[error("invalid commitment output result: {reason}")]
    InvalidCommitmentOutput { reason: String },
    #[error("invalid stage request: {reason}")]
    InvalidStageRequest { reason: String },
    #[error("invalid sumcheck output result: {reason}")]
    InvalidSumcheckOutput { reason: String },
    #[error(transparent)]
    Backend(#[from] jolt_backends::BackendError),
    #[error(transparent)]
    Verifier(#[from] jolt_verifier::VerifierError),
    #[error(transparent)]
    Witness(#[from] jolt_witness::WitnessError),
}
