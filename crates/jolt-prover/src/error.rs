use thiserror::Error;

#[derive(Debug, Error)]
pub enum ProverError {
    #[error("prover frontier is not implemented yet: {frontier}")]
    FrontierNotImplemented { frontier: &'static str },
    #[error("invalid commitment output result: {reason}")]
    InvalidCommitmentOutput { reason: String },
    #[error(transparent)]
    Backend(#[from] jolt_backends::BackendError),
    #[error(transparent)]
    Witness(#[from] jolt_witness::WitnessError),
}
