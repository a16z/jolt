use thiserror::Error;

#[derive(Debug, Error)]
pub enum BackendError {
    #[error("backend `{backend}` does not support `{task}`")]
    UnsupportedTask {
        backend: &'static str,
        task: &'static str,
    },
    #[error("backend `{backend}` received invalid `{task}` request: {reason}")]
    InvalidRequest {
        backend: &'static str,
        task: &'static str,
        reason: String,
    },
    #[error(transparent)]
    Witness(#[from] jolt_witness::WitnessError),
}
