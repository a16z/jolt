use thiserror::Error;

#[derive(Debug, Error)]
pub enum BackendError {
    #[error("backend `{backend}` does not support `{task}`")]
    UnsupportedTask {
        backend: &'static str,
        task: &'static str,
    },
    #[error(transparent)]
    Witness(#[from] jolt_witness::WitnessError),
}
