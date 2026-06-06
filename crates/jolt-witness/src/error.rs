use thiserror::Error;

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum WitnessError {
    #[error("unknown witness oracle in namespace `{namespace}`")]
    UnknownOracle { namespace: &'static str },
    #[error("requested witness view is unavailable in namespace `{namespace}`")]
    UnavailableView { namespace: &'static str },
    #[error("invalid witness dimensions in namespace `{namespace}`: {reason}")]
    InvalidDimensions {
        namespace: &'static str,
        reason: String,
    },
    #[error("invalid witness data in namespace `{namespace}`: {reason}")]
    InvalidWitnessData {
        namespace: &'static str,
        reason: String,
    },
    #[error("witness provider does not support this frontier: {frontier}")]
    UnsupportedFrontier { frontier: &'static str },
}
