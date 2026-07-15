use thiserror::Error;

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum WitnessError {
    #[error("unknown witness oracle for `{label}`")]
    UnknownOracle { label: &'static str },
    /// The id is valid protocol vocabulary but deliberately outside this
    /// backend's serving set; `reason` documents the classification.
    #[error("witness oracle {oracle} is not served by this backend: {reason}")]
    NotServed {
        oracle: String,
        reason: &'static str,
    },
    #[error("requested witness view is unavailable for `{label}`")]
    UnavailableView { label: &'static str },
    #[error("invalid witness dimensions for `{label}`: {reason}")]
    InvalidDimensions { label: &'static str, reason: String },
    #[error("invalid witness data for `{label}`: {reason}")]
    InvalidWitnessData { label: &'static str, reason: String },
    #[error("witness provider does not support this view: {view}")]
    UnsupportedView { view: &'static str },
}
