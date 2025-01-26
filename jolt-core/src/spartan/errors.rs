use core::{
    fmt::Display,
    fmt::{self, Debug},
};

#[derive(Debug, Default)]
pub enum ProofVerifyError {
    #[default]
    InternalError,
    DecompressionError([u8; 32]),
}

impl Display for ProofVerifyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self {
            ProofVerifyError::DecompressionError(bytes) => write!(
                f,
                "Compressed group element failed to decompress: {bytes:?}",
            ),
            ProofVerifyError::InternalError => {
                write!(f, "Proof verification failed",)
            }
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum R1CSError {
    /// returned if the number of constraints is not a power of 2
    NonPowerOfTwoCons,
    /// returned if the number of variables is not a power of 2
    NonPowerOfTwoVars,
    /// returned if a wrong number of inputs in an assignment are supplied
    InvalidNumberOfInputs,
    /// returned if a wrong number of variables in an assignment are supplied
    InvalidNumberOfVars,
    /// returned if a [u8;32] does not parse into a valid Scalar in the field of ristretto255
    InvalidScalar,
    /// returned if the supplied row or col in (row,col,val) tuple is out of range
    InvalidIndex,
}
