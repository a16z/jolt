use jolt_r1cs::{R1csBuilderError, Variable};
use thiserror::Error as ThisError;

#[derive(Clone, Debug, ThisError, PartialEq, Eq)]
pub enum Error {
    #[error(transparent)]
    R1csBuilder(#[from] R1csBuilderError),
    #[error("assigned scalar {name} is not backed by one allocated variable")]
    AssignedScalarNotVariable { name: &'static str },
    #[error("witness violates R1CS constraint {constraint}")]
    UnsatisfiedConstraint { constraint: usize },
    #[error("public input {index} does not match witness variable {variable:?}")]
    PublicInputMismatch { index: usize, variable: Variable },
    #[error("variable {variable:?} is not present in witness")]
    MissingWitnessVariable { variable: Variable },
    #[error("SNARK backend {backend} is not implemented yet")]
    BackendNotImplemented { backend: &'static str },
    #[error("{component} is not implemented yet")]
    Unimplemented { component: &'static str },
}
