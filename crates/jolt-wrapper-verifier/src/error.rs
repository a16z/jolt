use jolt_claims::protocols::wrapper_spartan_hyperkzg::WrapperRelationDimensions;
use jolt_r1cs::{R1csBuilderError, Variable};
use thiserror::Error as ThisError;

#[derive(Clone, Debug, ThisError, PartialEq, Eq)]
pub enum WrapperError {
    #[error(transparent)]
    R1csBuilder(#[from] R1csBuilderError),
    #[error("assigned scalar {name} is not backed by one allocated variable")]
    AssignedScalarNotVariable { name: &'static str },
    #[error("witness violates R1CS constraint {constraint}")]
    UnsatisfiedConstraint { constraint: usize },
    #[error("public input {index} does not match witness variable {variable:?}")]
    PublicInputMismatch { index: usize, variable: Variable },
    #[error("public input count mismatch: layout has {expected}, but protocol has {actual}")]
    PublicInputCountMismatch { expected: usize, actual: usize },
    #[error("variable {variable:?} is not present in witness")]
    MissingWitnessVariable { variable: Variable },
    #[error("R1CS relation proof shape mismatch: expected {expected:?}, got {actual:?}")]
    R1csRelationMismatch {
        expected: WrapperRelationDimensions,
        actual: WrapperRelationDimensions,
    },
    #[error("R1CS relation dimension {dimension}={value} does not fit in u64")]
    R1csRelationDimensionTooLarge {
        dimension: &'static str,
        value: usize,
    },
    #[error("invalid wrapper R1CS relation facts: {reason}")]
    InvalidR1csRelationFacts { reason: String },
    #[error("R1CS matrix MLE evaluation failed: {reason}")]
    R1csMatrixEvaluationFailed { reason: String },
    #[error("Spartan sumcheck failed: {reason}")]
    SpartanSumcheckFailed { reason: String },
    #[error("Spartan outer reduction does not match R1CS matrix claims")]
    SpartanOuterReductionMismatch,
    #[error("Spartan inner reduction does not match the witness opening claim")]
    SpartanInnerReductionMismatch,
    #[error("HyperKZG verification failed: {reason}")]
    HyperKzgVerificationFailed { reason: String },
    #[error("expected committed proof for {proof}")]
    ExpectedCommittedProof { proof: &'static str },
    #[error("invalid vector commitment capacity: required at least {required}, got {got}")]
    InvalidVectorCommitmentCapacity { required: usize, got: usize },
    #[error("BlindFold construction failed: {reason}")]
    BlindFoldConstructionFailed { reason: String },
    #[error("BlindFold verification failed: {reason}")]
    BlindFoldVerificationFailed { reason: String },
    #[error("{component} is not implemented yet")]
    Unimplemented { component: &'static str },
}

pub type Error = WrapperError;
