use crate::SumcheckKernelError;
use jolt_claims::protocols::jolt::{JoltChallengeId, JoltDerivedId, JoltOpeningId};
use jolt_claims::MissingOpeningValue;
use jolt_field::FieldCore;
use jolt_sumcheck::SumcheckError;
use jolt_verifier::VerifierError;
use jolt_witness::WitnessError;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum KernelError<F: FieldCore> {
    #[error(transparent)]
    Witness(#[from] WitnessError),

    #[error(transparent)]
    Openings(#[from] jolt_openings::OpeningsError),

    #[error(transparent)]
    Sumcheck(#[from] SumcheckError<F>),

    /// Relation-level failures (claim wiring, point derivation): kernels run
    /// the verifier's own relation methods as hard self-checks, so their
    /// errors are kernel errors here.
    #[error(transparent)]
    Verifier(#[from] VerifierError),

    #[error(transparent)]
    MissingOpeningValue(#[from] MissingOpeningValue<JoltOpeningId>),

    /// Extraction/self-check failures from the typed kernel seam
    /// (`SumcheckKernel::{output_claims, validate_derived_tables}`).
    #[error(transparent)]
    SumcheckKernel(#[from] SumcheckKernelError<F>),

    #[error(transparent)]
    CenteredDomain(#[from] jolt_poly::lagrange::CenteredIntegerDomainError),

    #[error(transparent)]
    ConstraintMatrix(#[from] jolt_r1cs::constraint::ConstraintMatrixEvalError),

    /// A polynomial's dimensions are incompatible with the commitment grid.
    #[error("invalid commitment geometry: {reason}")]
    InvalidGeometry { reason: String },

    /// A stream produced a chunk the kernel cannot place in the grid.
    #[error("unsupported polynomial chunk: {reason}")]
    UnsupportedChunk { reason: String },

    /// A relation's output expression references an opening with no table.
    #[error("no table for opening {id:?}")]
    MissingOpeningTable { id: JoltOpeningId },

    /// A relation's output expression references a derived term with no table.
    #[error("no table for derived term {id:?}")]
    MissingDerivedTable { id: JoltDerivedId },

    /// A relation's output expression references a challenge the drawn
    /// `Challenges` struct does not carry.
    #[error("no drawn challenge for {id:?}")]
    MissingChallenge { id: JoltChallengeId },

    /// A leaf table's evaluation count disagrees with the relation's rounds.
    #[error("table for {table} has {got} evaluations, expected {expected}")]
    TableSizeMismatch {
        /// The offending table's opening or derived id, debug-formatted.
        table: String,
        expected: usize,
        got: usize,
    },

    /// Final values were requested before every round was bound.
    #[error("final table values requested with {remaining} unbound rounds")]
    NotFullyBound { remaining: usize },

    /// A bound derived table's final value disagrees with the verifier's
    /// `derive_output_term` at the bound point — the hand-written table
    /// resolver drifted from the relation's scalar path.
    #[error("derived table {id:?} bound to {got}, but derive_output_term gives {expected}")]
    DerivedTableDrift {
        id: JoltDerivedId,
        expected: F,
        got: F,
    },

    /// A capability the kernel does not implement yet. Recoverable in
    /// principle: a caller may retry the slot against a different backend.
    #[error("unsupported: {reason}")]
    Unsupported { reason: &'static str },

    /// A contract the kernel's inputs must uphold (witness shape, point
    /// geometry, internal state) was violated — a bug, never a capability
    /// gap, so never worth retrying against another backend.
    #[error("kernel invariant violated: {reason}")]
    InvariantViolation { reason: &'static str },
}
