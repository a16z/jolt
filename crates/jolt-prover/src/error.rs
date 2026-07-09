use jolt_claims::protocols::jolt::{JoltChallengeId, JoltDerivedId, JoltOpeningId};
use jolt_claims::MissingOpeningValue;
use jolt_field::FieldCore;
use jolt_sumcheck::SumcheckError;
use jolt_verifier::VerifierError;
use thiserror::Error;

/// Errors surfaced while proving. The engine-level failures come through
/// [`SumcheckError`]; relation-level failures (claim wiring, point derivation)
/// through [`VerifierError`] — the prover runs the verifier's own relation
/// methods as hard self-checks, so their errors are prover errors here.
#[derive(Debug, Error)]
pub enum ProverError<F: FieldCore> {
    #[error(transparent)]
    Sumcheck(#[from] SumcheckError<F>),

    #[error(transparent)]
    Verifier(#[from] VerifierError),

    #[error(transparent)]
    MissingOpeningValue(#[from] MissingOpeningValue<JoltOpeningId>),

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
}
