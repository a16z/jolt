use jolt_field::FieldCore;
use jolt_kernels::KernelError;
use jolt_sumcheck::SumcheckError;
use jolt_verifier::VerifierError;
use thiserror::Error;

/// Errors surfaced while proving. The engine-level failures come through
/// [`SumcheckError`], compute failures through [`KernelError`], and
/// relation-level failures (claim wiring, point derivation) through
/// [`VerifierError`] — the prover runs the verifier's own relation methods as
/// hard self-checks, so their errors are prover errors here.
#[derive(Debug, Error)]
pub enum ProverError<F: FieldCore> {
    #[error(transparent)]
    Sumcheck(#[from] SumcheckError<F>),

    #[error(transparent)]
    Verifier(#[from] VerifierError),

    #[error(transparent)]
    Kernel(#[from] KernelError<F>),

    #[error(transparent)]
    Witness(#[from] jolt_witness::WitnessError),

    /// A stage's final running claim disagrees with the verifier's
    /// `expected_final_claim` fold over the produced openings.
    #[error("{stage}: final claim {got} != expected {expected}")]
    FinalClaimMismatch {
        stage: &'static str,
        expected: F,
        got: F,
    },

    /// A capability the modular prover does not implement yet.
    #[error("unsupported: {reason}")]
    Unsupported { reason: &'static str },
}
