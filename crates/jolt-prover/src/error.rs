use jolt_field::FieldCore;
use jolt_kernels::{KernelError, SumcheckKernelError};
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

    /// Extraction/self-check failures from the typed kernel seam.
    #[error(transparent)]
    SumcheckKernel(#[from] SumcheckKernelError<F>),

    #[error(transparent)]
    Witness(#[from] jolt_witness::WitnessError),

    /// A capability the modular prover does not implement yet, or an input
    /// regime it rejects up front. Recoverable in principle: the caller may
    /// fall back to another prover.
    #[error("unsupported: {reason}")]
    Unsupported { reason: &'static str },

    /// A cross-stage carry or kernel contract the prover itself must uphold
    /// was violated — a prover bug, never a capability gap, so never worth
    /// retrying with different inputs or another backend.
    #[error("prover invariant violated: {reason}")]
    InvariantViolation { reason: &'static str },
}
