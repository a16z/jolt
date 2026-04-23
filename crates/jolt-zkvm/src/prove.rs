//! Top-level prover API.
//!
//! [`prove`] takes a pre-built [`BufferProvider`] and [`Executable`],
//! validates the config, and executes the proving schedule.

use jolt_compute::{BufferProvider, ComputeBackend, Executable};
use jolt_crypto::HomomorphicCommitment;
use jolt_field::Field;
use jolt_openings::{AdditivelyHomomorphic, OpeningVerification};
use jolt_transcript::{AppendToTranscript, Transcript};
use jolt_verifier::ProverConfig;

use crate::runtime::execute;
/// Execute the proving pipeline.
///
/// The caller is responsible for building witness polynomials, R1CS data,
/// and assembling them into a [`BufferProvider`]. This function validates
/// the config and executes the compiled schedule.
///
/// # Panics
///
/// Panics if `config` is invalid (non-power-of-two trace length, etc.).
pub fn prove<B, F, T, PCS>(
    executable: &Executable<B, F>,
    provider: &mut impl BufferProvider<F>,
    backend: &B,
    pcs_setup: &PCS::ProverSetup,
    transcript: &mut T,
    config: ProverConfig,
) -> jolt_verifier::JoltProof<F, PCS>
where
    B: ComputeBackend,
    F: Field,
    T: Transcript<Challenge = F>,
    PCS: AdditivelyHomomorphic<Field = F> + OpeningVerification,
    PCS::Output: AppendToTranscript + HomomorphicCommitment<F>,
{
    if let Err(e) = config.validate() {
        panic!("invalid ProverConfig: {e}");
    }
    execute::<B, F, T, PCS>(executable, provider, backend, pcs_setup, transcript, config)
}
