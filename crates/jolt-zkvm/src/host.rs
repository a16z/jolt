//! Host-level proving and verification API.
//!
//! Bridges the tracer's execution trace to the full proving pipeline.
//! [`prove_trace`] is the top-level entry point: it takes a RISC-V execution
//! trace and produces a complete Jolt proof.

use std::sync::Arc;

use jolt_compute::ComputeBackend;
use jolt_field::{Field, WithChallenge};
use jolt_ir::zkvm::tags::poly;
use jolt_openings::{AdditivelyHomomorphic, CommitmentScheme};
use jolt_transcript::Transcript;
use jolt_verifier::StageDescriptor;
use tracer::instruction::Cycle;

use crate::preprocessing::{preprocess, JoltConfig};
use crate::proof::{JoltProof, JoltProvingKey};
use crate::prover::{prove, ProveError};
use crate::stage::ProverStage;
use crate::stages::s3_claim_reductions::ClaimReductionStage;
use crate::witness::generate::generate_witnesses;

/// Output of [`prove_trace`]: a proof and the verifying key needed to check it.
pub struct ProveOutput<F: Field, PCS: CommitmentScheme<Field = F>> {
    pub proof: JoltProof<F, PCS>,
    pub verifying_key: jolt_verifier::JoltVerifyingKey<F, PCS>,
    /// Polynomial evaluation tables for committed polynomials.
    /// Needed by the verifier descriptor builder in tests; in production
    /// the verifier would derive claimed_sums from prior claim evaluations.
    pub committed_tables: CommittedTables<F>,
}

/// Committed polynomial evaluation tables carried alongside the proof
/// so the verifier (in tests) can reconstruct claimed sums.
pub struct CommittedTables<F: Field> {
    pub rd_inc: Vec<F>,
    pub ram_inc: Vec<F>,
}

/// Proves an execution trace through the full Jolt pipeline.
///
/// Orchestrates: trace → witness gen → preprocess → commit → prove.
///
/// The `backend` parameter controls which compute backend is used for
/// sumcheck kernel evaluation. Pass `Arc::new(CpuBackend)` for CPU-only
/// or `Arc::new(HybridBackend::new(MetalBackend::new(), CpuBackend, threshold))`
/// for GPU-accelerated proving with automatic fallback.
///
/// Currently wires up S3 (increment claim reduction) as the sole sumcheck
/// stage, using RD_INC and RAM_INC from witness generation. Additional
/// stages (S2, S4–S7) will be added incrementally.
#[tracing::instrument(skip_all, name = "prove_trace")]
pub fn prove_trace<PCS, B>(
    trace: &[Cycle],
    pcs_setup: impl FnOnce(usize) -> (PCS::ProverSetup, PCS::VerifierSetup),
    backend: Arc<B>,
) -> Result<ProveOutput<PCS::Field, PCS>, ProveError>
where
    PCS: AdditivelyHomomorphic,
    PCS::Field: WithChallenge,
    <PCS::Field as WithChallenge>::Challenge: From<u128>,
    B: ComputeBackend,
{
    let output = generate_witnesses::<PCS::Field>(trace);

    let config = JoltConfig {
        num_cycles: output.cycle_witnesses.len(),
    };
    let key: JoltProvingKey<PCS::Field, PCS> = preprocess(&config, pcs_setup);

    // Extract committed polynomial tables from witness store.
    let rd_inc = output.witness_store.get(poly::RD_INC).to_vec();
    let ram_inc = output.witness_store.get(poly::RAM_INC).to_vec();

    // Commit stage polynomials. Order must match ClaimReductionStage::increment
    // which produces claims as [ram_inc, rd_inc].
    let (com_ram_inc, _) = PCS::commit(&ram_inc, &key.pcs_prover_setup);
    let (com_rd_inc, _) = PCS::commit(&rd_inc, &key.pcs_prover_setup);
    let poly_commitments = vec![com_ram_inc, com_rd_inc];

    let mut transcript = jolt_transcript::Blake2bTranscript::new(b"jolt-v2");

    let proof = prove::<PCS, jolt_transcript::Blake2bTranscript>(
        &key,
        &output.cycle_witnesses,
        poly_commitments,
        |_r_x, r_y, transcript| {
            build_prover_stages(&rd_inc, &ram_inc, r_y, transcript, Arc::clone(&backend))
        },
        &mut transcript,
    )?;

    let verifying_key = jolt_verifier::JoltVerifyingKey {
        spartan_key: key.spartan_key,
        pcs_setup: key.pcs_verifier_setup,
    };

    Ok(ProveOutput {
        proof,
        verifying_key,
        committed_tables: CommittedTables { rd_inc, ram_inc },
    })
}

/// Verifies a proof produced by [`prove_trace`].
///
/// The `committed_tables` are needed to reconstruct claimed sums for
/// the verifier stage descriptors. In production, these would be derived
/// from the claim threading chain instead.
pub fn verify_proof<PCS>(
    output: &ProveOutput<PCS::Field, PCS>,
) -> Result<(), jolt_verifier::JoltError>
where
    PCS: AdditivelyHomomorphic,
    PCS::Field: WithChallenge,
    <PCS::Field as WithChallenge>::Challenge: From<u128>,
{
    let mut transcript = jolt_transcript::Blake2bTranscript::new(b"jolt-v2");

    let tables = &output.committed_tables;

    let _ = jolt_verifier::verify::<PCS, jolt_transcript::Blake2bTranscript>(
        &output.proof,
        &output.verifying_key,
        |_r_x, r_y, transcript| {
            build_verifier_descriptors(&tables.rd_inc, &tables.ram_inc, r_y, transcript)
        },
        &mut transcript,
    )?;

    Ok(())
}

/// Builds prover stages for S2–S7.
///
/// Currently implements:
/// - S3 increment reduction: `c0·rd_inc + c1·ram_inc` with Fiat-Shamir coefficients
fn build_prover_stages<F, T, B>(
    rd_inc: &[F],
    ram_inc: &[F],
    r_y: &[F],
    transcript: &mut T,
    backend: Arc<B>,
) -> Vec<Box<dyn ProverStage<F, T>>>
where
    F: WithChallenge,
    F::Challenge: From<T::Challenge>,
    T: Transcript,
    B: ComputeBackend,
{
    let c0: F = F::Challenge::from(transcript.challenge()).into();
    let c1: F = F::Challenge::from(transcript.challenge()).into();

    // r_y = [r_cycle..., r_var...]; committed polys are indexed by cycle only.
    let num_cycle_vars = rd_inc.len().trailing_zeros() as usize;
    let r_cycle = &r_y[..num_cycle_vars];

    let inc_stage = ClaimReductionStage::increment(
        ram_inc.to_vec(),
        rd_inc.to_vec(),
        r_cycle.to_vec(),
        c0,
        c1,
        backend,
    );

    vec![Box::new(inc_stage)]
}

/// Builds verifier stage descriptors matching [`build_prover_stages`].
fn build_verifier_descriptors<F, T>(
    rd_inc: &[F],
    ram_inc: &[F],
    r_y: &[F],
    transcript: &mut T,
) -> Vec<StageDescriptor<F>>
where
    F: WithChallenge,
    F::Challenge: From<T::Challenge>,
    T: Transcript,
{
    use jolt_poly::EqPolynomial;

    let c0: F = F::Challenge::from(transcript.challenge()).into();
    let c1: F = F::Challenge::from(transcript.challenge()).into();

    // r_y = [r_cycle..., r_var...]; committed polys are indexed by cycle only.
    let num_cycle_vars = rd_inc.len().trailing_zeros() as usize;
    let r_cycle = &r_y[..num_cycle_vars];

    let n = rd_inc.len();
    let eq_table = EqPolynomial::new(r_cycle.to_vec()).evaluations();

    let claimed_sum: F = (0..n)
        .map(|j| eq_table[j] * (c0 * ram_inc[j] + c1 * rd_inc[j]))
        .sum();

    let desc = StageDescriptor::claim_reduction(
        r_cycle.to_vec(),
        vec![c0, c1],
        claimed_sum,
        vec![0, 1],
    )
    .with_reverse_challenges();

    vec![desc]
}
