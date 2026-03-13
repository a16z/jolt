//! Top-level proving and verification API.
//!
//! [`prove`] is the main entry point: it takes a RISC-V execution trace
//! and produces a complete Jolt proof. [`verify`] checks a proof against
//! its verifying key.
//!
//! For lower-level control, [`prove_pipeline`] accepts pre-processed keys
//! and a stage factory closure.

use std::sync::Arc;

use jolt_compute::ComputeBackend;
use jolt_field::Field;
use jolt_ir::zkvm::tags::poly;
use jolt_openings::{AdditivelyHomomorphic, CommitmentScheme, OpeningReduction, RlcReduction};
use jolt_poly::EqPolynomial;
use jolt_spartan::SpartanError;
use jolt_transcript::Transcript;
use tracer::instruction::Cycle;

use crate::pipeline::prove_stages;
use crate::preprocessing::{interleave_witnesses, preprocess, JoltConfig};
use crate::proof::{JoltProof, JoltProvingKey};
use crate::stage::ProverStage;
use crate::stages::s1_spartan::UniformSpartanStage;
use crate::stages::s3_claim_reductions::ClaimReductionStage;
use crate::witness::generate::generate_witnesses;
use jolt_verifier::{ProverConfig, StageDescriptor};

/// Output of [`prove`]: a proof and the verifying key needed to check it.
pub struct ProveOutput<F: Field, PCS: CommitmentScheme<Field = F>> {
    /// The complete Jolt proof.
    pub proof: JoltProof<F, PCS>,
    /// Verifying key (Spartan key + PCS verifier setup).
    pub verifying_key: jolt_verifier::JoltVerifyingKey<F, PCS>,
    /// Committed polynomial evaluation tables.
    /// Needed by the verifier descriptor builder in tests; in production
    /// the verifier derives claimed sums from the claim threading chain.
    pub committed_tables: CommittedTables<F>,
}

/// Committed polynomial evaluation tables carried alongside the proof
/// so the verifier (in tests) can reconstruct claimed sums.
pub struct CommittedTables<F: Field> {
    pub rd_inc: Vec<F>,
    pub ram_inc: Vec<F>,
}

/// Errors that can occur during proving.
#[derive(Debug)]
pub enum ProveError {
    /// Spartan R1CS proving failed (constraint violation or sumcheck error).
    Spartan(SpartanError),
}

impl From<SpartanError> for ProveError {
    fn from(e: SpartanError) -> Self {
        ProveError::Spartan(e)
    }
}

impl std::fmt::Display for ProveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProveError::Spartan(e) => write!(f, "spartan: {e}"),
        }
    }
}

impl std::error::Error for ProveError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ProveError::Spartan(e) => Some(e),
        }
    }
}

/// Proves a RISC-V execution trace through the full Jolt pipeline.
///
/// Orchestrates: trace → witness gen → preprocess → commit → prove.
///
/// The `backend` parameter controls which compute backend is used for
/// sumcheck kernel evaluation. Pass `Arc::new(CpuBackend)` for CPU-only
/// or a hybrid backend for GPU-accelerated proving.
///
/// Currently wires S3 (increment claim reduction) as the sole sumcheck
/// stage. Additional stages (S2, S4–S7) will be wired incrementally.
#[tracing::instrument(skip_all, name = "prove")]
pub fn prove<PCS, B>(
    trace: &[Cycle],
    pcs_setup: impl FnOnce(usize) -> (PCS::ProverSetup, PCS::VerifierSetup),
    backend: Arc<B>,
) -> Result<ProveOutput<PCS::Field, PCS>, ProveError>
where
    PCS: AdditivelyHomomorphic,
    B: ComputeBackend,
{
    let output = generate_witnesses::<PCS::Field>(trace);

    let config = JoltConfig {
        num_cycles: output.cycle_witnesses.len(),
    };
    let key: JoltProvingKey<PCS::Field, PCS> = preprocess(&config, pcs_setup);

    let rd_inc = output.witness_store.get(poly::RD_INC).to_vec();
    let ram_inc = output.witness_store.get(poly::RAM_INC).to_vec();

    let (com_ram_inc, _) = PCS::commit(&ram_inc, &key.pcs_prover_setup);
    let (com_rd_inc, _) = PCS::commit(&rd_inc, &key.pcs_prover_setup);
    let poly_commitments = vec![com_ram_inc, com_rd_inc];

    let mut transcript = jolt_transcript::Blake2bTranscript::<PCS::Field>::new(b"jolt-v2");

    let proof = prove_pipeline::<PCS, jolt_transcript::Blake2bTranscript<PCS::Field>>(
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

/// Verifies a proof produced by [`prove`].
///
/// The `committed_tables` are needed to reconstruct claimed sums for
/// the verifier stage descriptors. In production, these would be derived
/// from the claim threading chain instead.
pub fn verify<PCS>(output: &ProveOutput<PCS::Field, PCS>) -> Result<(), jolt_verifier::JoltError>
where
    PCS: AdditivelyHomomorphic,
{
    let mut transcript = jolt_transcript::Blake2bTranscript::<PCS::Field>::new(b"jolt-v2");

    let tables = &output.committed_tables;

    let _ = jolt_verifier::verify::<PCS, jolt_transcript::Blake2bTranscript<PCS::Field>>(
        &output.proof,
        &output.verifying_key,
        |_r_x, r_y, transcript| {
            build_verifier_descriptors(&tables.rd_inc, &tables.ram_inc, r_y, transcript)
        },
        &mut transcript,
    )?;

    Ok(())
}

/// Runs the Jolt proving pipeline from pre-processed inputs.
///
/// This is the lower-level entry point used by [`prove`] and by tests
/// that construct their own witness data. It orchestrates:
///
/// 1. Interleave per-cycle witnesses into the flat R1CS witness
/// 2. Commit the witness and append commitment to transcript
/// 3. Run uniform Spartan (S1) to produce the R1CS proof
/// 4. Build sumcheck stages via `build_stages(r_x, r_y)`
/// 5. Run S2–S7 sumcheck stages
/// 6. Collect all opening claims (stages + Spartan witness)
/// 7. RLC-reduce and produce batch PCS opening proofs (S8)
pub fn prove_pipeline<PCS, T>(
    key: &JoltProvingKey<PCS::Field, PCS>,
    cycle_witnesses: &[Vec<PCS::Field>],
    poly_commitments: Vec<PCS::Output>,
    build_stages: impl FnOnce(
        &[PCS::Field],
        &[PCS::Field],
        &mut T,
    ) -> Vec<Box<dyn ProverStage<PCS::Field, T>>>,
    transcript: &mut T,
) -> Result<JoltProof<PCS::Field, PCS>, ProveError>
where
    PCS: AdditivelyHomomorphic,
    T: Transcript<Challenge = PCS::Field>,
{
    // S0: Interleave per-cycle witnesses and commit.
    let (flat_witness, witness_commitment) = {
        let _span = tracing::info_span!("S0_witness_commit").entered();
        let flat_witness = interleave_witnesses(&key.spartan_key, cycle_witnesses);
        let (witness_commitment, _) = PCS::commit(&flat_witness, &key.pcs_prover_setup);
        transcript.append_bytes(format!("{witness_commitment:?}").as_bytes());
        tracing::info!(
            num_cycles = key.spartan_key.num_cycles,
            witness_len = flat_witness.len(),
            "witness committed"
        );
        (flat_witness, witness_commitment)
    };

    // S1: Uniform Spartan PIOP.
    let spartan_result = {
        let _span = tracing::info_span!("S1_spartan").entered();
        UniformSpartanStage::prove(&key.spartan_key, &flat_witness, &flat_witness, transcript)?
    };

    // Build stages using Spartan challenge vectors. Transcript is passed so
    // the factory can squeeze batching challenges at the correct Fiat-Shamir state.
    let mut stages = build_stages(&spartan_result.r_x, &spartan_result.r_y, transcript);

    // S2–S7: Sumcheck stages.
    let (stage_proofs, mut opening_claims) = prove_stages(&mut stages, transcript);

    // Spartan witness opening claim — added last to match verifier ordering.
    opening_claims.push(spartan_result.witness_opening_claim);

    // S8: RLC reduction + PCS opening proofs.
    let proofs = {
        let _span = tracing::info_span!("S8_opening_proofs").entered();
        tracing::info!(total_claims = opening_claims.len(), "reducing and opening");

        let (reduced, ()) =
            <RlcReduction as OpeningReduction<PCS>>::reduce_prover(opening_claims, transcript);

        tracing::info!(reduced_claims = reduced.len(), "opening PCS proofs");

        reduced
            .into_iter()
            .map(|claim| {
                let poly: PCS::Polynomial = claim.evaluations.into();
                PCS::open(
                    &poly,
                    &claim.point,
                    claim.eval,
                    &key.pcs_prover_setup,
                    None,
                    transcript,
                )
            })
            .collect()
    };

    Ok(JoltProof {
        config: ProverConfig {
            trace_length: key.spartan_key.num_cycles,
            ram_k: 0,
            one_hot_config: jolt_verifier::OneHotConfig::new(
                key.spartan_key.num_cycles.trailing_zeros() as usize,
            ),
            rw_config: jolt_verifier::ReadWriteConfig::new(
                key.spartan_key.num_cycles.trailing_zeros() as usize,
                0,
            ),
        },
        spartan_proof: spartan_result.proof,
        stage_proofs,
        opening_proofs: proofs,
        witness_commitment,
        commitments: poly_commitments,
    })
}

/// Builds prover stages for S2–S7.
///
/// Currently implements S3 increment reduction only.
fn build_prover_stages<F, T, B>(
    rd_inc: &[F],
    ram_inc: &[F],
    r_y: &[F],
    transcript: &mut T,
    backend: Arc<B>,
) -> Vec<Box<dyn ProverStage<F, T>>>
where
    F: Field,
    T: Transcript<Challenge = F>,
    B: ComputeBackend,
{
    let c0: F = transcript.challenge();
    let c1: F = transcript.challenge();

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
    F: Field,
    T: Transcript<Challenge = F>,
{
    let c0: F = transcript.challenge();
    let c1: F = transcript.challenge();

    let num_cycle_vars = rd_inc.len().trailing_zeros() as usize;
    let r_cycle = &r_y[..num_cycle_vars];

    let n = rd_inc.len();
    let eq_table = EqPolynomial::new(r_cycle.to_vec()).evaluations();

    let claimed_sum: F = (0..n)
        .map(|j| eq_table[j] * (c0 * ram_inc[j] + c1 * rd_inc[j]))
        .sum();

    let desc =
        StageDescriptor::claim_reduction(r_cycle.to_vec(), vec![c0, c1], claimed_sum, vec![0, 1])
            .with_reverse_challenges();

    vec![desc]
}
