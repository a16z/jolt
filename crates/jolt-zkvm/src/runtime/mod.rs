//! Prover runtime: execute a linked schedule to produce a proof.
//!
//! [`execute`] walks the complete [`Op`] sequence — compute, PCS, and
//! orchestration ops — and returns a [`JoltProof`]. It dispatches to:
//! - [`ComputeBackend`] for polynomial arithmetic (sumcheck, bind, evaluate)
//! - [`CommitmentScheme`] for cryptographic ops (commit, reduce, open)
//! - Direct calls for orchestration (transcript absorb/squeeze, lifecycle)
//!
//! ```text
//! Protocol → compile() → Module → link(backend) → Executable<B,F>
//!                                                       │
//!                                          execute(exe, provider, backend, pcs, transcript)
//!                                                       │
//!                                                       ▼
//!                                                 JoltProof<F, PCS>
//! ```

mod cpu_clock;
mod handlers;
mod helpers;
pub mod prefix_suffix;

use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

/// Short static tag for op-class saturation instrumentation. Returns `None` for
/// microsecond-scale orchestration ops (transcript absorb/squeeze/release)
/// whose per-call getrusage overhead would be a large relative cost.
fn op_class_tag(op: &jolt_compiler::module::Op) -> Option<&'static str> {
    use jolt_compiler::module::Op;
    Some(match op {
        Op::Materialize { .. } => "Materialize",
        Op::MaterializeUnlessFresh { .. } => "MaterializeUnlessFresh",
        Op::MaterializeIfAbsent { .. } => "MaterializeIfAbsent",
        Op::MaterializeSegmentedOuterEq { .. } => "MaterializeSegmentedOuterEq",
        Op::MaterializePBuffers { .. } => "MaterializePBuffers",
        Op::MaterializeRA { .. } => "MaterializeRA",
        Op::MaterializeCombinedVal { .. } => "MaterializeCombinedVal",
        Op::Commit { .. } => "Commit",
        Op::CommitStreaming { .. } => "CommitStreaming",
        Op::Open => "Open",
        Op::ReduceOpenings => "ReduceOpenings",
        Op::InstanceReduce { .. } => "InstanceReduce",
        Op::InstanceSegmentedReduce { .. } => "InstanceSegmentedReduce",
        Op::InstanceBind { .. } => "InstanceBind",
        Op::InstanceBindPreviousPhase { .. } => "InstanceBindPreviousPhase",
        Op::Bind { .. } => "Bind",
        Op::Evaluate { .. } => "Evaluate",
        Op::EvaluatePreprocessed { .. } => "EvaluatePreprocessed",
        Op::BatchRoundBegin { .. } => "BatchRoundBegin",
        Op::BatchRoundFinalize { .. } => "BatchRoundFinalize",
        Op::BatchAccumulateInstance { .. } => "BatchAccumulateInstance",
        Op::BatchInactiveContribution { .. } => "BatchInactiveContribution",
        Op::ReadCheckingReduce { .. } => "ReadCheckingReduce",
        Op::RafReduce { .. } => "RafReduce",
        Op::SumcheckRound { .. } => "SumcheckRound",
        Op::LagrangeProject { .. } => "LagrangeProject",
        Op::DuplicateInterleave { .. } => "DuplicateInterleave",
        Op::RegroupConstraints { .. } => "RegroupConstraints",
        Op::BindCarryBuffers { .. } => "BindCarryBuffers",
        Op::QBufferScatter { .. } => "QBufferScatter",
        Op::SuffixScatter { .. } => "SuffixScatter",
        Op::ExpandingTableUpdate { .. } => "ExpandingTableUpdate",
        Op::InitExpandingTable { .. } => "InitExpandingTable",
        Op::WeightedSum { .. } => "WeightedSum",
        Op::CheckpointEvalBatch { .. } => "CheckpointEvalBatch",
        Op::UpdateInstanceWeights { .. } => "UpdateInstanceWeights",
        Op::InitInstanceWeights { .. } => "InitInstanceWeights",
        _ => return None,
    })
}

use jolt_compiler::PolynomialId;
use jolt_compute::{Buf, BufferProvider, ComputeBackend, Executable};

use helpers::PendingClaim;
use jolt_crypto::HomomorphicCommitment;
use jolt_field::Field;
use jolt_openings::{AdditivelyHomomorphic, ProverClaim};
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::proof::SumcheckProof;
use jolt_transcript::{AppendToTranscript, Transcript};
use jolt_verifier::proof::{JoltProof, StageProof};
use jolt_verifier::ProverConfig;

/// Per-stage proof being incrementally built.
pub(super) struct StageBuilder<F: Field> {
    pub(super) round_polys: Vec<UnivariatePoly<F>>,
    pub(super) evals: Vec<F>,
}

impl<F: Field> StageBuilder<F> {
    pub(super) fn new() -> Self {
        Self {
            round_polys: Vec::new(),
            evals: Vec::new(),
        }
    }

    pub(super) fn finalize(self) -> StageProof<F> {
        StageProof {
            round_polys: SumcheckProof {
                round_polynomials: self.round_polys,
            },
            evals: self.evals,
        }
    }
}

/// Mutable state accumulated during schedule execution.
pub(super) struct RuntimeState<F: Field, PCS: AdditivelyHomomorphic<Field = F>>
where
    PCS::Output: HomomorphicCommitment<F>,
{
    pub(super) config: ProverConfig,
    pub(super) challenges: Vec<F>,
    pub(super) evaluations: HashMap<PolynomialId, F>,
    pub(super) staged_evals: HashMap<(PolynomialId, usize), F>,
    pub(super) current_stage_idx: usize,
    pub(super) last_round_coeffs: Vec<F>,
    pub(super) last_round_poly: Option<UnivariatePoly<F>>,
    pub(super) last_squeezed: F,

    pub(super) batch_instance_claims: Vec<Vec<F>>,
    pub(super) last_round_instance_evals: Vec<Vec<F>>,
    pub(super) batch_combined: Vec<F>,
    pub(super) bound_this_round: HashSet<PolynomialId>,
    pub(super) current_batch_round: usize,
    pub(super) segmented_outer_eqs: HashMap<(usize, usize), Vec<F>>,

    pub(super) current_stage: Option<StageBuilder<F>>,
    pub(super) stage_proofs: Vec<StageProof<F>>,

    pub(super) commitments: Vec<Option<PCS::Output>>,
    pub(super) hints: HashMap<PolynomialId, PCS::OpeningHint>,
    pub(super) pending_claims: Vec<PendingClaim<F>>,
    pub(super) pending_hints: Vec<PCS::OpeningHint>,
    pub(super) reduced_claims: Vec<ProverClaim<F>>,
    pub(super) reduced_hints: Vec<PCS::OpeningHint>,
    pub(super) opening_proofs: Vec<PCS::Proof>,
    pub(super) padded_poly_data: HashMap<PolynomialId, Vec<F>>,

    /// Per-cycle eq-weight vector for address-decomposition instance.
    /// Initialized at phase 0, updated at each phase transition.
    pub(super) instance_weights: Vec<F>,

    /// Scalar checkpoints for address-decomposition instance (None = not yet initialized).
    /// Must preserve None vs Some(F::zero()) distinction because some checkpoints
    /// (e.g. Eq) use `unwrap_or(F::one())` where None means "use default".
    pub(super) instance_checkpoints: Vec<Option<F>>,

    /// Intermediate [eval_0, eval_2] from ReadCheckingReduce, consumed by RafReduce.
    pub(super) read_checking_evals: [F; 2],
}

/// Execute the full prover schedule and return a complete proof.
///
/// Walks every op in the schedule, dispatching compute ops to `backend`,
/// PCS ops to the commitment scheme, and orchestration ops directly.
pub(crate) fn execute<B, F, T, PCS>(
    executable: &Executable<B, F>,
    provider: &mut impl BufferProvider<F>,
    backend: &B,
    pcs_setup: &PCS::ProverSetup,
    transcript: &mut T,
    config: ProverConfig,
) -> JoltProof<F, PCS>
where
    B: ComputeBackend,
    F: Field,
    T: Transcript<Challenge = F>,
    PCS: AdditivelyHomomorphic<Field = F>,
    PCS::Output: AppendToTranscript + HomomorphicCommitment<F>,
{
    let _prove_span = tracing::info_span!("modular_prove").entered();
    let module = &executable.module;

    let batch_instance_claims: Vec<Vec<F>> = module
        .prover
        .batched_sumchecks
        .iter()
        .map(|b| vec![F::zero(); b.instances.len()])
        .collect();

    let mut state = RuntimeState::<F, PCS> {
        config,
        challenges: vec![F::zero(); module.challenges.len()],
        evaluations: HashMap::new(),
        staged_evals: HashMap::new(),
        current_stage_idx: 0,
        last_round_coeffs: Vec::new(),
        last_round_poly: None,
        last_squeezed: F::zero(),
        batch_instance_claims,
        last_round_instance_evals: Vec::new(),
        batch_combined: Vec::new(),
        bound_this_round: HashSet::new(),
        current_batch_round: 0,
        segmented_outer_eqs: HashMap::new(),
        current_stage: None,
        stage_proofs: Vec::new(),
        commitments: Vec::new(),
        hints: HashMap::new(),
        pending_claims: Vec::new(),
        pending_hints: Vec::new(),
        reduced_claims: Vec::new(),
        reduced_hints: Vec::new(),
        opening_proofs: Vec::new(),
        padded_poly_data: HashMap::new(),
        instance_weights: Vec::new(),
        instance_checkpoints: Vec::new(),
        read_checking_evals: [F::zero(); 2],
    };

    let stage_point_indices: Vec<Vec<usize>> = helpers::precompute_stage_points(module);

    let mut device_buffers: HashMap<PolynomialId, Buf<B, F>> = HashMap::new();

    // Stage-level tracing span. Dropped/replaced on every BeginStage so each
    // stage's ops appear nested under a "stage" span in Perfetto/chrome traces,
    // giving per-stage self-time breakdown without hand-tagging every op.
    let mut _stage_span: Option<tracing::span::EnteredSpan> = None;

    // Per-stage (wall, cpu) saturation tracking. `cpu - wall_at_entry` / wall
    // gives effective core count for the stage. Emitted at end of execute via
    // tracing::info! on target "perf_stage". Overhead: 2 getrusage calls per
    // stage (~500 ns each), negligible at ~8 stages / prove.
    let prove_wall_start = Instant::now();
    let prove_cpu_start = cpu_clock::process_cpu_time();
    let mut stage_start: Option<(usize, Instant, Duration)> = None;
    let mut stage_stats: Vec<(usize, Duration, Duration)> = Vec::new();

    // Per-op-class saturation: aggregate (wall, cpu) by op variant name, keyed
    // by a short &'static str. Only the expensive op classes are instrumented
    // (Materialize*, Commit, Open, InstanceReduce, InstanceSegmentedReduce,
    // ReduceOpenings, BatchRoundBegin, SumcheckRound, Bind, Evaluate).
    // Orchestration ops (Squeeze, Absorb*, BeginStage, RecordEvals, …) run in
    // microseconds and the getrusage overhead would be a large relative cost.
    let mut op_class_stats: HashMap<&'static str, (Duration, Duration, u64)> = HashMap::new();

    for op in &executable.ops {
        if let jolt_compiler::module::Op::BeginStage { index } = op {
            if let Some((prev_idx, wall_at_entry, cpu_at_entry)) = stage_start.take() {
                stage_stats.push((
                    prev_idx,
                    wall_at_entry.elapsed(),
                    cpu_clock::process_cpu_time().saturating_sub(cpu_at_entry),
                ));
            }
            stage_start = Some((*index, Instant::now(), cpu_clock::process_cpu_time()));
            _stage_span = None;
            _stage_span = Some(tracing::info_span!("stage", index = *index).entered());
        }

        let class = op_class_tag(op);
        let (wall, cpu) = if class.is_some() {
            (Some(Instant::now()), Some(cpu_clock::process_cpu_time()))
        } else {
            (None, None)
        };

        handlers::dispatch_op(
            op,
            &mut state,
            &mut device_buffers,
            executable,
            provider,
            backend,
            pcs_setup,
            transcript,
            &stage_point_indices,
        );

        if let (Some(tag), Some(w_start), Some(c_start)) = (class, wall, cpu) {
            let dw = w_start.elapsed();
            let dc = cpu_clock::process_cpu_time().saturating_sub(c_start);
            let e = op_class_stats
                .entry(tag)
                .or_insert((Duration::ZERO, Duration::ZERO, 0));
            e.0 += dw;
            e.1 += dc;
            e.2 += 1;
        }
    }

    if let Some((prev_idx, wall_at_entry, cpu_at_entry)) = stage_start.take() {
        stage_stats.push((
            prev_idx,
            wall_at_entry.elapsed(),
            cpu_clock::process_cpu_time().saturating_sub(cpu_at_entry),
        ));
    }

    if let Some(builder) = state.current_stage.take() {
        state.stage_proofs.push(builder.finalize());
    }

    let total_wall = prove_wall_start.elapsed();
    let total_cpu = cpu_clock::process_cpu_time().saturating_sub(prove_cpu_start);
    let ncpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let total_wall_ms = total_wall.as_secs_f64() * 1000.0;
    let total_cpu_ms = total_cpu.as_secs_f64() * 1000.0;
    let avg_threads = if total_wall_ms > 0.0 {
        total_cpu_ms / total_wall_ms
    } else {
        0.0
    };
    tracing::info!(
        target: "perf_stage",
        wall_ms = total_wall_ms,
        cpu_ms = total_cpu_ms,
        threads_avg = avg_threads,
        cores = ncpus,
        saturation_pct = (avg_threads / ncpus as f64) * 100.0,
        "modular_prove total"
    );
    for (idx, wall, cpu) in &stage_stats {
        let w = wall.as_secs_f64() * 1000.0;
        let c = cpu.as_secs_f64() * 1000.0;
        let r = if w > 0.0 { c / w } else { 0.0 };
        tracing::info!(
            target: "perf_stage",
            stage = idx,
            wall_ms = w,
            cpu_ms = c,
            threads_avg = r,
            saturation_pct = (r / ncpus as f64) * 100.0,
            "stage"
        );
    }

    let mut op_classes: Vec<_> = op_class_stats.iter().collect();
    op_classes.sort_by(|a, b| b.1 .0.cmp(&a.1 .0));
    for (tag, (wall, cpu, calls)) in op_classes {
        let w = wall.as_secs_f64() * 1000.0;
        let c = cpu.as_secs_f64() * 1000.0;
        let r = if w > 0.0 { c / w } else { 0.0 };
        tracing::info!(
            target: "perf_op",
            op = tag,
            wall_ms = w,
            cpu_ms = c,
            calls = calls,
            threads_avg = r,
            saturation_pct = (r / ncpus as f64) * 100.0,
            "op_class"
        );
    }

    JoltProof {
        config: state.config,
        stage_proofs: state.stage_proofs,
        opening_proofs: state.opening_proofs,
        commitments: state.commitments,
    }
}
