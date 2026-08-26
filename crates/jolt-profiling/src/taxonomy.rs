//! The versioned span taxonomy for the modular prover — **the normative
//! schema** for every span the pipeline emits ([`TAXONOMY_VERSION`] = 1).
//!
//! One instrumentation layer, two renderings: the same `tracing` span stream
//! becomes both the Perfetto-viewable chrome trace and the machine-queryable
//! `summary.json`. That makes span labels a de-facto public schema — queries
//! (`jq` over the summary, `trace_processor` SQL over the trace) and
//! `jolt-eval` telemetry objectives key on the exact strings below. Renaming
//! a span is a schema change: bump [`TAXONOMY_VERSION`], update the constants
//! here, and expect downstream `telemetry:<workload>:total:<label>` objective
//! keys to break loudly (an absent label is a measurement error, never 0.0).
//!
//! # Naming convention
//!
//! Identity in the name — a span is greppable to its emission site:
//!
//! - `Type::method` for leaf and kernel-seam spans
//!   (`EqPolynomial::evals`, `SpartanOuterUniskip::prepare`),
//! - `prove_stage{N}` for the stage recipes,
//! - `<StageLabel>::prove` / `<Relation>::prepare` / `<Relation>::prove_round`
//!   / `<Relation>::finish_rounds` for the generated stage drivers, where
//!   `<StageLabel>` is the batch struct's name minus the `Sumchecks` suffix
//!   and `<Relation>` is the member's relation type name (`finish_rounds` is
//!   the terminal bind delivery, a direct child of `prove_batch` rather than
//!   of a `sumcheck_round`),
//! - bare snake-case names (`prove_batch`, `stream_witnesses`) for
//!   free-function seams whose crate context is unambiguous.
//!
//! # Prover modes
//!
//! The `zk` compile-time feature swaps two protocol seams for committed
//! siblings: `prove_uniskip_clear` → `prove_uniskip_committed` and
//! `HomomorphicBatch::prove_batch` → `HomomorphicBatch::prove_batch_zk`.
//! Exactly one of each pair fires per prove — [`always_present_spans`] takes
//! the [`ProverMode`] and returns the matching presence set. The `akita`
//! feature swaps the commitment seams wholesale: the packed prover commits
//! one native `OneHotTrace` group in stage 0 (no `commit_witness` stream, no
//! homomorphic stage-8 batch) and discharges it with a native same-point
//! opening, with the reconstruction phase between stages 7 and 8 —
//! [`AKITA_MODE_SPANS`]. Every other label is mode-neutral (the sumcheck
//! engine's `prove_batch` differs only in its recorder, not its function).
//!
//! # Level policy
//!
//! Spans are emitted at `INFO` (the `#[tracing::instrument]` default).
//! Counter samples ride `DEBUG` events carrying `counters.*` fields (see
//! `MetricsMonitor`, `monitor` feature); the flush-time pipeline rewrites them
//! into chrome counter tracks. `RUST_LOG` filters only the console layer —
//! the chrome and summary layers always see everything.
//!
//! # Hot-loop rule
//!
//! Round granularity is the floor: per-round spans
//! (`sumcheck_round`, `<Relation>::prove_round` — ~log T instances per batch)
//! are fine, spans inside per-index inner loops are not. This is what keeps
//! the full subscriber stack inside the ≤5% overhead budget.
//!
//! # Required fields
//!
//! - [`ROOT_SPAN`] carries `trace_length`,
//! - `prove_batch` carries `num_rounds` and `members`,
//! - `sumcheck_round` carries `round`.
//!
//! Everything else is label-only; fields are additive (adding one is not a
//! schema change, removing or renaming one is).
//!
//! # Evolving the taxonomy
//!
//! 1. Change the instrumentation and the constants here in the same commit.
//! 2. Bump [`TAXONOMY_VERSION`] on any rename/removal of a label or field.
//! 3. The `jolt-prover` profiling smoke test asserts every
//!    [`always_present_spans`] label appears in a freshly emitted trace, so
//!    a silent rename fails the smoke test rather than drifting. The smoke
//!    test is deliberately not wired into CI yet (the reference backend
//!    exceeds hosted-runner memory — see the NOTE in
//!    `.github/workflows/rust.yml`); until that job lands it must be run
//!    explicitly after taxonomy changes.

/// Version of the span label set documented in this module.
pub const TAXONOMY_VERSION: u32 = 1;

/// The whole-run root span (`crates/jolt-prover/src/prover.rs`). Named
/// `jolt_prover::prove` rather than bare `prove`, which jolt-dory uses for an
/// inner opening-proof span. Carries `trace_length`; the dark-time and
/// peak-memory summary metrics are computed over its interval.
pub const ROOT_SPAN: &str = "jolt_prover::prove";

/// The per-stage recipe spans, in pipeline order — depth-1 children of
/// [`ROOT_SPAN`], sequential on the root thread. The per-stage summary
/// rollup (wallclock, boundary RSS, windowed peak memory) keys on these.
pub const STAGE_SPANS: [&str; 10] = [
    "prove_stage0",
    "prove_stage1",
    "prove_stage2",
    "prove_stage3",
    "prove_stage4",
    "prove_stage5",
    "prove_stage6a",
    "prove_stage6b",
    "prove_stage7",
    "prove_stage8",
];

/// The generated stage drivers' per-batch spans (`<StageLabel>::prove`,
/// emitted by `impl_stage_prover!`). Stages 0 and 8 have no sumcheck batch —
/// their work shows under the kernel-seam spans instead.
pub const DRIVER_BATCH_SPANS: [&str; 8] = [
    "Stage1Batch::prove",
    "Stage2Batch::prove",
    "Stage3::prove",
    "Stage4::prove",
    "Stage5::prove",
    "Stage6a::prove",
    "Stage6b::prove",
    "Stage7::prove",
];

/// Sumcheck engine spans (`jolt-sumcheck`): the batched round loop and its
/// per-round child. Mode-neutral — clear and ZK proves run the same
/// functions with different recorders.
pub const SUMCHECK_ENGINE_SPANS: [&str; 2] = ["prove_batch", "sumcheck_round"];

/// Kernel-seam spans (`jolt-kernels` trait boundaries) that fire on every
/// prove regardless of workload. Emitted by the stage recipes in
/// `jolt-prover` at the slot call boundaries — not by any backend impl — so
/// every backend genuinely inherits them by implementing the same traits.
/// The [`UNISKIP_SEAM_SPANS`] tail is mode-neutral; the head is the
/// homomorphic path's commit/opening seams (the packed path swaps them for
/// [`AKITA_MODE_SPANS`]).
pub const KERNEL_SEAM_SPANS: [&str; 6] = [
    "commit_witness",
    "SpartanOuterUniskip::prepare",
    "SpartanOuterUniskip::first_round_poly",
    "SpartanProductUniskip::prepare",
    "SpartanProductUniskip::first_round_poly",
    "JointOpeningPolynomials::prepare",
];

/// The mode-neutral uni-skip slot boundaries — [`KERNEL_SEAM_SPANS`] minus
/// the homomorphic commit/opening seams; the packed prover fires exactly
/// these.
pub const UNISKIP_SEAM_SPANS: [&str; 4] = [
    "SpartanOuterUniskip::prepare",
    "SpartanOuterUniskip::first_round_poly",
    "SpartanProductUniskip::prepare",
    "SpartanProductUniskip::first_round_poly",
];

/// Kernel-seam spans that fire only on proves whose guest consumes advice.
/// Call-boundary spans like [`KERNEL_SEAM_SPANS`]; exempt from the smoke
/// test's presence assertion (fibonacci has no advice).
pub const ADVICE_SEAM_SPANS: [&str; 2] = ["commit_advice", "AdviceOpeningEvaluation::evaluate"];

/// Kernel-seam spans that fire only with committed-program preprocessing.
pub const COMMITTED_PROGRAM_SEAM_SPANS: [&str; 1] = ["build_committed_bytecode_chunk_coeffs"];

/// Witness-plane seams (`jolt-witness`).
pub const WITNESS_AND_OPENING_SPANS: [&str; 3] = [
    "stream_witnesses",
    "collect_bundles",
    "TraceBackend::oracle_table",
];

/// Clear-mode protocol seams: the uni-skip first round and the transparent
/// stage-8 joint opening.
pub const CLEAR_MODE_SPANS: [&str; 2] = ["prove_uniskip_clear", "HomomorphicBatch::prove_batch"];

/// ZK-mode siblings of [`CLEAR_MODE_SPANS`]: the Pedersen-committed uni-skip
/// first round and the hiding stage-8 joint opening.
pub const ZK_MODE_SPANS: [&str; 2] = [
    "prove_uniskip_committed",
    "HomomorphicBatch::prove_batch_zk",
];

/// Packed-mode (`akita` feature) seams: the `OneHotTrace` column assembly
/// and native group commit at stage 0, the reconstruction phase at the head
/// of the stage-8 region (span always fires; its sumcheck batch — the
/// `Reconstruction::prove` driver span — runs only with advice or a
/// committed program), and the native same-point stage-8 opening. The packed
/// prover keeps `prove_uniskip_clear` (its recorders are clear; `akita` and
/// `zk` are mutually exclusive) and fires no `commit_witness` /
/// `stream_witnesses` / `JointOpeningPolynomials::prepare` /
/// `HomomorphicBatch::*`.
pub const AKITA_MODE_SPANS: [&str; 5] = [
    "assemble_one_hot_trace",
    "CommitmentScheme::commit_batch",
    "prove_reconstruction",
    "CommitmentScheme::open_batch",
    "prove_uniskip_clear",
];

/// Which compiled prover emitted a trace: the `zk` feature swaps the
/// [`CLEAR_MODE_SPANS`] seams for their [`ZK_MODE_SPANS`] siblings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProverMode {
    Clear,
    Zk,
    /// The packed (lattice) prover — transparent by construction.
    Akita,
}

/// Every v1 label that fires on all proves of the given mode: the presence
/// set the `jolt-prover` profiling smoke test asserts against a freshly
/// emitted trace.
///
/// Deliberately excludes the per-member driver spans (`<Relation>::prepare`,
/// `<Relation>::prove_round`, `<Relation>::finish_rounds`) whose names vary
/// with the batch composition, plus the advice- and committed-program-only
/// seams.
pub fn always_present_spans(mode: ProverMode) -> Vec<&'static str> {
    let mut labels = vec![ROOT_SPAN];
    labels.extend(STAGE_SPANS);
    labels.extend(DRIVER_BATCH_SPANS);
    labels.extend(SUMCHECK_ENGINE_SPANS);
    match mode {
        ProverMode::Clear | ProverMode::Zk => {
            labels.extend(KERNEL_SEAM_SPANS);
            labels.extend(WITNESS_AND_OPENING_SPANS);
        }
        // The packed prover streams no witness commit and runs no
        // homomorphic joint opening; its bundle collection and oracle reads
        // still fire (stage-0 assembly, the naive kernels).
        ProverMode::Akita => {
            labels.extend(UNISKIP_SEAM_SPANS);
            labels.extend(["collect_bundles", "TraceBackend::oracle_table"]);
        }
    }
    labels.extend(match mode {
        ProverMode::Clear => CLEAR_MODE_SPANS.as_slice(),
        ProverMode::Zk => ZK_MODE_SPANS.as_slice(),
        ProverMode::Akita => AKITA_MODE_SPANS.as_slice(),
    });
    labels
}
