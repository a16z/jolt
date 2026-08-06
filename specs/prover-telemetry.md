# Spec: Prover Telemetry

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | @moodlezoup                    |
| Created     | 2026-07-28                     |
| Status      | implemented                    |
| PR          | [#1712](https://github.com/a16z/jolt/pull/1712) |

## Summary

Until [PR #1712](https://github.com/a16z/jolt/pull/1712), the modular prover stack (`crates/jolt-prover`, `crates/jolt-kernels`, and the leaf crates they orchestrate) had zero performance instrumentation — not even a `tracing` dependency — so a Perfetto trace of a modular prove showed only disconnected leaf spans (Dory, `jolt-poly`) with no stage or kernel attribution above them. #1712 lands the human-readable half of the fix: per-stage, per-member, and per-round spans, a benchmark harness mirroring the legacy CLI, and peak-RSS reporting. The legacy prover's telemetry stack (ad-hoc `tracing` spans + `tracing-chrome` + manual Perfetto UI inspection + hand-written SQL + `MetricsMonitor` counters + allocative flamegraphs + an iai-callgrind bench) worked, but it accreted without a governing policy and every analysis step assumed a human in the loop. This spec codifies #1712's span conventions as the v1 of a principled, versioned taxonomy and specifies the rest of the telemetry pipeline, in which a single span stream renders both ways: a Perfetto-viewable trace for humans and a machine-queryable artifact for AI-driven workflows (autoresearch-style optimization via `jolt-eval`). It also extends the deterministic instruction-count lane (iai-callgrind microbenchmarks, today a single unregistered legacy bench) to the modular crates and registers it as `jolt-eval` objectives, so the optimizer's accept/reject decisions need not depend on wallclock noise.

## Intent

### Goal

Instrument the modular prover stack (`jolt-prover`, `jolt-kernels`, and the leaf crates they call) with a principled span-and-counter taxonomy, and extend `crates/jolt-profiling` into a telemetry pipeline where every profile run emits both a Perfetto-viewable trace and a machine-queryable artifact that AI agents and `jolt-eval` objectives consume without a human in the loop.

Key abstractions and boundaries:

- **One instrumentation layer, two renderings.** `tracing` spans are the single source of truth. The Perfetto trace and the machine-queryable summary are both derived from the same span stream at run time; there is no second bookkeeping channel. Consequence: the span taxonomy is a de-facto public schema.
- **Versioned span taxonomy**, documented in the `jolt-profiling` crate docs (`crates/jolt-profiling/src/taxonomy.rs` — the sole normative source, kept as doc comments rather than a README by implementation decision; the v1 label set is the convention shipped in PR #1712, pinned in Architecture): identity-in-the-name naming (`Type::method` for leaf and kernel-seam spans, `prove_stage{N}` for stage spans, `<StageLabel>::prove` / `<Relation>::prepare` / `<Relation>::prove_round` for driver spans), tracing-level policy, and the hot-loop rule (round-granularity spans are fine; none inside per-index inner loops).
- **Artifacts.** Each profile run writes into its own directory, `benchmark-runs/{timestamp}_{trace_name}/`, holding every artifact the run produces — `trace.json` (chrome trace: Perfetto UI for humans, `trace_processor` SQL for machines), `summary.json` (thin, schema-versioned aggregates), `memory.html`, the `{StageLabel}_prepared.folded` heap snapshots, and `timings.csv` (the directory name carries the run identity, so the files inside use fixed names) — and `benchmark-runs/latest_{trace_name}` is symlinked to the newest successful run. `{trace_name}` = `modular_{workload}_{scale}` (hyphens mapped to underscores, `{scale}` the decimal log2 trace length); the `modular_` prefix disambiguates side-by-side legacy artifacts and is dropped when the legacy prover is deleted. Consumers that know the workload and scale read through the `latest_` link — deterministic paths with run history preserved (`jolt-eval` chooses both components — see Architecture). (Implementation note, evolution: originally a flat `benchmark-runs/perfetto_traces/` layout overwriting in place; regrouped by run once multiple artifacts per run existed.)
- **Profile entry point**: a feature-gated `[[bin]] name = "jolt-prover"` in `crates/jolt-prover` (`required-features = ["profiling"]`) with a `profile` subcommand and a backend selector defaulting to `reference`; it subsumes and retires #1712's `examples/modular_benchmark.rs`.
- **`jolt-eval` telemetry objectives**: a new, third measurement channel whose objectives are string-keyed and parameterized (`telemetry:<workload>:<metric>`; grammar pinned in Performance), so an optimization agent can target any span it discovers in a trace without editing `jolt-eval`; a small curated set ships as named defaults.
- **Deterministic microbenchmark lane**: iai-callgrind benchmarks over selected `jolt-kernels`/`jolt-poly` hot paths, registered as `jolt-eval` objectives (`callgrind:<bench-name>:instructions`); opt-in (requires Valgrind), microbenchmark-scale only.

### Invariants

No existing `jolt-eval` invariants are modified and no new ones are added; the eight registered invariants (`split_eq_bind` ×2, `field_mul_scalar`, `soundness`, `transcript_prover_verifier_consistency` ×3, `source_to_jolt_expansion_equivalence`) check prover/field/transcript semantics that telemetry must not perturb, and they continue to hold unchanged.

The correctness properties of the telemetry system itself are enforced by ordinary tests rather than `jolt-eval` invariants (decision made during spec review):

- **Observational purity** — instrumentation must not change prover output. Guarded by `jolt-prover`'s byte-diff harness (run explicitly via `cargo nextest run -p jolt-prover --features prover-fixtures`; the `muldiv` module is stage-granular, the remaining modules whole-proof, all byte-comparing against a live legacy prove), which exercises the instrumented code paths. #1712 already ran this gate green over the span instrumentation.
- **Trace/summary consistency** — `summary.json` numbers equal a deterministic aggregation of the trace's span events (both derive from one span stream by construction). Guarded by unit tests over fixture traces.
- **Taxonomy conformance** — spans emitted by instrumented crates match the documented naming convention and carry required fields. Guarded two ways: unit tests over fixture traces, and the e2e smoke test asserting that every v1 label that fires on all proves (root span, stage, driver-phase, and always-present kernel-seam labels) is present in a freshly emitted trace, so a silent rename fails the smoke test rather than drifting. (The smoke test is deliberately not CI-wired yet — see Testing Strategy — so until that job lands it must be run explicitly after taxonomy changes.)

### Non-Goals

- **`perf-event` / hardware-counter integration** (per-span PMU counters). Future work; Linux-only and real integration effort. The deterministic-measurement need is covered at microbenchmark scale by the iai-callgrind lane.
- **iai-callgrind on end-to-end proves.** Callgrind's ~20–100× slowdown (per the Valgrind manual) limits it to microbenchmarks; e2e measurement stays wallclock.
- **Retrofitting `jolt-prover-legacy`.** The legacy prover keeps its existing instrumentation, its `benches/iai.rs` callgrind bench, and its scripts (`postprocess_trace.py` et al.) until it is deleted. (The small additive hooks #1712 gave the legacy bin — `StageMemoryLayer` and the peak-RSS printout, for apples-to-apples comparison — are fine and stay.)
- **CI-side longitudinal tracking.** No changes to `ci-bench.yml`, `bench-crates.yml`, or the gh-pages dashboard; this spec ships local and `jolt-eval` tooling only. (A `rust.yml` change to run the smoke test's feature-gated lane is in scope — see Testing Strategy.)
- **Replacing the pprof lane.** pprof remains a separate, feature-gated tool as today.
- **Verifier instrumentation.** Prover-side only.
- **Native Perfetto protobuf output.** Chrome JSON now; see Alternatives Considered.

In scope, for the avoidance of doubt: heap/memory telemetry — both the cheap lane (CPU/memory counter tracks riding the same event stream via `MetricsMonitor`) and the deep lane (allocative heap flamegraphs for the new prover's types, behind an `allocative` feature).

## Evaluation

### Acceptance Criteria

- [ ] **Entry point**: a single documented command (`cargo run --release -p jolt-prover --features profiling -- profile --name <workload> --format chrome`) profiles the modular prover on a named workload and emits both artifacts from one run. Workloads (the four scalable ones #1712's harness supports): fibonacci, sha2-chain, sha3-chain, btreemap, with default scales pinned here — fibonacci 2^16, sha2-chain 2^22, sha3-chain 2^22, btreemap 2^20 — overridable via `--scale <log2 trace length>`. The bin subsumes `examples/modular_benchmark.rs` (same guest-input construction and verify gate; the example is retired when the bin lands). The `profiling` feature enables `jolt-profiling/monitor`, so this one command emits counter data too.
- [ ] **Attribution coverage**: in the resulting trace, every stage (`prove_stage0`–`prove_stage8`, incl. 6a/6b), every driver span (`<StageLabel>::prove`, per-member `<Relation>::prepare`, `prove_batch` with per-round `<Relation>::prove_round`), and every bespoke kernel-slot invocation (`commit_witness`, both uni-skip slots, advice opening, `JointOpeningPolynomials::prepare`) is present — largely landed in #1712. Dark time — defined in Architecture; reported in `summary.json` — is ≤ 5% on sha2-chain at scale 2^22.
- [ ] **Canonical queries**: documented one-liners (no Perfetto UI) answer: total time across all instances of a span label; top-N spans by inclusive time and by self time; per-stage wallclock breakdown; per-stage memory (boundary-RSS deltas and windowed sample-max). All four are answerable from `summary.json` alone (e.g. via `jq`); arbitrary ad-hoc queries go through `trace_processor` SQL against the trace (provisioned via the pinned `perfetto` pip package documented in the book page).
- [ ] **Native counter tracks**: CPU/memory counters render as Perfetto counter tracks directly from the emitted trace — no offline post-processing step (no `postprocess_trace.py` equivalent in the workflow).
- [ ] **`jolt-eval` integration**: `cargo run -p jolt-eval --bin measure-objectives -- --objective telemetry:fibonacci:prover_time_s` prints a measurement when executed in a fresh worktree, and at least one curated `ObjectiveFunction` wrapping a telemetry objective appears in `optimize --list`.
- [ ] **Callgrind lane**: at least one iai-callgrind microbenchmark over a `jolt-kernels` or `jolt-poly` hot path, with its instruction count measurable as `callgrind:<bench-name>:instructions` via `measure-objectives` (opt-in; skipped with a clear error when Valgrind is absent).
- [ ] **Overhead budget**: `prove()` wallclock with the full subscriber stack attached (`--format chrome`: chrome layer + summary layer + monitor) is ≤ 105% of a no-subscriber run (`--format none`, which installs no subscriber and times `prove()` with `std::time::Instant`), measured on sha2-chain at scale 2^22, median of 3 runs on the same machine, comparing the root span's duration against the `Instant` measurement. Guest compilation, tracer execution, and preprocessing are excluded from the measured interval.
- [ ] **Allocative lane**: with `--features profiling,allocative`, the same profile command additionally captures per-batch heap snapshots — one per driver batch, taken right after `prepare_members` with every member kernel live and its tables materialized, covering the member kernels and the `ProofSession` contents — persisted as exact-bytes folded-stacks files (`{run_dir}/{StageLabel}_prepared.folded`), rolled into `summary.json`'s `heap` section (a schema field, empty when the lane is off), and rendered by the self-contained `memory.html` timeline page (RSS envelope + stage bands + composition columns + click-to-icicle detail). Verified by the existence check in the Execution phase-6 procedure. (Implementation note, evolution: inferno SVG emission was dropped once the timeline page landed — the SVGs' integer-MiB hover rounding and per-file normalization made them strictly worse than the page + folded pair.) (Implementation note, deviation: the spec originally asked for end-of-stage `stage{N}.svg` snapshots of the clear-output carriers; those proved near-empty by construction — stage working sets free on exit, and cross-boundary carries surface inside the consuming kernel in the next batch's `_prepared` snapshot — so the boundary snapshots were replaced by the mid-stage ones, which is where the reference backend's multi-GiB tables actually show.)

### Testing Strategy

Must keep passing, unchanged:

- The full `cargo nextest` workspace suite, plus `jolt-prover`'s feature-gated suites run explicitly: the byte-diff harness (`--features prover-fixtures`, which also guards observational purity — instrumented code still byte-matches a live legacy prove) and `engine_twins`.
- The `muldiv` e2e in both `--features host` and `--features host,zk` (legacy crate; workspace health check since instrumentation touches shared leaf crates).
- `cargo clippy` clean in both feature modes.

ZK mode needs no telemetry-specific testing — spans are mode-independent — but both-mode clippy/test runs stay mandatory because instrumentation touches shared code.

New tests:

- **Unit tests for the aggregation/query layer** against small fixture traces checked into the repo (deterministic, fast, no prover run). These cover trace/summary consistency and taxonomy conformance, validating against the checked-in JSON Schema for `summary.json` (see Architecture).
- **One e2e smoke test**: a `#[test]` in `jolt-prover` gated behind the `profiling` feature, calling the profile entry point as a library function (the bin's `main` is a thin wrapper) on fibonacci at scale 2^13 (implementation note: the spec originally said 2^16, but the reference backend retains ~18 GiB in stage 2 regardless of scale — ram_K is priced off the guest's default 32 MB heap — and peaks near 80 GiB at 2^16; 2^13 is fibonacci's minimum guest scale). Asserts: both artifacts exist and parse; `summary.json` validates against the checked-in schema; every v1 label that fires on all proves — the root span, all `prove_stage*` and driver labels (`<StageLabel>::prove`, `prove_batch`), and the kernel-seam labels `commit_witness`, `SpartanOuterUniskip::prepare`, `SpartanProductUniskip::prepare`, `JointOpeningPolynomials::prepare` — appears in the emitted trace (`commit_advice` and `AdviceOpeningEvaluation::evaluate` are exempt: fibonacci exercises no advice). CI: `jolt-prover` is currently tested by `rust.yml`'s sharded per-crate discovery loop, which enables no extra features and installs no guest toolchain, so enabling this test requires a special-cased step or dedicated job running `cargo nextest run -p jolt-prover --features profiling` with the ZeroOS musl toolchain and jolt CLI install steps the guest compile path requires (mirroring the legacy test jobs). Without this the feature-gated test would silently never run. (Implementation note: the CI job is deliberately deferred — even at minimum scale the reference backend's ~18 GiB retained footprint exceeds hosted-runner memory; the test ships in-tree, run explicitly, and the job gets wired up once an optimized backend fits runner memory. A pointer comment sits in rust.yml next to the legacy test jobs.)
- **The 5% dark-time and 5% overhead budgets** are verified by a documented manual/bench procedure, not CI-gated (wallclock thresholds on shared CI runners are flaky gates).
- **iai-callgrind lane is opt-in**: documented command, no CI job, no Valgrind requirement imposed on contributors.

### Performance

**Existing objectives moved: none** (expected movement ~0).

- The static-analysis objectives (`lloc`, `cognitive_complexity_avg`, `halstead_bugs`) measure `crates/jolt-prover-legacy/src/`, which this feature does not touch.
- The Criterion microbenches install no tracing subscriber, so added instrumentation costs a branch-on-disabled per span callsite — unmeasurable. The ≤5% overhead budget applies only when a subscriber is attached.

**New objectives** (added via `/new-objective` during implementation). Implementation-wise these are exactly **two new objective types** — one parameterized `TelemetryObjective` and one `CallgrindObjective` — everything else is a key or a curated name over them:

1. **Parameterized telemetry family** — key grammar, pinned:

   ```
   telemetry:<workload>:<metric>
   <workload> ::= [a-z0-9-]+            (case-sensitive; must be in the workload table)
   <metric>   ::= prover_time_s         (root-span duration, seconds)
                | peak_rss_gib          (process-lifetime getrusage high-water mark)
                | peak_memory_gib       (max over counters.memory_gib samples in the root span)
                | total:<span-label>    (inclusive time summed over all instances, seconds)
                | self:<span-label>     (exclusive time summed over all instances, seconds)
                | heap:<snapshot>       (allocative mid-stage snapshot total, exact bytes)
                | heap:<snapshot>:<root> (one root frame's bytes; root is verbatim)
   ```

   Parsing: split on the first three `:` only — everything after the third colon is the **verbatim span label and may itself contain `:`** (e.g. `telemetry:sha2-chain:total:EqPolynomial::evals`). Measurement runs the workload at its default scale from the workload table. A key referencing a label absent from `summary.json` is a **measurement error, never 0.0** (silent zeros would corrupt optimizer accept/reject decisions). All time metrics are reported in seconds (converted from the summary's ns), matching existing objectives; `heap:` metrics report exact bytes and build the profile subprocess with the `allocative` feature (the optimizer's shared per-workload run enables it when any sharer needs it). (Implementation note, extension: the `heap:` family was added post-implementation once the allocative lane's snapshots landed in `summary.json` — snapshot labels are the flamegraph names, root frames the kernel type names.)

   Curated defaults ship as named `ObjectiveFunction`s (modular-prover e2e time, per-stage totals, commit time, per-stage `prove_batch` round-loop time) so `optimize --list` stays useful.

2. **iai-callgrind instruction counts** — `callgrind:<bench-name>:instructions`, parsed from iai-callgrind's machine-readable JSON output (`--output-format=json`; the `Ir` event kind — the instruction count, rendered as "Instructions" only in console output); deterministic objectives for selected `jolt-kernels`/`jolt-poly` microbenches; the noise-free signal for optimizer accept/reject decisions.

Note on the string-keyed variant: `OptimizationObjective` is `Copy + Eq + Hash` and curated `ObjectiveFunction`s store `&'static [OptimizationObjective]`. Runtime-parsed telemetry/callgrind keys preserve this by interning: parsed keys are leaked once (`Box::leak`) into `&'static str`, so the new variants stay `Copy` and const-compatible.

**Explicitly waived** (decision made during spec review): query-latency / artifact-size budgets for the tooling itself, and optimizer-loop turnaround budgets. Implicitly bounded by choosing small workloads for optimization runs.

## Design

### Architecture

```
jolt-prover / jolt-kernels / leaf crates        library crates depend on `tracing` only
        │
        │  one span + event stream
        ▼
crates/jolt-profiling subscriber stack          the only crate hosting subscribers
        ├── fmt layer (console, RUST_LOG)
        ├── tracing-chrome layer ──────────────► {run_dir}/trace.json ──┬─► Perfetto UI (human)
        │     └── integrated flush-time counter rewrite                  └─► trace_processor SQL (agent, ad-hoc)
        ├── summary aggregation layer ────────────────► {run_dir}/summary.json ──► jolt-eval TelemetryObjective
        └── MetricsMonitor thread (feature "monitor")                                    (measure-objectives / optimize)
```

**Instrumentation (the span taxonomy, v1 label set — the conventions shipped in PR #1712):**

- `crates/jolt-prover` (landed in #1712): root span on `prove()` (`src/prover.rs`, field: `trace_length`) — **renamed from #1712's bare `prove` to `jolt_prover::prove`**, since the bare name collides with a jolt-dory inner span (#1712's own `StageMemoryLayer` special-cases around the collision); per-stage spans `prove_stage0`…`prove_stage8` incl. `prove_stage6a`/`prove_stage6b`; driver spans emitted by the `impl_stage_prover!` macro (`src/driver.rs`) — `<StageLabel>::prove` per batch (e.g. `Stage2Batch::prove`), `<Relation>::prepare` per member, and per-round `<Relation>::prove_round` via the instrumentation-only `SpannedRounds` shim wrapped around each kernel.
- `crates/jolt-kernels` (landed in #1712): spans at the trait seams — `commit_witness`, `commit_advice`, `SpartanOuterUniskip::prepare`/`::first_round_poly`, `SpartanProductUniskip::prepare`/`::first_round_poly`, `AdviceOpeningEvaluation::evaluate`, `JointOpeningPolynomials::prepare`, `build_committed_bytecode_chunk_coeffs` — so any future optimized backend inherits attribution for free by implementing the same traits.
- Supporting crates (landed in #1712): `jolt-sumcheck` `prove_batch` (plus per-round `sumcheck_round`, ~log T per batch) and the mode-selected uni-skip round (`prove_uniskip_clear`, or `prove_uniskip_committed` under the `zk` feature); `jolt-witness` `stream_witnesses`, `collect_bundles`, `TraceBackend::oracle_table`; `jolt-openings` `HomomorphicBatch::prove_batch` (or `HomomorphicBatch::prove_batch_zk` under `zk`). Round-granularity spans are fine; none inside per-index inner loops (the overhead-budget discipline).
- Leaf crates (`jolt-poly`, `jolt-dory`, …) keep and extend their existing `#[tracing::instrument(skip_all, name = "Type::method")]` spans, brought under the codified convention.
- Library crates gain only the workspace `tracing` dependency; no subscriber code in library crates.
- The `jolt-profiling` taxonomy module (`crates/jolt-profiling/src/taxonomy.rs`, Execution phase 1) is the sole normative source for the taxonomy; the label set above is its v1 content and the acceptance criteria are evaluated against it.

**Pipeline (`crates/jolt-profiling`, extended):**

- Remains the only crate hosting subscribers. Gains: (a) an in-process **summary aggregation layer** that accumulates per-label/per-stage aggregates and writes the run's `summary.json` at flush; (b) **integrated counter conversion** — `tracing-chrome` cannot emit chrome counter events (`"ph": "C"`), so the pipeline rewrites `counters.*`-prefixed events into counter events at flush time, inside the same command (this is what kills the offline `postprocess_trace.py` step); (c) counter-name unification (legacy emits `counters.memory_gb`, `jolt-profiling` emits `counters.memory_gib` — the new pipeline standardizes on `memory_gib`).
- MetricsMonitor, pprof, and allocative stay behind their existing feature flags (`monitor`, `pprof`, `allocative`).
- **`summary.json` schema**: the serde structs in `crates/jolt-profiling` are the normative definition, mirrored by a checked-in JSON Schema that the smoke test and fixture tests validate against, versioned via a `schema_version` field. Contents: run metadata (workload, scale, backend, timestamp, git rev), per-span-label aggregates (instance count, total/inclusive ns, self/exclusive ns), per-stage rollup (wallclock, boundary-RSS open/close/Δ, and windowed peak memory per stage), dark-time fraction at root, headline peak RSS and windowed peak memory, counter summaries.
- **Aggregation semantics under rayon parallelism** (load-bearing for the 5% gate and the objective family):
  - *Dark time* = the root span's wallclock minus the **union of the intervals** of its depth-1 child spans, computed on the root thread's timeline (the pipeline is sequential at stage granularity, so depth-1 children do not overlap in practice; union semantics make the definition robust if that changes).
  - *Self time* of a span = its inclusive duration minus the union of the intervals of its **same-thread** children. Spans opened on other rayon worker threads attribute to their own labels but never subtract from a parent on a different thread.
  - *Per-label totals* sum inclusive durations across all instances on all threads and may legitimately exceed wallclock under parallelism; objectives consume them as-is.
  - *Per-stage peak memory* = max over `counters.memory_gib` samples that fall within the stage span's interval; **nullable** in the schema, since short stages may contain zero samples at the monitor's minimum 50 ms sampling interval (certain at smoke-test scale).
  - *Per-stage boundary RSS* — the open/close/Δ rows recorded by #1712's `StageMemoryLayer` (retained growth per stage; deliberately not within-stage peak) — is copied into the per-stage rollup alongside the sample-max.
  - *Peak memory overall* (the `peak_memory_gib` metric) = max over samples within the **root span's** interval — consistent with the time metrics' exclusion of guest compile, tracer execution, and preprocessing.
  - *Headline peak RSS* (the `peak_rss_gib` metric) = `jolt_profiling::peak_rss_bytes()` from #1712 — the `getrusage` high-water mark. Process-lifetime (includes guest compile/trace), but it cannot miss short allocation spikes, which sampling can; the two peak metrics are complementary, not redundant.

**Entry point:**

- `[[bin]] name = "jolt-prover"` in `crates/jolt-prover`, `required-features = ["profiling"]`; `main` is a thin wrapper over a library entry point (so the smoke test can call it in-process). Promotes #1712's `examples/modular_benchmark.rs` — keeping its guest-input construction (cycles-per-op constants, 90%-of-2^scale target), verify-as-correctness-gate, and CSV output — after which the example is retired. The `profiling` feature pulls `jolt-profiling` (with `jolt-profiling/monitor`), clap, and the guest compile/trace path (legacy `host` machinery, as the existing `prover-fixtures` feature already does).
- `profile` subcommand: `--name <workload>` (workload table with per-workload default scales), `--scale <log2 trace length>` (override), `--format <default|chrome|none>`, `--backend <reference>`.
  - `default`: console fmt layer only (span-close timings), no artifacts.
  - `chrome`: full stack; emits both artifacts (the format `jolt-eval` invokes).
  - `none`: no subscriber; times `prove()` with `std::time::Instant` and prints it — the overhead-budget baseline.
  - `--backend`: `reference` is the only backend today and is a test oracle, so absolute numbers are provisional — attribution is meaningful relatively, and optimized backends slot into the same instrumented seams.

**`jolt-eval` integration:**

- The `OptimizationObjective` union gains two variants alongside the existing `StaticAnalysis` and `Performance` ones: `Telemetry(TelemetryObjective)` and `Callgrind(CallgrindObjective)` — string-keyed per the grammars in Performance, parsed at runtime (interned to preserve `Copy`; see Performance).
- `jolt-eval` owns the normative measurement-scale table: `TelemetryObjective` measurement always invokes the profile bin with an explicit `--scale` from that table (initialized to this spec's workload-table defaults), spawning it as a subprocess with `current_dir` = the target work dir (worktree-safe — same rationale as the subprocess-in-worktree comment on `check_invariants` in `bin/optimize.rs`, whose pattern `RealEnv::measure` already follows for Criterion benches), then parses `{work_dir}/benchmark-runs/latest_{trace_name}/summary.json` (trace_name = `modular_{workload}_{scale}`, hyphens mapped to underscores) — deterministic because it chose both components and because the harness flips the `latest_` link only after a run completes (jolt-eval removes the link before spawning, which is the stale-candidate protection). Wired into `RealEnv::measure` (`bin/optimize.rs`) and `bin/measure_objectives.rs`.
- The iai-callgrind lane registers `CallgrindObjective`s that parse iai-callgrind's JSON output; bench targets live in a `jolt-eval/benches/callgrind/` subdirectory. `sync_targets.sh` currently deletes **all** `[[bench]]` blocks from `jolt-eval/Cargo.toml` and reinserts only those scanned from top-level `benches/*.rs`, so it must learn to preserve (or regenerate, with explicit `path = "benches/callgrind/<name>.rs"` and `harness = false`) the callgrind entries during that pass.

### Alternatives Considered

- **Separate metrics channel** (schema'd measurements recorded at the driver seam, independent of `tracing`, with Perfetto as one renderer). Rejected: one instrumentation layer with two renderings guarantees the human and the machine see the same numbers; a second channel invites drift. The cost — span renames break queries — is accepted deliberately: it makes the taxonomy a versioned schema rather than an aesthetic.
- **Custom `tracing` layer emitting SQLite** (+ chrome JSON). Rejected: full control and native counters, but we would own a subscriber that must get threading and flush right, replacing battle-tested `tracing-chrome`.
- **Trace-JSON-parsing query CLI as the only query engine.** Rejected: least machinery, but re-implements aggregation that Perfetto's `trace_processor` already does better, and grows into a bespoke query language over time.
- **Chosen: hybrid** — `trace_processor` SQL as the arbitrary-query engine (the same SQL already used manually in the Perfetto UI, now scriptable) plus a thin flush-time `summary.json` for the stable-schema consumers (`jolt-eval`, quick agent queries).
- **Curated-only objective enum.** Rejected for a parameterized, string-keyed objective family: autoresearch agents can target any span they discover without first editing `jolt-eval`. Curated defaults are kept for discoverability.
- **In-pipeline hardware counters** (`perf-event` groups attached per span). Deferred to future work: Linux-only, real integration effort; the deterministic-measurement need is served at microbenchmark scale by iai-callgrind.
- **Entry point in `jolt-profiling` or `jolt-eval`.** Rejected: the former inverts `jolt-profiling`'s role as a dependency-light subscriber host; the latter couples a standalone dev tool to the eval harness.
- **`tracing-perfetto` (native protobuf traces).** Considered and deferred. Native `.pftrace` would buy much smaller files, faster UI loads, and first-class counter tracks — but the crates are young and thinly maintained (some pull the Perfetto C++ SDK via FFI), it does not move machine-queryability at all (`trace_processor` ingests chrome JSON and protobuf identically; `summary.json` is trace-encoding-independent), and the decision is cheaply reversible: only `crates/jolt-profiling` knows the trace encoding, so a later swap is a one-crate change that invalidates nothing in the taxonomy, summary schema, or `jolt-eval` integration.

## Documentation

- **Rewrite `book/src/usage/profiling/zkvm_profiling.md`** around the modular prover as the primary workflow: the profile command, Perfetto UI, `summary.json` + `jq` recipes, `trace_processor` SQL recipes for common questions (including the pinned `perfetto` pip package provisioning), memory/allocative lanes. Legacy-prover instructions move to a clearly-marked legacy section until that crate is deleted.
- **Span taxonomy schema** lives in the `jolt-profiling` crate docs (`crates/jolt-profiling/src/taxonomy.rs`, doc comments + machine-usable label constants), versioned: naming convention, the v1 pipeline label set, required fields, level policy, hot-loop rule, and the process for evolving it.
- **Update `jolt-eval/README.md`**: the telemetry objective category (key grammar as pinned in this spec, curated defaults) and the iai-callgrind lane (opt-in, Valgrind requirement, how to run).
- **Update `CLAUDE.md`** profiling commands so agents discover the new workflow (profile command, canonical summary queries, trace_processor usage).

## Execution

Suggested phase ordering:

1. **Instrument** — largely landed in PR #1712 (spans across `jolt-prover`, `jolt-kernels`, `jolt-sumcheck`, `jolt-witness`, `jolt-openings`; the `SpannedRounds` shim; byte-diff suite green; overhead measured at noise level). Remaining delta: rename the root span from bare `prove` to `jolt_prover::prove` (the bare name collides with a jolt-dory inner span, which forces `StageMemoryLayer` to special-case around it), and write the taxonomy crate docs (`taxonomy.rs`) documenting the shipped v1 label set as the normative schema.
2. **Pipeline** — extend `jolt-profiling`: summary aggregation layer (serde structs + checked-in JSON Schema); integrated flush-time counter conversion (note: `tracing-chrome` cannot emit `"ph": "C"` events itself); unify counter names on `memory_gib`; fold the `StageMemoryLayer` boundary rows and `peak_rss_bytes()` headline (both landed in #1712) into `summary.json`. Legacy scripts: `postprocess_trace.py` is left untouched — it serves the legacy prover only. (Implementation note, deviation: with `jolt_benchmarks.sh` retired in favor of the bin's `benchmark` sweep, `plot_memory_usage.py` was repointed at the modular `summary.json` artifacts — its legacy-trace parsing left with its only caller — and `benchmark_summary.py`/`plot_benchmarks.py` defaults now read `modular_timings.csv`.)
3. **Entry point** — `profiling` feature + `jolt-prover` bin, promoting #1712's `examples/modular_benchmark.rs` (keep its guest-input construction, verify gate, and CSV output; retire the example); add the default-scale table pinned in the Acceptance Criteria, the `--format none` baseline mode, and the backend selector defaulting to `reference`.
4. **`jolt-eval`** — `TelemetryObjective` channel; interned string-keyed variants on `OptimizationObjective`; wire into `RealEnv::measure` and `measure-objectives`. Landmines: measure via subprocess in the worktree, never in-process (stale-binary trap — see the comment on `check_invariants` in `bin/optimize.rs`); a shared `CARGO_TARGET_DIR` breaks `{work_dir}/target`-relative assumptions — the summary artifact path is cwd-relative by design, but resolve any target-dir-relative paths explicitly.
5. **Callgrind lane** — iai-callgrind bench targets under `jolt-eval/benches/callgrind/` for selected hot paths (candidates: `jolt-poly` binding/eq-table evals, `NaiveSumcheckProver::prove_round`); `CallgrindObjective` parses the JSON output (`Ir` event kind); `sync_targets.sh` must preserve or regenerate the callgrind `[[bench]]` entries (explicit `path`, `harness = false`) during its delete-and-reinsert pass; opt-in, no CI. The legacy `crates/jolt-prover-legacy/benches/iai.rs` bench is left untouched (legacy-retrofit non-goal).
6. **Validate + document** — run the documented budget procedure (dark time ≤ 5%, overhead ≤ 105%, sha2-chain at scale 2^22, median of 3); check the allocative .folded snapshots and memory.html exist per the acceptance criterion; book/README/CLAUDE.md updates.

## References

- [PR #1712](https://github.com/a16z/jolt/pull/1712) — per-stage/per-member/per-round Perfetto coverage, the `modular_benchmark` harness, and peak-RSS reporting; the landed foundation whose conventions this spec codifies as taxonomy v1
- [`perf-event`](https://crates.io/crates/perf-event) — hardware-counter crate considered for future work
- [`iai-callgrind`](https://lib.rs/crates/iai-callgrind) — deterministic instruction-count benchmarking (already a workspace dependency; legacy bench at `crates/jolt-prover-legacy/benches/iai.rs`)
- [Perfetto `trace_processor`](https://perfetto.dev/docs/analysis/trace-processor) — SQL query engine over traces
- [`tracing-chrome`](https://crates.io/crates/tracing-chrome) — chrome-format trace layer (current)
- `specs/akita-perf-plan.md` — the "TRACE FIRST" optimization loop this spec mechanizes
- `specs/prover-stage-drivers.md` — the generated stage-driver seam the instrumentation hooks into
- `jolt-eval/README.md` — the invariant/objective framework the telemetry objectives plug into
- `book/src/usage/profiling/zkvm_profiling.md` — current (legacy) profiling documentation
- `scripts/postprocess_trace.py` — the offline counter-conversion step this spec obsoletes for the new prover
