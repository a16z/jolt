# Spec: `jolt-prover` Frontier Harness

| Field | Value |
|-------|-------|
| Author(s) | Markos Georghiades, Codex |
| Created | 2026-05-26 |
| Status | draft |
| PR | TBD |

## Summary

This spec defines the correctness and performance harness required to finish
the modular `jolt-prover` migration without losing CPU fast-path performance.
The harness is intentionally built around replayable fixtures and backend CPU
optimization ports, not around slow full-flow correctness experiments.

The goal is a two-phase additive proof migration:

```text
phase 1: backend kernel inventory completion
  -> account for every jolt-core prover optimization ID that targets cpu-backend
  -> port the real optimized core algorithms into jolt-backends::cpu
  -> add focused microbenchmarks and analytical memory budgets
  -> certify backend results against replay fixtures and core measurements

phase 2: prover frontier migration
  -> one verifier-visible prover frontier at a time
  -> jolt-prover drives Fiat-Shamir and dispatches optimized backend requests
  -> pass jolt-verifier correctness using native jolt-verifier types and flows
  -> pass core performance parity on every relevant measured axis
  -> promote the frontier so later stages use modular outputs instead of fixtures
  -> next frontier
```

The harness is not production proving code. It exists to make the migration
fast to iterate: fixtures let us replay a narrow slice without regenerating a
large proof, and performance gates ensure optimized backend kernels are ported
before prover frontiers are accepted. Correctness-only implementations are not
acceptable migration output.

No frontier may be considered done by relying on fixture-only proof fragments,
ad hoc proof splicing, temporary comparison scaffolding, or generic fallback
code when the core algorithm has an optimized CPU path. Those artifacts may
help inspect state during development, but they are not acceptance rails.

## Goal-Mode Algorithm

The next prover goal mode should follow this order:

1. Build and certify the backend kernel inventory first.
2. For every optimization in
   [`Jolt Core Prover Optimization Inventory`](./jolt-core-prover-optimization-inventory.md)
   whose port target includes `cpu-backend`, either port the optimized
   algorithm into `jolt-backends::cpu` or record a measured replacement that is
   at least as fast on the canonical hardware.
3. Add or extend focused backend microbenchmarks, run them, and inspect the
   time/memory shape before wiring the algorithm through `jolt-prover`.
4. Only after the required backend kernel is `Ported` should the corresponding
   `jolt-prover` frontier be implemented.
5. Only after verifier correctness and measured core performance parity pass
   should a frontier be treated as replacing its `jolt-core` slice.

The harness exposes a backend kernel ledger. A frontier that names an
optimization ID requiring a CPU backend port must have a matching ledger entry.
Goal-mode acceptance should raise the required status from `Required` to
`Ported`, then to `ParityCertified`, before promoting a frontier to production
replacement status.

## Sunset Rule

`jolt-prover-harness` is temporary migration infrastructure. It may depend on
`jolt-core`, `jolt-verifier` core fixtures, `jolt_backends::cpu`, and profiling
tools. No production crate may depend on it.

The crate should be removed once:

- `jolt-prover` is the production prover for transparent Jolt;
- `jolt-prover` is the production prover for BlindFold;
- advice and field-inline prover paths verify through `jolt-verifier`;
- `jolt-core` is no longer needed as the prover parity oracle.

## Design Goals

- Make every implemented prover frontier testable through `jolt-verifier`.
- Prevent stages from landing with only unit tests or compile-only validation.
- Save canonical fixture inputs, outputs, transcript state, and challenge data
  so each stage can be replayed without re-running a long core proof.
- Port the optimized `jolt-core` CPU closure or algorithm for the frontier into
  `jolt-backends::cpu` before accepting the frontier.
- Measure CPU time, memory, and other relevant performance axes against
  `jolt-core` before accepting any slice.
- Keep `jolt-core` and tracer internals out of production `jolt-prover`,
  `jolt-witness`, and `jolt-backends` paths.
- Reuse one harness for transparent, BlindFold, advice, and field-inline
  bring-up.
- Make regression gates mechanical enough that worker agents can run them slice
  by slice.

## Non-Goals

- No production dependency from `jolt-prover` to `jolt-core`.
- No permanent mixed proof format.
- No synthetic-only benchmark suite.
- No acceptance of verifier-invisible comparisons as a substitute for
  `jolt-verifier` correctness.
- No exact proof-byte equality requirement for randomized/ZK paths unless a
  deterministic RNG fixture is explicitly selected.
- No slow full-flow correctness testing as the primary local iteration loop.

## Frontier Definition

A frontier is the smallest coherent prover slice whose output is meaningful to
`jolt-verifier` or to a later verifier-visible proof construction step.

Each frontier has:

- a name;
- required feature modes;
- input fixture shape;
- modular output type;
- correctness gate through `jolt-verifier`;
- performance gate against `jolt-core`;
- optimization-inventory IDs touched;
- performance metrics to collect;
- required tests and benchmark commands.

Representative frontiers:

| Frontier | Modular Owner | Required Gates |
|----------|---------------|----------------|
| preprocessing shape | `jolt-program`, `jolt-witness`, `jolt-prover` config | verifier correctness |
| stage0 commitments | `jolt-prover::stages::stage0`, `jolt_backends::cpu::commitments` | verifier correctness + core performance parity |
| advice commitments | stage0 + witness advice providers + CPU Dory contexts | verifier correctness + core performance parity |
| stage1-2 | `jolt-prover::stages::{stage1,stage2}` + CPU sumcheck kernels | verifier correctness + core performance parity |
| stage3-7 | remaining relation stages | verifier correctness + core performance parity for sumcheck/opening work |
| stage8 openings | `jolt-prover::stages::stage8` + CPU opening kernels | verifier correctness + core performance parity |
| BlindFold | committed sumchecks + `jolt-blindfold` witness material | ZK verifier correctness + core performance parity |
| field-inline | gated witness/stage/CPU extensions | transparent/ZK verifier correctness + core performance parity |

## Gates

The harness has only two gates:

- `VerifierCorrectness`: `jolt-verifier` accepts the modular frontier output,
  using the narrowest replay fixture that exercises the verifier-visible
  surface.
- `CorePerformanceParity`: the modular prover plus selected `jolt-backends::cpu`
  implementation matches `jolt-core` on the relevant time, memory, proof-size,
  and allocation axes for the frontier.

Both gates are mandatory for every frontier.

`CorePerformanceParity` is never optional. A missing required timing or memory
measurement is a gate failure, not a pass. If a metric cannot be measured on a
platform, the frontier is not parity-certified on that platform.

Core fixture comparisons, transcript checkpoints, proof-shape comparisons, and
backend reference checks are tools for implementing those gates. They are not
separate gates and should not appear in the frontier manifest.

Replay fixture equality is not a substitute for a real modular prover path.
Fixture replay can prove that a slice is understood, but replacement requires
`jolt-prover` to produce the verifier-visible object by dispatching optimized
`jolt-backends::cpu` requests through the real transcript lifecycle.

## Replay Fixtures

Canonical fixtures are generated from the canonical SHA-chain workload and from
small smoke workloads. A saved fixture slice records enough data to replay one
frontier without running a long core proof:

- normalized prover inputs;
- prior frontier outputs needed by the slice;
- transcript prefix/state and challenge values owned by `jolt-prover`;
- backend request inputs and expected backend outputs;
- verifier-visible output claims, proof payloads, and opening facts;
- core timing/memory measurements for the matching optimization inventory IDs.

The replay frontier grows additively. Until a prior stage is accepted, the
harness may feed the current stage from saved core fixture outputs. Once that
prior stage is accepted, later stages should consume the modular output by
default and use the core fixture only as an oracle/reference artifact.

## Harness Crate/Layout

Add a dev-only internal harness surface. Preferred layout:

```text
crates/jolt-prover-harness/
  Cargo.toml
  src/
    lib.rs
    fixtures.rs
    frontier.rs
    replay.rs
    compare.rs
    metrics.rs
    perf.rs
    field_inline.rs
  tests/
    frontier_commitments.rs
    frontier_advice.rs
    frontier_stages.rs
    frontier_blindfold.rs
    frontier_field_inline.rs
  benches/
    frontier_perf.rs
```

If a separate crate is too heavy for the first PR, the same modules may live
temporarily under `crates/jolt-prover/tests/support/frontier/`, but they should
be promoted before multiple crates need the helpers.

Rules:

- production crates do not depend on the harness;
- the harness may depend on `jolt-core` behind an explicit `core-fixtures`
  feature;
- the harness may depend on `jolt_backends::cpu` to instantiate the canonical
  backend;
- fixture generation and replay helpers live in the harness, not in
  `jolt-prover`;
- perf code lives in `benches/` or harness modules, not stage orchestration.
- fixture ingestion must adapt every source, including `jolt-core` compatibility
  and synthetic fixtures, into the normalized `jolt-program` execution artifact
  before constructing a witness provider.

Suggested features:

```toml
[features]
default = []
core-fixtures = ["jolt-core/host", "jolt-verifier/core-fixtures"]
zk = ["jolt-prover/zk", "jolt-verifier/zk", "jolt-core/zk"]
field-inline = [
  "jolt-prover/field-inline",
  "jolt-witness/field-inline",
  "jolt-verifier/field-inline",
]
perf = ["dep:jolt-profiling"]
```

## Frontier Manifest

Every implemented frontier should add a manifest entry:

```rust
pub struct FrontierSpec {
    pub name: &'static str,
    pub fixtures: &'static [FixtureKind],
    pub features: &'static [FeatureMode],
    pub gates: &'static [FrontierGate],
    pub perf: Option<PerfGate>,
    pub optimization_ids: &'static [&'static str],
    pub backend_kernel_ports: &'static [&'static str],
}
```

The manifest is review material. A slice is incomplete if it adds prover code
without naming its frontier and gates. `VerifierCorrectness` and
`CorePerformanceParity` are mandatory for every frontier.

Optimization IDs must be present in
[`Jolt Core Prover Optimization Inventory`](./jolt-core-prover-optimization-inventory.md).
Every frontier must name concrete optimization IDs and include a `PerfGate`.
The harness parses the inventory and rejects unknown IDs.

For optimization IDs whose inventory port target includes `cpu-backend`, the
harness also requires backend-kernel accounting:

```rust
pub struct BackendKernelPortSpec {
    pub name: &'static str,
    pub family: BackendKernelFamily,
    pub optimization_ids: &'static [&'static str],
    pub source_locations: &'static [&'static str],
    pub cpu_entrypoints: &'static [&'static str],
    pub microbenchmarks: &'static [&'static str],
    pub certification_evidence_files: &'static [&'static str],
    pub status: KernelPortStatus,
}
```

Status semantics:

```text
Required         inventory item is known and must be ported before prover replacement
Ported           optimized CPU algorithm exists behind a backend request
ParityCertified  backend and prover frontier pass correctness and core perf parity
```

`Required` is accounting, not acceptance. A prover frontier that is intended to
replace core must require `Ported` at implementation time and
`ParityCertified` before completion.

The frontier's `backend_kernel_ports` list must cover every named optimization
ID whose inventory port target includes `cpu-backend`. This prevents a frontier
from getting accidental credit from an unrelated kernel that shares a broad
optimization ID such as `OPT-SC-007` or `OPT-EQ-004`.

The registered backend kernel ledger must also cover every optimization ID in
the global inventory whose port target includes `cpu-backend`.
`validate_global_cpu_backend_inventory_coverage` is the backlog guard: new
inventory rows cannot be added without an explicit kernel ledger owner.

Goal-mode completion must call `validate_frontier_replacement_ready`, not only
manifest validation. That check requires `ParityCertified` kernel accounting and
passing `KernelBenchmarkEvidence` for the relevant backend ports.

`Ported` requires an isolated backend microbenchmark result. A kernel should
not be integrated through a `jolt-prover` stage until the microbench suggests
the shape is plausibly competitive with core on the target hardware. This keeps
iteration local to `jolt-backends` while algorithmic mistakes are still cheap
to diagnose.

`ParityCertified` additionally requires machine-readable benchmark evidence for
the registered kernel. The evidence must name the kernel, benchmark, sample
count, optimization IDs, core metrics, modular metrics, and an analytical memory
budget. Harness validation rejects certification when the benchmark name is not
registered, the optimization IDs do not match exactly, sample count is below the
frontier gate, a required axis is missing, the modular peak exceeds the
analytical budget, or time/RSS exceed the core ratio threshold. Certified
ledger entries must name the JSON evidence files in
`certification_evidence_files`, and
`validate_parity_certified_kernel_evidence_files` must pass against those files.
Bench code should emit evidence through the canonical helper:
`KernelBenchmarkEvidence::write_canonical_json`, which writes under
`target/frontier-metrics/kernel-evidence/<kernel>/<benchmark>.json` with path
components sanitized by the harness.

Required fixture dimensions:

```text
muldiv-small          transparent smoke and compatibility base
fibonacci-small       simple trace sanity
memory-ops            RAM/read-write stress
sha2-chain-2^16       canonical perf workload
sha2-chain-2^20       large confirmation workload
advice-consumer       trusted/untrusted advice paths
field-inline-small    FR operation and bridge rows
zk-muldiv-small       BlindFold smoke
zk-advice-consumer    BlindFold advice path
```

Required feature matrix:

```text
default
field-inline
zk
zk,field-inline
core-fixtures
core-fixtures,zk
core-fixtures,field-inline
core-fixtures,zk,field-inline
```

Each frontier should state which rows of this matrix it touches. Any frontier
that touches feature-gated code must include both enabled and disabled rows.

Field-inline fixtures may be synthetic until the full guest/tracer path is
ready, but synthetic fixtures must exercise the same `jolt-program` execution
contract consumed by `jolt-witness`.

## Prover Input Ingestion

The harness should model the real host/SDK path instead of inventing a separate
test-only data path:

```text
SDK / CLI / harness fixture
  -> jolt_program::execution::JoltProgram
  -> jolt_program::execution::TraceInputs
  -> ExecutionBackend
  -> jolt_program::execution::TraceOutput
  -> jolt_witness::protocols::jolt_vm::JoltVmWitnessProvider
  -> jolt_prover::prove_with_backend(...)
```

Allowed harness sources:

- SDK-style guest fixtures;
- CLI-style prebuilt ELF/image fixtures;
- `jolt-core` compatibility fixtures behind `core-fixtures`;
- synthetic execution artifacts for narrow unit frontiers.

All allowed sources must present the same normalized artifact to
`jolt-witness`. The harness may inspect `jolt-core` or tracer internals to build
compatibility fixtures, but that data must be converted before it crosses the
witness/prover boundary. `jolt-prover` tests should not accept
`tracer::Cycle`, CPU state, lazy trace internals, or core proof internals as
inputs.

Each frontier fixture should record an ingestion descriptor:

```text
surface: sdk | cli | harness-core-fixture | harness-synthetic
artifact: jolt-program-execution | core-compatibility-execution | synthetic-execution
feature mode: transparent | zk | field-inline | zk-field-inline
fixture kind
normalized-jolt-program-artifact: true
uses-tracer-internals-after-boundary: false
```

This descriptor is not a production API; it is a drift guard. A frontier is not
ready for SDK/CLI integration if it can only be driven by direct tracer/core
objects.

## Correctness Gates

Every frontier must have a `VerifierCorrectness` gate. For local iteration,
run the narrowest replay test for the frontier and avoid regenerating large
core fixtures unless the fixture snapshot itself is being updated.

Baseline commands:

```bash
cargo fmt -q
cargo clippy -p jolt-prover -p jolt-backends -p jolt-witness -p jolt-prover-harness -q --all-targets -- -D warnings
cargo nextest run -p jolt-prover -p jolt-backends -p jolt-witness -p jolt-prover-harness --cargo-quiet
```

Feature gates required when touched:

```bash
cargo clippy -p jolt-prover -p jolt-witness -q --all-targets --features field-inline -- -D warnings
cargo clippy -p jolt-prover -q --all-targets --features zk -- -D warnings
cargo clippy -p jolt-prover -q --all-targets --features zk,field-inline -- -D warnings
cargo nextest run -p jolt-prover --cargo-quiet --features zk
cargo nextest run -p jolt-prover --cargo-quiet --features field-inline
cargo nextest run -p jolt-prover --cargo-quiet --features zk,field-inline
```

Harness gates:

```bash
cargo nextest run -p jolt-prover-harness <frontier_replay_test> --cargo-quiet --features core-fixtures
cargo nextest run -p jolt-prover-harness <frontier_replay_test> --cargo-quiet --features core-fixtures,zk
cargo nextest run -p jolt-prover-harness <frontier_replay_test> --cargo-quiet --features core-fixtures,field-inline
```

Use exact test filters per frontier. Broad fixture sweeps and full workspace
E2E runs are regression/acceptance checks, not the default inner loop.

## Iteration Ladder

Use the cheapest gate that can falsify the current change before moving to a
more expensive one:

1. Static ledger and drift checks:
   `cargo nextest run -p jolt-prover-harness optimization_inventory source_drift --cargo-quiet`.
2. Backend unit/reference tests for the touched request family:
   `cargo nextest run -p jolt-backends <kernel_or_request_filter> --cargo-quiet`.
3. Benchmark compile check:
   `cargo bench -p jolt-backends --bench sumcheck_kernels --no-run`.
4. Focused backend microbench on the touched kernel, producing
   `KernelBenchmarkEvidence` through `write_canonical_json`.
5. Narrow frontier replay test for the single stage/fixture.
6. Canonical perf run on `sha2-chain-2^16`, then one confirmation run on
   `sha2-chain-2^20` only after the smaller run passes.
7. Full workspace or broad feature matrix only for regression confidence after
   the local gates pass.

Do not start with broad prover E2E for a kernel algorithm question. If a kernel
fails the unit/reference test or isolated microbench, fix it in `jolt-backends`
before spending prover-path time.

## Performance Gates

Every frontier must have a `CorePerformanceParity` gate. The metric axes are
chosen according to what the frontier touches:

- witness generation;
- commitments;
- PCS openings;
- sumcheck kernels;
- RA/one-hot/shared-eq paths;
- sparse read/write matrices;
- BlindFold private witness construction, row commitments/openings, and folding
  kernels;
- memory retention/drop policy.

Canonical benchmark policy:

```text
workload: sha2-chain
size:     2^16
samples:  3
compare:  jolt-core same branch/config vs modular frontier
gate:     <= 15% regression in frontier time and peak RSS
warn:     > 5% unexplained regression
confirm:  if 2^16 passes, run 2^20 once
```

Required metrics:

- total prove time when available;
- frontier time;
- backend kernel time for the optimized request;
- committed witness time;
- per-stage time for stage frontiers;
- stage8 opening time;
- BlindFold time for ZK frontiers;
- peak RSS;
- allocation count/bytes when available;
- proof size where the frontier changes proof payload;
- qualitative core saturation when available.

For every frontier, timing and peak RSS are required unless a narrower
frontier-specific `PerfGate` explicitly justifies another axis. Proof-size is
required when the frontier changes verifier-visible proof payload. Allocation
count/bytes are required for memory-representation, RA, sparse-matrix, and
Stage 8 work.

Suggested command shape:

```bash
cargo bench -p jolt-prover-harness --bench frontier_perf --features core-fixtures,perf -- sha2-chain --size 65536 --samples 3
cargo bench -p jolt-prover-harness --bench frontier_perf --features core-fixtures,perf -- sha2-chain --size 1048576 --samples 1
```

The harness should emit machine-readable summaries, for example:

```text
target/frontier-metrics/<frontier>/<timestamp>.json
```

Each summary should include:

```json
{
  "frontier": "stage0_commitments",
  "features": ["transparent"],
  "fixture": "sha2-chain",
  "size": 65536,
  "samples": 3,
  "core": {"time_ms": 0, "peak_rss_bytes": 0},
  "modular": {"time_ms": 0, "peak_rss_bytes": 0},
  "ratio": {"time": 1.0, "peak_rss": 1.0},
  "optimization_ids": []
}
```

## Instrumentation

Use stable span names so `jolt-core` and modular paths can be compared.

Required span namespaces:

```text
core::prove::<frontier>
modular::prove::<frontier>
modular::witness::<frontier>
modular::backend::cpu::<frontier>
modular::stage::<stageN>
modular::stage8::opening
modular::blindfold
```

If existing `jolt-core` spans are too coarse, add minimal spans around the
corresponding slice. Do not refactor core prover code just for instrumentation.

Peak memory should be measured consistently across core and modular runs. If
platform-specific RSS measurement is unavailable, the harness must report that
memory was not measured rather than silently passing the memory gate.

## Stage File Pattern

`jolt-prover` stage modules should keep orchestration readable:

```text
stageN/
  mod.rs
  input.rs
  output.rs
  request.rs      // backend request construction, if needed
  prove.rs        // thin orchestration only
  tests.rs
```

The default flow is:

```text
input -> request -> backend result -> output
```

`prove.rs` should show that flow directly. It should not absorb large mapping
tables, dense fixture helpers, backend kernels, or long tests. `output.rs`
owns the typed construction of verifier-visible proof components plus private
state passed to later stages. A separate `assembly.rs` is not part of the
default pattern; if a later stage truly needs a large helper, prefer private
helpers under `output.rs` first and split only after the type becomes too large
to read.

`jolt-backends::cpu` modules should keep the CPU fast path explicit:

```text
cpu/
  mod.rs
  config.rs
  backend.rs
  commitments/
    mod.rs
    stream.rs
  sumcheck/
  openings/
  blindfold/
  field_inline/
  state.rs
  tests.rs
```

Fine-grained reusable kernels may move later, but only after the coarse CPU
backend path proves parity.

Backend contract types should be split by family:

```text
commitments/
  request.rs
  result.rs
sumcheck/
  request.rs
  result.rs
openings/
  request.rs
  result.rs
blindfold/
  request.rs
  result.rs
```

The harness includes structure tests for these rails. If a frontier needs a
new backend family, add the request/result contract first and keep concrete CPU
helpers under `cpu/<family>/`.

Backend requests are hardware-agnostic contracts. Keep protocol-specific CPU
fast paths behind `cpu/<family>/` submodules, using backend-local relation IDs,
witness oracle refs, value slots, and request labels to connect them to the
protocol schedule built by `jolt-prover`.

Every backend request family must include a `CONTRACT.md` documenting:

- who constructs the request;
- what slots mean;
- what witness views are required;
- what backend state may be cached;
- which result fields are verifier-visible versus private;
- which optimization IDs the family preserves;
- feature-gated fields and disabled-feature behavior.

Static drift checks live in the harness and optional Semgrep rules:

```bash
cargo nextest run -p jolt-prover-harness --cargo-quiet
scripts/semgrep-rails.sh
```

The Rust harness tests enforce the critical boundaries even when Semgrep is not
available.

The harness also scans the prover-side specs and crate markdown for retired
terms from earlier designs, including backend-owned planning language and old
proof-construction wording. If the request/result contract changes, update the
docs and the retired-term list together.

## Frontier Review Checklist

Before a frontier lands:

- The frontier manifest entry names fixtures, feature modes, gates, and
  optimization IDs.
- The frontier includes `VerifierCorrectness`.
- The frontier includes `CorePerformanceParity`, concrete optimization IDs, and
  a `PerfGate`.
- Every CPU-backend optimization ID named by the frontier has a backend kernel
  ledger entry with the required status.
- For production replacement, every named CPU-backend optimization ID is
  `ParityCertified`.
- `jolt-verifier` accepts the implemented frontier using the narrowest replay
  fixture that exercises the verifier-visible surface.
- The real optimized core CPU closure/algorithm for the slice is ported into
  `jolt-backends::cpu`; generic fallbacks do not satisfy the gate when core has
  a specialized algorithm.
- Saved replay fixtures include the inputs, outputs, transcript state, and
  backend request/result facts needed to rerun the frontier without a long core
  proof.
- Randomized/ZK fields are compared with deterministic RNG or explicitly marked
  as shape/acceptance-only.
- Backend outputs are slot-keyed; stage output construction is logical-ID
  keyed.
- No transcript operation occurs in `jolt-backends`.
- No lookup semantics are duplicated outside `jolt-lookup-tables` or a
  documented thin adapter.
- No production path depends on `jolt-core` or tracer internals.
- `jolt-prover` does not import concrete `jolt_backends::cpu` modules.
- `jolt-backends` does not depend on `jolt-core`, `jolt-verifier`, or
  `jolt-claims` unless a spec explicitly introduces and justifies an exception.
- `jolt-backends` does not use transcript APIs.
- Prover inputs cross into `jolt-witness` only as normalized `jolt-program`
  execution artifacts.
- Field-inline disabled builds expose no field-inline prover data.
- FR-off field-inline-enabled fixtures are structurally ordinary Jolt.
- `jolt-prover` feature flags forward to matching `jolt-backends` capability
  flags; feature-specific code stays in typed input/request/output or named
  backend capability modules.
- Every frontier includes a benchmark summary for the relevant axes.
- No frontier is accepted as a fixture-only, ad hoc spliced, or
  correctness-only implementation.

## References

- [`jolt-prover` model crate spec](./jolt-prover-model-crate.md)
- [`jolt-prover` CPU backend port spec](./jolt-prover-cpu-backend-port.md)
- [`jolt-witness` crate spec](./jolt-witness-crate.md)
- [`Jolt Core Prover Optimization Inventory`](./jolt-core-prover-optimization-inventory.md)
- [`jolt-verifier` model crate spec](./jolt-verifier-model-crate.md)
