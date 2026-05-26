# Spec: `jolt-prover` Frontier Harness

| Field | Value |
|-------|-------|
| Author(s) | Markos Georghiades, Codex |
| Created | 2026-05-26 |
| Status | draft |
| PR | TBD |

## Summary

This spec defines the correctness, parity, and performance harness required to
finish the modular `jolt-prover` migration without losing confidence or CPU
fast-path performance. The harness is intentionally built before the remaining
prover stages are ported.

The goal is incremental proof migration:

```text
one verifier-visible prover frontier at a time
  -> modular implementation
  -> verifier acceptance
  -> jolt-core parity where meaningful
  -> CPU performance parity
  -> next frontier
```

The harness is not production proving code. It exists to compare modular output
against verifier expectations and `jolt-core` compatibility artifacts while the
modular stack becomes sovereign.

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
- Compare modular output with `jolt-core` where deterministic or structurally
  comparable.
- Measure CPU time and memory before accepting a performance-sensitive slice.
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
- No acceptance of verifier-invisible parity as a substitute for verifier
  acceptance.
- No exact proof-byte equality requirement for randomized/ZK paths unless a
  deterministic RNG fixture is explicitly selected.

## Frontier Definition

A frontier is the smallest coherent prover slice whose output is meaningful to
`jolt-verifier` or to a later verifier-visible proof construction step.

Each frontier has:

- a name;
- required feature modes;
- input fixture shape;
- modular output type;
- verifier acceptance target;
- `jolt-core` parity target where applicable;
- optimization-inventory IDs touched;
- performance metrics to collect;
- required tests and benchmark commands.

Representative frontiers:

| Frontier | Modular Owner | Acceptance Target |
|----------|---------------|-------------------|
| preprocessing shape | `jolt-program`, `jolt-witness`, `jolt-prover` config | parity with verifier/core preprocessing facts |
| stage0 commitments | `jolt-prover::stages::stage0`, `jolt_backends::cpu::commitments` | grafted `JoltCommitments` accepted by verifier |
| advice commitments | stage0 + witness advice providers + CPU Dory contexts | advice fixture accepted by verifier |
| stage1-2 | `jolt-prover::stages::{stage1,stage2}` + CPU sumcheck kernels | verifier stage acceptance / full proof graft |
| stage3-7 | remaining relation stages | verifier stage acceptance / full proof graft |
| stage8 openings | `jolt-prover::stages::stage8` + CPU opening kernels | final PCS proof verifies |
| BlindFold | committed sumchecks + `jolt-blindfold` witness material | full ZK verifier acceptance |
| field-inline | gated witness/stage/CPU extensions | transparent and ZK verifier acceptance |

## Acceptance Modes

### Full-Proof Graft Mode

Use this when a modular frontier produces verifier-visible data that can replace
the corresponding data inside a `jolt-core`-derived fixture proof without
invalidating later transcript-dependent fields.

Flow:

```text
core fixture proof
  -> convert to jolt-verifier proof
  -> replace frontier payload with modular output
  -> run jolt-verifier::verify
```

Examples:

- commitment payloads when transcript-compatible;
- proof metadata/config fields;
- deterministic stage outputs where later proof material remains valid.

Required checks:

- the grafted proof verifies;
- proof-shape fields match the target `JoltProtocolConfig`;
- no backend vector order is used to select verifier fields;
- tampering the grafted modular payload is rejected when the verifier surface
  already binds that payload.

### Prefix/Checkpoint Mode

Use this when replacing a frontier changes transcript challenges or later proof
payloads, so full grafting would invalidate the rest of a core proof.

Flow:

```text
same fixture input
  -> core compatibility run to checkpoint
  -> modular run to checkpoint
  -> compare transcript-visible facts, output claims, opening events, and
     verifier-stage acceptance for the implemented prefix
```

Examples:

- stages whose output challenges feed later stages;
- committed/BlindFold sumchecks with RNG-dependent commitments;
- stage8 joint-opening construction before full sovereign proof exists.

Required checks:

- all checkpoint facts named in the frontier manifest match or are accepted by
  the corresponding verifier-stage routine;
- randomized values use deterministic RNG fixtures for exact comparison, or
  compare verifier acceptance and shape instead of bytes;
- every unmatched value is recorded as intentionally random, backend-private,
  or not yet ported.

### Sovereign Mode

Use this once all dependencies for a proof path are modular.

Flow:

```text
jolt-program execution artifact
  -> jolt-witness provider
  -> jolt-prover + selected backend
  -> jolt-verifier::verify
```

This is the final state for transparent, BlindFold, advice, and field-inline
proofs.

## Harness Crate/Layout

Add a dev-only internal harness surface. Preferred layout:

```text
crates/jolt-prover-harness/
  Cargo.toml
  src/
    lib.rs
    fixtures.rs
    frontier.rs
    graft.rs
    parity.rs
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
- fixture generation and graft helpers live in the harness, not in
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
    pub mode: AcceptanceMode,
    pub fixtures: &'static [FixtureKind],
    pub features: &'static [FeatureMode],
    pub parity: &'static [ParityTarget],
    pub perf: Option<PerfGate>,
    pub optimization_ids: &'static [&'static str],
}
```

The manifest is review material. A slice is incomplete if it adds prover code
without naming its frontier and gate.

Optimization IDs must be present in
[`Jolt Core Prover Optimization Inventory`](./jolt-core-prover-optimization-inventory.md),
unless the frontier is explicitly non-performance-sensitive and uses the
reserved `NON-PERF` marker. The harness parses the inventory and rejects
unknown IDs.

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

Every frontier must pass:

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
cargo nextest run -p jolt-prover-harness frontier_ --cargo-quiet --features core-fixtures
cargo nextest run -p jolt-prover-harness frontier_ --cargo-quiet --features core-fixtures,zk
cargo nextest run -p jolt-prover-harness frontier_ --cargo-quiet --features core-fixtures,field-inline
```

Use exact test filters per frontier once the tests exist. Do not require full
workspace E2E on every local edit, but each integrated frontier should run the
matching harness gate.

## Performance Gates

Performance gates apply to any frontier that touches:

- witness generation;
- commitments;
- PCS openings;
- sumcheck kernels;
- RA/one-hot/shared-eq paths;
- sparse read/write matrices;
- BlindFold private witness construction;
- memory retention/drop policy.

Canonical benchmark policy:

```text
workload: sha2-chain
size:     2^16
samples:  3
compare:  jolt-core same branch/config vs modular frontier
gate:     <= 10% regression in frontier time and peak RSS
warn:     > 5% unexplained regression
confirm:  if 2^16 passes, run 2^20 once
```

Required metrics:

- total prove time when available;
- frontier time;
- committed witness time;
- per-stage time for stage frontiers;
- stage8 opening time;
- BlindFold time for ZK frontiers;
- peak RSS;
- proof size where the frontier changes proof payload;
- qualitative core saturation when available.

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

- The frontier manifest entry names fixtures, modes, parity targets, and
  optimization IDs.
- `jolt-verifier` accepts the implemented frontier through full-proof graft,
  prefix/checkpoint, or sovereign mode.
- `jolt-core` parity is checked where deterministic or structurally
  comparable.
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
- Performance-sensitive changes include a benchmark summary or a documented
  reason why the frontier is not performance-sensitive.

## References

- [`jolt-prover` model crate spec](./jolt-prover-model-crate.md)
- [`jolt-prover` CPU backend port spec](./jolt-prover-cpu-backend-port.md)
- [`jolt-witness` crate spec](./jolt-witness-crate.md)
- [`Jolt Core Prover Optimization Inventory`](./jolt-core-prover-optimization-inventory.md)
- [`jolt-verifier` model crate spec](./jolt-verifier-model-crate.md)
