# jolt-eval

Mechanically checkable **invariants** and **objectives** for the Jolt zkVM.

The motivation is twofold:
1. **Maximize agent productivity** — give AI agents a way to check their work without a human in the loop.
2. **Minimize the human verification surface** — humans gain assurance about the larger codebase while only focusing on a smaller kernel of invariants and objectives.

## Concepts

**Invariants** are evaluations with a binary outcome — things that must always hold. From a single invariant description (a small amount of Rust), the framework can synthesize:
- A `#[test]` (via the `#[invariant(Test, Fuzz)]` macro)
- A `libfuzzer_sys` fuzz target (via the `fuzz_invariant!` macro)
- A "red team" harness for AI agents to try to find a violation

**Objectives** are measurable properties of the codebase. They come in three flavors:
- **Code quality** (static analysis) — measured via `rust-code-analysis`: LLOC, cognitive complexity, Halstead bugs
- **Performance** (benchmarks) — measured via Criterion: polynomial binding, end-to-end prover time
- **Instruction count** (microbenchmarks) — measured via iai-callgrind: isolated deterministic kernels

**Objective functions** combine one or more objectives into a single scalar that the optimizer minimizes. They are declared as `const` structs with a name, input objectives, and an evaluate function.

## Built-in invariants

| Invariant | Targets | Description |
|---|---|---|
| `split_eq_bind_low_high` | Test, Fuzz, RedTeam | `GruenSplitEqPolynomial::bind` (LowToHigh) matches `DensePolynomial::bound_poly_var_bot` |
| `split_eq_bind_high_low` | Test, Fuzz | `GruenSplitEqPolynomial::bind` (HighToLow) matches `DensePolynomial::bound_poly_var_top` |
| `soundness` | RedTeam | For any deterministic guest program + input, only one (output, panic) pair is accepted by the verifier |

## Built-in objectives

### Code quality (static analysis)

| Objective | Units | Description |
|---|---|---|
| `lloc` | lines | Total logical lines of code in `crates/jolt-prover-legacy/src/` |
| `cognitive_complexity_avg` | — | Average cognitive complexity per function |
| `halstead_bugs` | — | Estimated delivered bugs (Halstead volume / 3000) |

### Performance (Criterion benchmarks)

| Benchmark | Description |
|---|---|
| `bind_parallel_low_to_high` | `DensePolynomial::bind_parallel` with LowToHigh binding (2^20 evaluations) |
| `bind_parallel_high_to_low` | `DensePolynomial::bind_parallel` with HighToLow binding (2^20 evaluations) |
| `hamming_weight_pushforward` | Stage-7 one-hot pushforward (2^22 rows, production 8-bit/16-2-3 geometry) |
| `naive_sort_time` | Wall-clock time of the `naive_sort` function in `jolt-eval/src/sort_targets.rs` |
| `prover_time_fibonacci_100` | End-to-end prover time for `fibonacci(100)` |
| `prover_time_sha2_chain_100` | End-to-end prover time for 100 iterations of SHA-256 chain |
| `prover_time_secp256k1_ecdsa_verify` | End-to-end prover time for secp256k1 ECDSA signature verification |

Note: `prover_time_*` benchmarks are standalone Criterion bench targets (run via `cargo bench -p jolt-eval --bench <name>`). They are **not** included in `PerformanceObjective::all()` and are not tracked by the `optimize` or `measure-objectives` binaries.

### Telemetry (modular prover summary.json)

A string-keyed, parameterized objective family over the modular prover's
telemetry pipeline (see `book/src/usage/profiling/zkvm_profiling.md`):
measurement runs `cargo run --release -p jolt-prover --features profiling --
profile --name <workload> --scale <scale> --format chrome` as a subprocess in
the work dir and reads the metric from
`benchmark-runs/latest_modular_{workload}_{scale}/summary.json` (through the
symlink the profile harness flips to the newest successful run).
Key grammar, pinned:

```text
telemetry:<workload>:<metric>
<workload> ::= fibonacci | sha2-chain | sha3-chain | btreemap
<metric>   ::= prover_time_s         (root-span duration, seconds)
             | peak_rss_gib          (process-lifetime getrusage high-water mark)
             | peak_memory_gib       (max over memory samples in the root span)
             | total:<span-label>    (inclusive time summed over all instances, seconds)
             | self:<span-label>     (exclusive time summed over all instances, seconds)
             | heap:<snapshot>       (allocative mid-stage snapshot total, exact bytes)
             | heap:<snapshot>:<root> (one root frame's bytes; root is verbatim, may contain ':')
```

Everything after the third colon is the **verbatim span label and may itself
contain `:`** (e.g. `telemetry:sha2-chain:total:EqPolynomial::evals`), so an
optimization agent can target any span it discovers in a trace without
editing `jolt-eval`. Measurement uses each workload's scale from the
normative table in `src/objective/telemetry.rs` (fibonacci 2^16, sha2-chain
2^22, sha3-chain 2^22, btreemap 2^20). A key referencing a label absent from
the summary is a **measurement error, never 0.0**. `heap:` metrics build the
profile subprocess with the `allocative` feature automatically (the
optimizer's shared per-workload run enables it when any sharer needs it) and
report exact bytes; snapshot labels are the flamegraph names
(`Stage2Batch_prepared`, …) and root frames are the kernel type names from
the summary's `heap` section.

```bash
cargo run -p jolt-eval --bin measure-objectives -- \
    --objective telemetry:fibonacci:prover_time_s
```

### Instruction counts (iai-callgrind, opt-in)

Deterministic instruction-count objectives over the microbenchmarks in
`benches/callgrind/` — the noise-free signal for optimizer accept/reject
decisions. Key grammar: `callgrind:<bench-name>:instructions` (the `Ir`
event kind, parsed from iai-callgrind's `--output-format=json`).

Opt-in: requires Valgrind plus `cargo install iai-callgrind-runner`
(version matching the workspace `iai-callgrind`); no CI job. Absent
tooling produces a clear measurement error, never a silent zero.

```bash
# Run the bench directly
cargo bench -p jolt-eval --bench eq_evals

# Or measure as an objective
cargo run -p jolt-eval --bin measure-objectives -- \
    --objective callgrind:eq_evals:instructions
```

Callgrind bench targets keep an explicit `path = "benches/callgrind/..."`
in `Cargo.toml`; `sync_targets.sh` regenerates those entries alongside the
Criterion ones.

### Metal single-kernel scaffold

`benches/metal/metal_fr_bind.rs` is the GPU timing template: setup and buffer
wrapping outside the timed loop; one synchronous dispatch per sample; a
shared GPU lock. Run it on macOS with:

```bash
cargo bench -p jolt-eval --features metal --bench metal_fr_bind
```

Metal kernels use Criterion wall time because Callgrind observes host
instructions, not device instructions. Copy the scaffold into
`benches/metal/`, replace `KernelId`, params, buffers, and thread count, then
run `sync_targets.sh`.

### Objective functions

| Name | Inputs | Description |
|---|---|---|
| `minimize_lloc` | lloc | Minimize logical lines of code |
| `minimize_cognitive_complexity` | cognitive_complexity_avg | Minimize average cognitive complexity |
| `minimize_halstead_bugs` | halstead_bugs | Minimize estimated delivered bugs |
| `minimize_bind_low_to_high` | bind_parallel_low_to_high | Minimize LowToHigh binding time |
| `minimize_bind_high_to_low` | bind_parallel_high_to_low | Minimize HighToLow binding time |
| `minimize_naive_sort_time` | naive_sort_time | Minimize naive sort wall-clock time (e2e test target) |
| `minimize_modular_prover_time` | telemetry:fibonacci:prover_time_s | Minimize modular-prover e2e time |
| `minimize_modular_commit_time` | telemetry:fibonacci:total:commit_witness | Minimize witness commitment time |
| `minimize_modular_round_loop_time` | telemetry:fibonacci:total:prove_batch | Minimize the batched sumcheck round loop |
| `minimize_modular_stage_totals` | telemetry:fibonacci:total:prove_stage0..8 | Minimize the sum of per-stage inclusive totals |

Custom composite objective functions can be defined as `ObjectiveFunction` structs:

```rust
use jolt_eval::objective::objective_fn::ObjectiveFunction;
use jolt_eval::objective::{normalized, LLOC, HALSTEAD_BUGS};

const WEIGHTED_QUALITY: ObjectiveFunction = ObjectiveFunction {
    name: "weighted_quality",
    inputs: &[LLOC, HALSTEAD_BUGS],
    evaluate: |m, _b| {
        2.0 * m.get(&LLOC).unwrap_or(&0.0) + m.get(&HALSTEAD_BUGS).unwrap_or(&0.0)
    },
};
```

## Usage

### Defining an invariant

```rust
use jolt_eval::{Invariant, CheckError, InvariantViolation};

#[jolt_eval_macros::invariant(Test, Fuzz)]
#[derive(Default)]
pub struct MyInvariant;

impl Invariant for MyInvariant {
    type Setup = ();
    type Input = u64;

    fn name(&self) -> &str { "my_invariant" }
    fn description(&self) -> String {
        "Human-readable description, also used as context for AI red-teaming.".into()
    }
    fn setup(&self) -> Self::Setup {}
    fn check(&self, _setup: &(), input: u64) -> Result<(), CheckError> {
        // ... check the invariant ...
        Ok(())
    }
    fn seed_corpus(&self) -> Vec<u64> {
        vec![0, 1, u64::MAX]
    }
}
```

The `#[invariant(Test, Fuzz)]` macro generates:
- `InvariantTargets` impl returning the listed targets
- `#[test] fn seed_corpus()` — runs every seed input through `check`
- `#[test] fn random_inputs()` — runs `JOLT_RANDOM_ITERS` (default 10) randomly-generated inputs

### Fuzzing

Each fuzz target is a 3-line file in `fuzz/fuzz_targets/`:

```rust
#![no_main]
use jolt_eval::invariant::split_eq_bind::SplitEqBindLowHighInvariant;
jolt_eval::fuzz_invariant!(SplitEqBindLowHighInvariant::default());
```

Run with `cargo fuzz`:
```bash
cd jolt-eval/fuzz
cargo fuzz run split_eq_bind_low_high
```

### Measuring objectives

```bash
# All objectives (static analysis + Criterion benchmarks)
cargo run -p jolt-eval --bin measure-objectives

# Static analysis only (skip benchmarks)
cargo run -p jolt-eval --bin measure-objectives -- --no-bench

# A specific objective
cargo run -p jolt-eval --bin measure-objectives -- --objective lloc
```

### Running Criterion benchmarks directly

```bash
# All benchmarks
cargo bench -p jolt-eval

# A specific benchmark
cargo bench -p jolt-eval --bench prover_time_fibonacci

# Quick mode (fewer samples)
cargo bench -p jolt-eval --bench bind_parallel_low_to_high -- --quick
```

Criterion results are saved to `target/criterion/` (symlinked from `jolt-eval/benches/results/`).

### AI red-teaming

```bash
# List available invariants
cargo run --release -p jolt-eval --bin redteam -- --list

# Red-team a specific invariant
cargo run --release -p jolt-eval --bin redteam -- \
    --invariant soundness --iterations 10 \
    --hint "Look for edge cases in the memory layout"

# Run the built-in e2e sort test
cargo run --release -p jolt-eval --bin redteam -- --test --verbose
```

The red-team harness runs the AI agent in an isolated git worktree. For the soundness invariant, the agent can edit `guest-sandbox/` directly — the harness captures the diff automatically via `git diff`.

### AI-driven optimization

```bash
# List available objective functions
cargo run --release -p jolt-eval --bin optimize -- --list

# Optimize a specific objective function
cargo run --release -p jolt-eval --bin optimize -- \
    --objective minimize_lloc --iterations 5 \
    --hint "Focus on reducing complexity in crates/jolt-prover-legacy/src/subprotocols/"

# With a custom result branch name
cargo run --release -p jolt-eval --bin optimize -- \
    --objective minimize_lloc --branch my-optimization

# Run the built-in e2e sort test
cargo run --release -p jolt-eval --bin optimize -- --test --verbose
```

Each iteration: the agent works in an isolated worktree, the diff is applied, objectives are re-measured (including Criterion benchmarks with `--save-baseline`), invariants are checked, and the change is committed or reverted. The result is a git branch (`auto-optimize/{objective}-{timestamp}` by default, or a custom name via `--branch`) with one commit per accepted iteration.

### Defining a performance benchmark

Implement the `Objective` trait and create a bench file:

```rust
// src/objective/performance/my_bench.rs
use crate::objective::Objective;

#[derive(Default)]
pub struct MyBenchObjective;

impl Objective for MyBenchObjective {
    type Setup = MySetup;
    fn name(&self) -> &str { "my_bench" }
    fn setup(&self) -> MySetup { /* one-time setup */ }
    fn run(&self, setup: MySetup) { /* hot path */ }
}
```

```rust
// benches/my_bench.rs
use jolt_eval::objective::performance::my_bench::MyBenchObjective;
jolt_eval::bench_objective!(MyBenchObjective);

// Or with custom Criterion config for slow benchmarks:
jolt_eval::bench_objective!(
    MyBenchObjective::default(),
    config: sample_size(10), sampling_mode(::criterion::SamplingMode::Flat),
);
```

Then run `./sync_targets.sh` to update `Cargo.toml` bench entries.

## Syncing targets

```bash
./jolt-eval/sync_targets.sh
```

This script:
- Scans `src/invariant/` for `#[invariant(...Fuzz...)]` structs and generates/removes fuzz target files + `fuzz/Cargo.toml` entries
- Scans `benches/*.rs` and updates `Cargo.toml` `[[bench]]` entries

Bench files are hand-authored (they carry domain-specific Criterion config). The script only syncs `Cargo.toml` entries from existing files.

## Framing tasks in terms of invariants and objectives

| Task | Invariants | Objectives |
|---|---|---|
| **New feature** | Add new invariants capturing the feature's behavior; modify existing invariants as necessary | Document expected impact; mechanically validate |
| **Bug fix** | Add/modify invariant to fail without the fix; verify all others still hold | Document impact |
| **Security review** | Try to find a counterexample to some invariant (via red-team) | — |
| **Optimization** | Ensure all invariants still hold | Maximize an objective function f(o₁, …, oₙ) |
| **Refactor** | Ensure all invariants still hold | Special case of optimization where the objective captures code quality |
