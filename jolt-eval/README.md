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

**Objectives** come in two flavors:
- **Code quality** (static analysis) — measured via `rust-code-analysis`: LLOC, cognitive complexity, Halstead bugs
- **Performance** (benchmarks) — measured via Criterion: polynomial binding, end-to-end prover time

## Built-in invariants

| Invariant | Targets | Description |
|---|---|---|
| `split_eq_bind_low_high` | Test, Fuzz | `GruenSplitEqPolynomial::bind` (LowToHigh) matches `DensePolynomial::bound_poly_var_bot` |
| `split_eq_bind_high_low` | Test, Fuzz | `GruenSplitEqPolynomial::bind` (HighToLow) matches `DensePolynomial::bound_poly_var_top` |
| `soundness` | RedTeam | For any deterministic guest program + input, only one (output, panic) pair is accepted by the verifier |

## Built-in objectives

### Code quality (static analysis)

| Objective | Direction | Units | Description |
|---|---|---|---|
| `lloc` | Minimize | lines | Total logical lines of code in `jolt-core/src/` |
| `cognitive_complexity_avg` | Minimize | — | Average cognitive complexity per function |
| `halstead_bugs` | Minimize | — | Estimated delivered bugs (Halstead volume / 3000) |

### Performance (Criterion benchmarks)

| Benchmark | Description |
|---|---|
| `bind_parallel_low_to_high` | `DensePolynomial::bind_parallel` with LowToHigh binding (2^20 evaluations) |
| `bind_parallel_high_to_low` | `DensePolynomial::bind_parallel` with HighToLow binding (2^20 evaluations) |
| `prover_time_fibonacci_100` | End-to-end prover time for `fibonacci(100)` |
| `prover_time_sha2_chain_100` | End-to-end prover time for 100 iterations of SHA-256 chain |
| `prover_time_secp256k1_ecdsa_verify` | End-to-end prover time for secp256k1 ECDSA signature verification |

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
cargo run --release -p jolt-eval --bin redteam -- \
    --invariant soundness --iterations 10 \
    --hint "Look for edge cases in the memory layout"
```

The red-team harness runs the AI agent in an isolated git worktree. For the soundness invariant, the agent can edit `guest-sandbox/` directly — the harness captures the diff automatically via `git diff`.

### AI-driven optimization

```bash
cargo run --release -p jolt-eval --bin optimize -- \
    --objectives lloc,cognitive_complexity_avg --iterations 5 \
    --hint "Focus on reducing complexity in jolt-core/src/subprotocols/"
```

Each iteration: the agent works in an isolated worktree, the diff is applied, objectives are re-measured (including Criterion benchmarks with `--save-baseline`), invariants are checked, and the change is committed or reverted.

### Defining a performance benchmark

Implement `PerfObjective` and create a bench file:

```rust
// src/objective/performance/my_bench.rs
use crate::objective::PerfObjective;

#[derive(Default)]
pub struct MyBenchObjective;

impl PerfObjective for MyBenchObjective {
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
