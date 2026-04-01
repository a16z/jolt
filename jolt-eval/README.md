# jolt-eval

Mechanically checkable **invariants** and **objectives** for the Jolt zkVM.

The motivation is twofold:
1. **Maximize agent productivity** -- give AI agents a way to check their work without a human in the loop.
2. **Minimize the human verification surface** -- humans gain assurance about the larger codebase while only focusing on a smaller kernel of invariants and objectives.

## Concepts

**Invariants** are evaluations with a binary outcome -- things that must always hold. From a single invariant description (a small amount of Rust), the framework can synthesize:
- A `#[test]`
- A `libfuzzer_sys` fuzz target
- A "red team" harness for AI agents to try to find a violation

**Objectives** are evaluations with a numerical outcome -- things we want to optimize. They serve as building blocks for AI-driven optimization loops.

## Built-in invariants

| Invariant | Description |
|---|---|
| **Soundness** | Mutated proofs must be rejected by the verifier |
| **Verifier completeness** | Honest proofs must be accepted by the verifier |
| **Prover completeness** | The prover must not panic on valid inputs |
| **Determinism** | Same program + input produces byte-identical proofs |
| **Serialization roundtrip** | `deserialize(serialize(proof)) == proof` |
| **ZK consistency** | Prove + verify succeeds in the current compilation mode (run with both `--features host` and `--features host,zk`) |

## Built-in objectives

| Objective | Direction | Description |
|---|---|---|
| `peak_rss` | Minimize | Peak resident set size during proving (MB) |
| `prover_time` | Minimize | Wall-clock prover time (seconds) |
| `proof_size` | Minimize | Serialized proof byte length |
| `verifier_time` | Minimize | Wall-clock verifier time (seconds) |
| `guest_cycle_count` | Minimize | Guest instruction cycle count |
| `inline_lengths` | Maximize | Count of optimized inline instructions |
| `wrapping_cost` | Minimize | Constraint system size (padded trace length) |

## Usage

### Defining an invariant

```rust
use jolt_eval::{Invariant, InvariantViolation, SynthesisTarget};
use enumset::EnumSet;

#[jolt_eval_macros::invariant(targets = [Test, Fuzz, RedTeam])]
#[derive(Default)]
pub struct MyInvariant;

impl Invariant for MyInvariant {
    type Setup = ();
    type Input = u64;

    fn name(&self) -> &str { "my_invariant" }
    fn description(&self) -> String {
        "Human-readable description, also used as context for AI red-teaming.".into()
    }
    fn targets(&self) -> EnumSet<SynthesisTarget> {
        SynthesisTarget::Test | SynthesisTarget::Fuzz | SynthesisTarget::RedTeam
    }
    fn setup(&self) -> Self::Setup {}
    fn check(&self, _setup: &(), input: u64) -> Result<(), InvariantViolation> {
        // ... check the invariant ...
        Ok(())
    }
    fn seed_corpus(&self) -> Vec<u64> {
        vec![0, 1, u64::MAX]
    }
}
```

The `#[invariant]` macro generates:
- `#[test] fn seed_corpus()` -- runs every seed input through `check`
- `#[test] fn random_inputs()` -- runs 10 randomly-generated `Arbitrary` inputs
- `my_invariant_fuzz_check(data: &[u8])` -- call from a `fuzz_target!` body
- `my_invariant_redteam_description() -> String` -- for the red-team harness

### Running invariants with the CLI

```bash
# Check all invariants against a compiled guest ELF
cargo run --bin check-invariants -- --elf path/to/guest.elf

# Check a specific invariant with more random inputs
cargo run --bin check-invariants -- --elf path/to/guest.elf \
    --invariant soundness --num-random 100
```

### Fuzzing

The `fuzz` binary runs randomized inputs (via the `Arbitrary` trait) against invariants that include `SynthesisTarget::Fuzz`:

```bash
# Fuzz all invariants with 1000 random inputs
cargo run --bin fuzz -- --elf path/to/guest.elf --iterations 1000

# Fuzz a specific invariant with a time limit
cargo run --bin fuzz -- --elf path/to/guest.elf \
    --invariant soundness --duration 5m

# List available fuzzable invariants
cargo run --bin fuzz -- --list
```

For deeper coverage, the `#[invariant]` macro generates a `_fuzz_check` function suitable for use with `cargo fuzz` / `libfuzzer_sys`:

```rust
// fuzz/fuzz_targets/soundness.rs
#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    my_crate::my_soundness_invariant_fuzz_check(data);
});
```

### Measuring objectives

```bash
# Measure all objectives
cargo run --bin measure-objectives -- --elf path/to/guest.elf

# Measure a specific objective with multiple samples
cargo run --bin measure-objectives -- --elf path/to/guest.elf \
    --objective prover_time --samples 5
```

### AI red-teaming

```bash
cargo run --bin redteam -- --elf path/to/guest.elf \
    --invariant soundness --iterations 10 --model claude-sonnet-4-20250514
```

The red-team harness runs the AI agent in an isolated git worktree. The invariant is checked in the original working tree so the agent cannot cheat.

### AI-driven optimization

```bash
# Optimize prover_time and proof_size over 5 iterations
cargo run --bin optimize -- --elf path/to/guest.elf \
    --objectives prover_time,proof_size --iterations 5

# With a hint to guide the agent
cargo run --bin optimize -- --elf path/to/guest.elf \
    --hint "Focus on the sumcheck inner loop in jolt-core/src/subprotocols/"
```

Each iteration: the agent works in an isolated worktree, the diff is applied to the real repo, objectives are re-measured, invariants are checked, and the change is committed or reverted.

### Programmatic API

```rust
use std::sync::Arc;
use jolt_eval::{TestCase, SharedSetup, check_all_invariants};
use jolt_eval::invariant::soundness::SoundnessInvariant;

// Create a test case from a compiled guest program
let test_case = Arc::new(TestCase { elf_contents, memory_config, max_trace_length: 65536 });

// Run a specific invariant
let inv = SoundnessInvariant::new(Arc::clone(&test_case), default_inputs);
let results = inv.run_checks(/* num_random */ 10);

// Or measure objectives
use jolt_eval::objective::{Objective, prover_time::ProverTimeObjective};
let setup = SharedSetup::new(test_case);
let obj = ProverTimeObjective::new(setup.test_case, setup.prover_preprocessing, inputs);
let seconds = obj.collect_measurement().unwrap();
```

## Framing tasks in terms of invariants and objectives

| Task | Invariants | Objectives |
|---|---|---|
| **New feature** | Add new invariants capturing the feature's behavior; modify existing invariants as necessary | Document expected impact; mechanically validate |
| **Bug fix** | Add/modify invariant to fail without the fix; verify all others still hold | Document impact |
| **Security review** | Try to find a counterexample to some invariant (via red-team) | -- |
| **Optimization** | Ensure all invariants still hold | Maximize an objective function $f(o_1, \ldots, o_n)$ |
| **Refactor** | Ensure all invariants still hold | Special case of optimization where the objective captures code quality |
