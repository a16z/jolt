---
name: new-objective
description: Implement a new objective for jolt-eval
argument-hint: "<objective-name>"
---

<Purpose>
Implement a new objective for the jolt-eval optimization framework. An objective is a measurable quantity that the AI optimizer tries to minimize — either a static analysis metric or a performance benchmark.

This skill handles all the boilerplate: creating the objective struct, implementing the `Objective` trait, registering it in the appropriate enum, creating a const key, adding an `ObjectiveFunction`, creating a Criterion benchmark, and running `sync_targets.sh`.
</Purpose>

<Execution_Policy>
- The user must provide an objective name (lowercase with underscores, e.g. `cyclomatic_complexity`).
- Use the request, spec, and repository context to identify the measurement and objective kind; ask only for missing requirements.
- Follow existing patterns exactly — study lloc.rs (static analysis) and binding.rs (performance) as models.
- Always run clippy and tests before reporting success.
</Execution_Policy>

<Steps>

## Phase 1: Gather Requirements

1. Validate the argument `{{ARGUMENTS}}`: must be a valid Rust identifier (lowercase alphanumeric + underscores). Reject otherwise.
2. Gather these requirements from existing context; ask the user only for missing information:
   - What is being measured? (becomes the `description()`)
   - Is this a **static analysis** objective or a **performance** objective?
     - Static analysis: computes a metric by analyzing source code (e.g. lines of code, complexity). Overrides `collect_measurement()`, uses `Setup = ()`.
     - Performance: computes a metric by running/profiling some computation (e.g. wall-clock time, max RSS). Overrides `setup()` and `run()`.
   - What are the units? (e.g. "lines", "s", "bugs")
   - What files/directories does this objective measure? (used for `diff_paths()` scoping)

## Phase 2: Explore Context

1. Read `jolt-eval/src/objective/mod.rs` to understand the current enums and dispatch methods.
2. Read `jolt-eval/src/objective/objective_fn/mod.rs` to understand objective function registration.
3. Read an existing objective for reference:
   - Static analysis: `jolt-eval/src/objective/code_quality/lloc.rs`
   - Performance: `jolt-eval/src/objective/performance/binding.rs`
4. If the objective measures jolt-prover-legacy functionality, explore the relevant modules.

## Phase 3: Implement

Create the objective file following the full template for the objective kind (both in this skill's directory):

- **Static analysis**: `jolt-eval/src/objective/code_quality/<objective_name>.rs` — follow `references/static-analysis.md`
- **Performance**: `jolt-eval/src/objective/performance/<objective_name>.rs` — follow `references/performance.md` (includes Criterion setup/run guidelines)

## Phase 4: Register in Enums

### 4a. Add module declaration

Edit the appropriate `mod.rs`:
- Static analysis: `jolt-eval/src/objective/code_quality/mod.rs` — add `pub mod <objective_name>;`
- Performance: `jolt-eval/src/objective/performance/mod.rs` — add `pub mod <objective_name>;`

### 4b. Add enum variant and dispatch

Edit `jolt-eval/src/objective/mod.rs`:

**For static analysis**, add to `StaticAnalysisObjective`:
1. New variant in the enum
2. Entry in `all()` with the `target_dir` field
3. Match arm in every dispatch method: `name()`, `description()`, `collect_measurement()`, `collect_measurement_in()`, `units()`

**For performance**, add to `PerformanceObjective`:
1. New variant in the enum
2. Entry in `all()`
3. Match arm in every dispatch method: `name()`, `units()`, `description()`
4. Match arm in `diff_paths()` — return the appropriate path slice

### 4c. Add const re-export

Add a `pub use` line in `jolt-eval/src/objective/mod.rs`:
```rust
pub use <submodule>::<objective_name>::<UPPER_NAME>;
```

## Phase 5: Add Objective Function

Edit `jolt-eval/src/objective/objective_fn/mod.rs`:

1. Import the const key:
   ```rust
   use super::{..., <UPPER_NAME>};
   ```

2. Add a const `ObjectiveFunction`:
   ```rust
   pub const MINIMIZE_<UPPER_NAME>: ObjectiveFunction = ObjectiveFunction {
       name: "minimize_<objective_name>",
       inputs: &[<UPPER_NAME>],
       evaluate: |m, _| m.get(&<UPPER_NAME>).copied().unwrap_or(f64::INFINITY),
   };
   ```

3. Add it to `ObjectiveFunction::all()`.

## Phase 6: Create Criterion Benchmark (performance objectives only)

Performance objectives are measured via Criterion — without the bench file, the optimization harness can't measure them. Create `jolt-eval/benches/<objective_name>.rs`:

```rust
use jolt_eval::objective::performance::<objective_name>::<Name>Objective;
jolt_eval::bench_objective!(<Name>Objective);
```

Then run `./jolt-eval/sync_targets.sh` to update `Cargo.toml` with the new `[[bench]]` entry.

## Phase 7: Validate

Run these commands (all must pass):

```bash
# Format
cargo fmt -q

# Lint
cargo clippy -p jolt-eval -q --all-targets -- -D warnings

# Run tests
cargo nextest run -p jolt-eval --cargo-quiet

# For static analysis objectives, verify the measurement works
cargo run -p jolt-eval --bin measure-objectives -- --objective <objective_name>

# For performance objectives, verify the benchmark compiles
cargo bench -p jolt-eval --bench <objective_name> -- --test
```

If any step fails, fix the issue and re-run.

</Steps>

Task: Implement a new objective for jolt-eval. {{ARGUMENTS}}
