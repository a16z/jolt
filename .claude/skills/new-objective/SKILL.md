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
- Ask the user what is being measured and whether it's a static analysis or performance objective.
- Follow existing patterns exactly — study lloc.rs (static analysis) and binding.rs (performance) as models.
- Always run clippy and tests before reporting success.
</Execution_Policy>

<Steps>

## Phase 1: Gather Requirements

1. Validate the argument `{{ARGUMENTS}}`: must be a valid Rust identifier (lowercase alphanumeric + underscores). Reject otherwise.
2. Ask the user:
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
4. If the objective measures jolt-core functionality, explore the relevant modules.

## Phase 3: Implement

### For Static Analysis Objectives

Create `jolt-eval/src/objective/code_quality/<objective_name>.rs`:

```rust
use std::path::Path;
use crate::objective::{
    MeasurementError, Objective, OptimizationObjective, StaticAnalysisObjective,
};

pub const <UPPER_NAME>: OptimizationObjective =
    OptimizationObjective::StaticAnalysis(StaticAnalysisObjective::<VariantName>(<Name>Objective {
        target_dir: "<target_directory>",
    }));

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct <Name>Objective {
    pub(crate) target_dir: &'static str,
}

impl <Name>Objective {
    pub fn collect_measurement_in(&self, repo_root: &Path) -> Result<f64, MeasurementError> {
        let src_dir = repo_root.join(self.target_dir);
        // Implement measurement logic
        todo!()
    }
}

impl Objective for <Name>Objective {
    type Setup = ();

    fn name(&self) -> &str { "<objective_name>" }

    fn description(&self) -> String {
        format!("Description of measurement in {}", self.target_dir)
    }

    fn setup(&self) {}

    fn collect_measurement(&self) -> Result<f64, MeasurementError> {
        let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
        self.collect_measurement_in(repo_root)
    }

    fn units(&self) -> Option<&str> { Some("units") }
}
```

### For Performance Objectives

Create `jolt-eval/src/objective/performance/<objective_name>.rs`:

```rust
use crate::objective::Objective;

pub const <UPPER_NAME>: OptimizationObjective =
    OptimizationObjective::Performance(PerformanceObjective::<VariantName>(<Name>Objective));

pub struct <Name>Setup {
    // Pre-computed data for each iteration
}

#[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct <Name>Objective;

impl Objective for <Name>Objective {
    type Setup = <Name>Setup;

    fn name(&self) -> &str { "<objective_name>" }

    fn description(&self) -> String {
        "What is being benchmarked and at what scale".to_string()
    }

    fn setup(&self) -> <Name>Setup {
        // Use thread_local! { static SHARED: ... } pattern for expensive one-time init
        // that should be amortized across Criterion iterations.
        // Return a fresh Setup that can be consumed by run().
        todo!()
    }

    fn run(&self, setup: <Name>Setup) {
        // The hot path that Criterion measures.
        // Use std::hint::black_box() to prevent dead-code elimination.
        todo!()
    }

    fn units(&self) -> Option<&str> { Some("s") }
}
```

**Performance objective guidelines:**
- Use `thread_local!` with a `Shared` struct for expensive setup (random data generation, etc.) that should be amortized
- The `setup()` method is called per-iteration by Criterion — keep it cheap (clone from shared state)
- The `run()` method is what Criterion measures — this is the hot path
- Use `std::hint::black_box()` on the result to prevent the compiler from optimizing away the computation

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

Create `jolt-eval/benches/<objective_name>.rs`:

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

<Examples>
<Good>
User: "/new-objective cyclomatic_complexity"
Action: Asks whether static or performance, creates the objective file, registers in all enums, adds objective function, runs tests.
Why good: Full pipeline — every enum, dispatch method, and test count is updated.
</Good>

<Bad>
User: "/new-objective my-objective"
Action: Accepts the name with a hyphen.
Why bad: Rust identifiers use underscores, not hyphens. Should reject and suggest `my_objective`.
</Bad>

<Bad>
User: "/new-objective bind_compact"
Action: Creates a performance objective but doesn't create the Criterion benchmark file.
Why bad: Performance objectives are measured via Criterion — without the bench file, `cargo bench` won't find it and the optimization harness can't measure it.
</Bad>
</Examples>

Task: Implement a new objective for jolt-eval. {{ARGUMENTS}}
