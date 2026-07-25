# Performance Objective Template

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

**Guidelines:**
- Use `thread_local!` with a `Shared` struct for expensive setup (random data generation, etc.) that should be amortized
- The `setup()` method is called per-iteration by Criterion — keep it cheap (clone from shared state)
- The `run()` method is what Criterion measures — this is the hot path
- Use `std::hint::black_box()` on the result to prevent the compiler from optimizing away the computation
