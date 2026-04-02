# Invariants and Objectives

I want to introduce a Rust framework that gives some explicit structure to the "evaluation" part of the intent-execution-evaluation model described [here](https://gist.github.com/moodlezoup/e9f95839d9d848938eb54b662c6c5d25). The motivation is twofold:
1. Maximize agent productivity
2. Minimize the human verification surface

"Evaluation" should be further broken down into **invariants** and **objectives**.

**Invariants** are evaluations with a binary outcome, i.e. things that we want to always hold:
- All the tests pass
- No linter warnings/errors
- No unused dependencies

**Objectives** are evaluations with a numerical outcome, i.e. things we may want to optimize for:
- Peak memory usage
- Runtime
- Code coverage
- Some subjective score of code quality, as judged by AI

Note that by definition, invariants are a special case of objectives, but it's useful to think of them as separate categories.

The key property for both invariants are objectives is that they must be **mechanically checkable**. This is important for both of our motivations: it increases agent productivity, by giving the agent a way to check its work without a human in the loop; and it allows the human to gain assurance about the larger codebase while only focusing on a smaller kernel of invariants/objectives. 

## Invariants

Given a single invariant description (a small amount of Rust encoding the invariant), we should be able to mechanically synthesize it into:
- A test,
- A `libfuzzer_sys` fuzz target,
- And/or a "red team" harness for AI agents to try to find a violation of the invariant
	- Assuming the invariant and harness are well-written, this should totally eliminate the possibility of false positives
	- Should be flexible with respect to the agent setup (which model, how many agents, guiding prompt, etc.)

In the long-term we should also be able to formally verify certain invariants.

The invariant description should specify which of the above to generate. The "regular" tests generated from invariants should be run in CI. Fuzzing and AI-driven security reviews can be run at a less-frequent cadence or ad-hoc.

Pseudocode for an Invariant trait:
```rust
 trait Invariant: Send + Sync {
    type Setup;
    type Input: Arbitrary;

    fn name(&self) -> &str;
    /// Used as context for an AI agent trying to violate this invariant
    fn description(&self) -> String;
    /// What to synthesize from this invariant
    fn targets(&self) -> EnumSet<SynthesisTarget> // ⊆ {Test, Fuzz, RedTeam}
    fn setup(&self) -> Self::Setup;
    fn check(&self, setup: &Self::Setup, input: Self::Input) -> Result<(), InvariantViolation>;
    /// Returns a seed corpus for tests/fuzzing (known-interesting inputs)
    fn seed_corpus(&self) -> Vec<Self::Input> {
        vec![]
    }
}

```

Pseudocode for the AI "red team" harness:
```rust
fn auto_redteam(invariant: Invariant, prompt: String) {
    for _ in NUM_ITERATIONS {
        // Note: AI should run in an isolated worktree to produce the 
        // claimed bad input. The invariant is checked in the original
        // working tree so the AI cannot cheat.
        if let Some(bad_input) = todo!("Tell Claude to find violation of invariant") {
            if let Err(e) = invariant.check(bad_input) {
                todo!("Log counterexample and error");
                todo!("Tell Claude to summarize how it found the violation");
                break;
            }
        } else {
            todo!("Clean up the worktree, cache description of failed attempt")
        }
    }
}

struct InvariantCounterexample<I: Invariant> {
    description: String,
    input: I::Input,
    error: InvariantViolation,
}

struct FailedAttempt {
    description: String,
    approach: String,       // What the agent tried
    failure_reason: String, // Why it didn't produce a valid counterexample
}
```

## Objectives

The top-level interface for working with objectives should look something like:
```rust
fn measure_objectives(objectives: Vec<Objective>) -> HashMap<Objective, f64>;
```
The function would iterate through the provided objectives, dispatch to their respective `collect_measurement` methods.

Psuedocode for an Objective trait
```rust
trait AbstractObjective: Send + Sync {
    fn name(&self) -> &str;
    fn collect_measurement(&self) -> Result<f64, MeasurementError>;
    /// How many samples to take for statistical significance
    fn recommended_samples(&self) -> usize { 1 }
    /// What threshold is considered a regression, e.g., 5% slowdown
    fn regression_threshold(&self) -> Option<f64> { None }
    /// Is lower better or higher better?
    fn direction(&self) -> Direction; // Minimize or Maximize
}
```

Objectives can be used as building blocks for expressive, AI-driven optimization tasks (cf. [autoresearch](https://github.com/karpathy/autoresearch)). 
Pseudocode for a simple optimization harness:
```rust
fn auto_optimize<F: Fn(HashMap<Objective, f64>) -> f64>(objectives: Vec<Objective>, objective_function: F, prompt: String) {
    let mut baseline = objective_function(measure_objectives(objectives));
    for _ in NUM_ITERATIONS {
       	todo!("Tell Claude Code to optimize for the given objective function");
       	// Can also point Claude to specific functions/snippets to optimize
       	let new_score = objective_function(measure_objectives(objectives));
       	let invariants_hold = check_invariants();
       	if invariants_hold && new_score > baseline {
            // Successful optimization
            baseline = new_score;
            todo!("Commit changes for async human review");
       	} else {
       	    todo!("Revert changes, cache description of the failed attempt");
       	}
    } 
}

pub enum Objective {
    PeakRss(PeakRssObjective),
    ProverTime(ProverTimeObjective),
    ProofSize(ProofSizeObjective),
    VerifierTime(VerifierTimeObjective),
    GuestCycleCount(GuestCycleCountObjective),
    // ...
}

impl Objective {
    pub fn collect_measurement(&self) -> Measurement {
        match self {
           	Self::PeakRss(o) => o.collect_measurement(),
           	Self::ProverTime(o) => o.collect_measurement(),
           	// ...
        }
    }
}

struct OptimizationAttempt<I: Invariant> {
    description: String,       // What the agent tried
    diff: String,              // The actual code change
    measurements: HashMap<Objective, f64>,
    invariants_passed: bool
}
```

Objectives are ideally reproducible, deterministic, and quick to obtain, though none of these are hard rules –– in particular, performance metrics like runtime inevitably have some variance and may be slow to obtain. 
## Framing tasks in terms of invariants and objectives

### Implementing a new feature
- Add new invariants to capture the behavior of the feature
- Modify existing invariants as necessary
- The spec for a new feature should clearly document new and modified invariants, as well as expected impact on objectives
	- Impact on objectives can be mechanically validated
- Ensure that all invariants hold
### Bug fix
- Add a new invariant (or modify existing one) to fail without the fix
- Ensure that all other invariants still hold
- Document impact on objectives
### Security review
- Try to find a counterexample to some invariant
### Optimization
- For some function $f(o_1, o_2, \dots, o_n)$ that takes as input the objectives and outputs a single score, maximize the score
- Can apply techniques from multi-objective optimization literature
- Ensure that all invariants still hold
### Refactor
- Special case of optimization, where the objective function captures some notion of code quality

## As applied to Jolt

### Example invariants

- **Soundness**: For a fixed program, input, and honest prover output/proof, the verifier does not accept for any other output/proof. 
- **(Verifier) Completeness**: For a fixed program, input, and honest prover output/proof, the verifier accepts the honest output/proof. 
- **(Prover) Completness**: For a fixed program, input, and valid size parameters for that program/input pair, the prover should produce a proof (or OOM/timeout). 
- **Determinism**: Same program + input → same proof (byte-identical).
- **Serialization roundtrip**: `deserialize(serialize(proof)) == proof`
### Example objectives

- Peak RSS (prover memory)
- Prover time
- Proof size
- Verifier time
- Guest cycle counts
- Virtual/inline sequence lengths
- Wrapping cost (Transpiled verifier constraint count)
  
### Crate structure

```
jolt-eval/
  ├── Cargo.toml
  ├── src/
  │   ├── lib.rs                          # Re-exports, top-level check/measure fns
  │   │
  │   ├── invariant/
  │   │   ├── mod.rs                      # Invariant trait, InvariantViolation, SynthesisTarget,
  │   │   │                               #   FailedAttempt, centralized Invariant enum
  │   │   ├── soundness.rs                # Soundness invariant (proof mutation)
  │   │   ├── completeness_verifier.rs    # Verifier completeness (honest proof accepted)
  │   │   ├── completeness_prover.rs      # Prover completeness (prover doesn't panic)
  │   │   ├── determinism.rs              # Same input → same proof
  │   │   ├── serialization_roundtrip.rs  # serialize(deserialize(x)) == x
  │   │   ├── zk_consistency.rs           # host and host,zk both produce valid proofs
  │   │   └── synthesis/
  │   │       ├── mod.rs                  # Synthesis registry, shared types
  │   │       ├── test.rs                 # #[test] generation from invariants
  │   │       ├── fuzz.rs                 # libfuzzer_sys target generation
  │   │       └── redteam.rs              # auto_redteam loop, worktree orchestration,
  │   │                                   #   InvariantCounterexample, prompt construction
  │   │
  │   └── objective/
  │       ├── mod.rs                      # AbstractObjective trait, Measurement, Unit, Direction,
  │       │                               #   centralized Objective enum, measure_objectives()
  │       ├── peak_rss.rs                 # Peak resident set size
  │       ├── prover_time.rs              # Wall-clock prover time
  │       ├── proof_size.rs               # Serialized proof byte length
  │       ├── verifier_time.rs            # Wall-clock verifier time
  │       ├── guest_cycles.rs             # Guest instruction cycle count
  │       ├── inline_lengths.rs           # Virtual/inline sequence lengths
  │       ├── wrapping_cost.rs            # Transpiled verifier constraint count
  │       └── optimize.rs                 # auto_optimize loop, OptimizationAttempt,
  │                                       #   baseline tracking, commit/revert logic
  │    
  |
  ├── macros/
  │   ├── Cargo.toml                      # jolt-eval-macros proc-macro crate
  │   └── src/
  │       └── lib.rs                      # #[invariant(targets = [...])] attribute macro
  │
  ├── bin/
  │   ├── check_invariants.rs             # CLI: run all or selected invariants
  │   ├── measure_objectives.rs           # CLI: measure all or selected objectives, compare to baseline
  │   └── redteam.rs                      # CLI: --invariant <name> --iterations N --model <model>
  │
  └── tests/
      └── integration.rs                  # Smoke tests for the framework itself
```
