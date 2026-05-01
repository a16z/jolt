---
name: new-invariant
description: Implement a new invariant for jolt-eval
argument-hint: "<invariant-name>"
---

<Purpose>
Implement a new invariant for the jolt-eval framework. An invariant is a property that must always hold — the framework can test it with seed inputs, fuzz it with random/structured inputs, and red-team it with an AI agent.

This skill handles all the boilerplate: creating the invariant struct + input type, implementing the `Invariant` trait, registering it in the `JoltInvariants` enum, creating a fuzz target (if applicable), and running `sync_targets.sh`.
</Purpose>

<Execution_Policy>
- The user must provide an invariant name (lowercase with underscores, e.g. `sumcheck_binding`).
- Ask the user what property is being checked and what the input type should look like before writing code.
- Follow existing patterns exactly — study the split_eq_bind and soundness invariants as models.
- Always run clippy and the auto-generated tests before reporting success.
</Execution_Policy>

<Steps>

## Phase 1: Gather Requirements

1. Validate the argument `{{ARGUMENTS}}`: must be a valid Rust identifier (lowercase alphanumeric + underscores). Reject otherwise.
2. Ask the user:
   - What property does this invariant check? (becomes the `description()`)
   - What does the input look like? (fields, types, ranges)
   - What synthesis targets should it support? (`Test`, `Fuzz`, `RedTeam`)
   - Does it need non-trivial setup? (e.g. preprocessing, compilation — default to `Setup = ()`)

## Phase 2: Explore Context

1. Read `jolt-eval/src/invariant/mod.rs` to understand the current `JoltInvariants` enum and `dispatch!` macro.
2. Read an existing invariant for reference:
   - Simple: `jolt-eval/src/invariant/split_eq_bind.rs`
   - Complex (with setup, enrich_input): `jolt-eval/src/invariant/soundness.rs`
3. If the invariant tests jolt-core functionality, explore the relevant jolt-core modules to understand the types and APIs involved.

## Phase 3: Implement

Create the invariant file at `jolt-eval/src/invariant/<invariant_name>.rs` with:

### Input Type

```rust
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct <Name>Input {
    // fields
}

impl<'a> Arbitrary<'a> for <Name>Input {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        // Generate random inputs with reasonable bounds
    }
}
```

Key requirements for the input type:
- Must derive `Debug`, `Clone`, `Serialize`, `Deserialize`, `JsonSchema`
- Must implement `Arbitrary` manually (for fuzzing)
- Use bounded ranges in `Arbitrary` impl (e.g. `u.int_in_range(2..=16)?`) to keep inputs meaningful

### Invariant Struct

```rust
#[jolt_eval_macros::invariant(Test, Fuzz, RedTeam)]  // adjust targets as needed
#[derive(Default)]
pub struct <Name>Invariant;

impl Invariant for <Name>Invariant {
    type Setup = ();  // or a custom setup type
    type Input = <Name>Input;

    fn name(&self) -> &str { "<invariant_name>" }
    fn description(&self) -> String { "...".into() }
    fn setup(&self) -> Self::Setup { /* ... */ }
    fn check(&self, setup: &Self::Setup, input: Self::Input) -> Result<(), CheckError> {
        // 1. Validate input — return Err(CheckError::InvalidInput(...)) for degenerate cases
        // 2. Run the property check
        // 3. Return Ok(()) if the invariant holds
        // 4. Return Err(CheckError::Violation(...)) if violated
    }
    fn seed_corpus(&self) -> Vec<Self::Input> {
        // Include: minimal case, typical case, edge case (large values, boundary conditions)
    }
}
```

### Guidelines for `check()`
- Use `CheckError::InvalidInput` for degenerate inputs that should be skipped (not counted as violations)
- Use `CheckError::Violation(InvariantViolation::with_details(...))` for actual violations — include diagnostic info
- Compare against a known-correct reference implementation when testing optimized code

## Phase 4: Register

Edit `jolt-eval/src/invariant/mod.rs`:

1. Add `pub mod <invariant_name>;` to the module declarations at the top.
2. Add a variant to `JoltInvariants`:
   ```rust
   <VariantName>(<invariant_name>::<Name>Invariant),
   ```
3. Add the variant to `JoltInvariants::all()`:
   ```rust
   Self::<VariantName>(<invariant_name>::<Name>Invariant),
   ```
   Use `<Name>Invariant::default()` if the struct has fields.
4. Add the variant to the `dispatch!` macro:
   ```rust
   JoltInvariants::<VariantName>($inv) => $body,
   ```

## Phase 5: Create Fuzz Target (if targets include Fuzz)

Create `jolt-eval/fuzz/fuzz_targets/<invariant_name>.rs`:

```rust
#![no_main]
use jolt_eval::invariant::<invariant_name>::<Name>Invariant;
jolt_eval::fuzz_invariant!(<Name>Invariant::default());
```

Then run `./jolt-eval/sync_targets.sh` to update `fuzz/Cargo.toml`.

## Phase 6: Validate

Run these commands (all must pass):

```bash
# Format
cargo fmt -q

# Lint
cargo clippy -p jolt-eval -q --all-targets -- -D warnings

# Run auto-generated tests (seed_corpus + random_inputs)
cargo nextest run -p jolt-eval --cargo-quiet invariant::<invariant_name>

# If fuzz target was created, verify it compiles
cd jolt-eval/fuzz && cargo check 2>&1 | head -20
```

If any step fails, fix the issue and re-run.

</Steps>

<Examples>
<Good>
User: "/new-invariant sumcheck_eval_consistency"
Action: Asks what property is being checked, creates the invariant file with proper Input type, registers it, creates fuzz target, runs tests.
Why good: Follows the full pipeline, tests pass before reporting success.
</Good>

<Bad>
User: "/new-invariant sumcheck_eval_consistency"
Action: Creates the invariant file but forgets to add the variant to the `dispatch!` macro.
Why bad: Code won't compile — the dispatch macro must match all enum variants.
</Bad>

<Bad>
User: "/new-invariant my-invariant"
Action: Accepts the name with a hyphen.
Why bad: Rust identifiers use underscores, not hyphens. Should reject and suggest `my_invariant`.
</Bad>
</Examples>

Task: Implement a new invariant for jolt-eval. {{ARGUMENTS}}
