# Task 11: Split jolt-core Module Builder Into Readable Sections

## Status: TODO

## Context
The jolt-core module (`jolt-core/examples/jolt_core_module.rs` or wherever the hand-written Module reference lives) encodes the entire Jolt protocol as a single `Module` using `ModuleBuilder`. This serves as the canonical reference implementation that the compiler will eventually optimize from.

Currently it's monolithic — all stages in one function — making it hard to read, review, and extend for stage 6+.

## Fix
Split the module construction into per-stage functions:

```rust
fn build_jolt_module(config: &JoltConfig) -> Module {
    let mut builder = ModuleBuilder::new(config);
    
    stage1_spartan_outer(&mut builder);
    stage2_product_virtualization(&mut builder);
    stage3_instruction_lookups(&mut builder);
    stage4_bytecode_raf(&mut builder);
    stage5_ram_read_write(&mut builder);
    // stage6_claim_reductions(&mut builder);  // future
    
    builder.finish()
}

fn stage1_spartan_outer(b: &mut ModuleBuilder) {
    // Outer sumcheck + univariate skip
    // Clear comments explaining what this stage proves
}
```

Each function should have a brief comment explaining:
1. What claim(s) this stage proves
2. What upstream data it consumes (which evals/challenges from prior stages)
3. What it produces for downstream stages

## Benefits
- Each stage is independently reviewable
- Adding stage 6 is just adding a new function
- Makes the compiler's "reference protocol" self-documenting
- Easier to cross-reference with jolt-core's prover.rs

## Test
```bash
cargo nextest run -p jolt-equivalence transcript_divergence --cargo-quiet
```
Must remain byte-identical — this is a pure refactor of the builder call site.

## Risk: None
Pure code organization change.

## Dependencies: None (can be done in parallel with anything)
