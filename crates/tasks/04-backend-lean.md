# Task 04: backend-lean

**Status:** Not started
**Phase:** After task 01
**Dependencies:** `jolt-ir` core (task 01)
**Blocks:** `zklean-extractor` migration

## Objective

Implement the **Lean4 backend**: given an `Expr`, emit Lean4 syntax representing the expression as a function over opening and challenge variables.

This replaces the `MleAst` → string formatting pipeline in `zklean-extractor/src/mle_ast.rs` for claim-level expressions. Full verifier tracing (capturing orchestration logic) remains separate.

## Deliverables

### `src/backends/lean.rs`

```rust
pub struct LeanConfig {
    /// Prefix for let-bound variables (CSE output).
    pub let_prefix: String,
    /// Name for opening variables (e.g., "opening").
    pub opening_name: String,
    /// Name for challenge variables (e.g., "challenge").
    pub challenge_name: String,
    /// CSE depth threshold: subtrees deeper than this get let-bound.
    pub cse_threshold: usize,
}

impl Default for LeanConfig { /* sensible defaults */ }

impl Expr {
    pub fn to_lean4(&self, config: &LeanConfig) -> String;
}
```

The emitter:
1. Optionally runs CSE (`eliminate_common_subexpressions`)
2. Walks the DAG, emitting `let` bindings for shared subexpressions
3. Emits the root expression as a Lean4 term

Output example for `γ * (h² - h)`:
```lean
let x0 := opening_0 * opening_0
let x1 := x0 - opening_0
challenge_0 * x1
```

## Testing

- Known expressions produce expected Lean4 output (string comparison)
- Output is syntactically valid (basic bracket matching, no dangling refs)
- CSE: expressions with shared subtrees produce `let` bindings
- Single-node expressions (just a constant, just a variable) produce clean output

## Reference

- `zklean-extractor/src/mle_ast.rs:180-198` — existing Lean4 formatting logic
- `zklean-extractor/src/sumchecks.rs` — how claim expressions are extracted for ZkLean
