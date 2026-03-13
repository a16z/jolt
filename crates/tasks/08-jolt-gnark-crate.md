# Task 08: jolt-gnark crate

**Status:** Done
**Phase:** After task 05 (circuit backend)
**Dependencies:** `jolt-ir` circuit backend (task 05), PR #1322 (wonderland gnark transpiler)
**Blocks:** Production gnark verifier circuit

## Objective

Create a `jolt-gnark` crate that implements `jolt-ir::CircuitEmitter` for gnark/Go code generation. This ports the claim-formula codegen from PR #1322's `transpiler/src/gnark_codegen.rs` to use the `jolt-ir` expression IR instead of `MleAst`.

## Context

PR #1322 (defi-wonderland) builds a full symbolic transpilation pipeline:

1. **Symbolic execution**: Run the Rust verifier with `MleAst` (a symbolic `JoltField`) to record all operations
2. **AST capture**: Thread-local arena collects the full verifier trace as `Node` trees
3. **Code generation**: `GnarkCodeGen` converts the AST to gnark Go code with CSE

This task extracts step 3 for **claim-level formulas only**. The full verifier tracing (steps 1-2) remains in the transpiler crate. `jolt-gnark` provides the `CircuitEmitter` implementation so claim formulas defined via `jolt-ir::Expr` can be mechanically translated to gnark circuit code.

## Deliverables

### `crates/jolt-gnark/`

A new crate with a single primary type:

```rust
/// Gnark Go code emitter for jolt-ir expressions.
///
/// Implements `CircuitEmitter` to produce gnark API calls:
/// - Constants → `big.NewInt(val)` or inline literal
/// - Variables → `circuit.Opening_N` / `circuit.Challenge_N`
/// - Arithmetic → `api.Add(l, r)`, `api.Sub(l, r)`, `api.Mul(l, r)`, `api.Neg(x)`
pub struct GnarkEmitter {
    /// Maps opening indices to Go field names (e.g., 0 → "circuit.Stage1_H")
    pub opening_names: BTreeMap<u32, String>,
    /// Maps challenge indices to Go field names
    pub challenge_names: BTreeMap<u32, String>,
    /// CSE let-binding prefix (e.g., "cse_0" for constraint 0)
    pub cse_prefix: String,
}

impl CircuitEmitter for GnarkEmitter {
    type Wire = String; // Go expression text
    // ...
}
```

Additional utilities ported from `transpiler/src/gnark_codegen.rs`:
- `sanitize_go_name()` — convert Rust identifiers to valid Go identifiers
- `generate_assertion()` — wrap an expression in `api.AssertIsEqual(expr, 0)`
- CSE-aware emission using `Expr::to_circuit()` (cached traversal)

### What is NOT in scope

- Full verifier symbolic execution (`MleAst`, symbolic transcript, etc.) — stays in transpiler
- Poseidon circuit integration — stays in transpiler
- Go test harness / witness generation — stays in transpiler
- `AstBundle` / per-constraint isolation — handled by `jolt-ir`'s CSE + `to_circuit()`

## Architecture

```
jolt-ir (expression IR)
  └── CircuitEmitter trait
       └── jolt-gnark::GnarkEmitter (Go code strings)

transpiler (full verifier tracing, PR #1322)
  ├── symbolic execution (MleAst as JoltField)
  ├── verifier stage capture
  └── uses jolt-gnark for claim-formula codegen (future)
```

The transpiler currently does everything end-to-end with `MleAst`. Once `jolt-gnark` exists, claim-formula codegen can be migrated to use `Expr::to_circuit(&mut GnarkEmitter)` instead of the `GnarkCodeGen::generate_expr()` path for claim-level expressions.

## Testing

- Known expressions produce expected Go output (string comparison)
- Generated code uses correct gnark API calls (`api.Add`, `api.Mul`, etc.)
- CSE: shared subexpressions produce intermediate variable assignments
- Round-trip: for each test expression, verify the gnark output would compute the same value as `Expr::evaluate()`
- Variable naming: custom opening/challenge name maps produce correct field references

## Reference

- PR #1322 `transpiler/src/gnark_codegen.rs` — existing gnark codegen (1018 lines)
- PR #1322 `transpiler/src/gnark_codegen.rs:170-280` — `GnarkCodeGen` struct and `generate_expr()`
- `zklean-extractor/src/mle_ast.rs:86-108` — `Node` enum (maps to `jolt-ir::ExprNode`)
- `jolt-ir/src/backends/circuit.rs` — `CircuitEmitter` trait
