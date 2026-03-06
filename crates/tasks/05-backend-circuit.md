# Task 05: backend-circuit

**Status:** Not started
**Phase:** After task 01
**Dependencies:** `jolt-ir` core (task 01)
**Blocks:** gnark transpiler migration (PR #1322 follow-up)

## Objective

Implement the **circuit transpilation backend**: given an `Expr`, emit circuit constraints for recursive verification (gnark/Groth16 or a generic circuit IR).

This replaces the `MemoizedCodeGen` pipeline in PR [#1322](https://github.com/a16z/jolt/pull/1322)'s gnark transpiler for claim-level formulas. Full verifier tracing (capturing the entire verification pipeline) remains separate.

## Deliverables

### `src/backends/circuit.rs`

A trait-based approach so different circuit targets can be plugged in:

```rust
/// Backend-specific circuit emitter.
///
/// Implementations produce circuit constraints in their target format
/// (gnark Go code, bellman, plonky2, etc.).
pub trait CircuitEmitter {
    type Wire;
    fn constant(&mut self, val: i128) -> Self::Wire;
    fn variable(&mut self, var: Var) -> Self::Wire;
    fn neg(&mut self, inner: Self::Wire) -> Self::Wire;
    fn add(&mut self, lhs: Self::Wire, rhs: Self::Wire) -> Self::Wire;
    fn sub(&mut self, lhs: Self::Wire, rhs: Self::Wire) -> Self::Wire;
    fn mul(&mut self, lhs: Self::Wire, rhs: Self::Wire) -> Self::Wire;
}

impl Expr {
    pub fn to_circuit<E: CircuitEmitter>(&self, emitter: &mut E) -> E::Wire;
}
```

A concrete gnark emitter could live in the gnark transpiler crate (not in jolt-ir), implementing `CircuitEmitter` to produce Go code strings. jolt-ir just defines the trait and the traversal.

## Testing

- Mock emitter that records operations → verify traversal order and completeness
- Expression with shared subexpressions → verify each subexpr emitted once (if CSE applied)
- Empty / trivial expressions

## Notes

- This is likely the lowest-priority backend. Can be deferred until recursion work resumes.
- The gnark transpiler (PR #1322) also captures orchestration logic via MleAst tracing — that part is NOT replaced by this backend. This only covers claim-formula emission.

## Reference

- PR [#1322](https://github.com/a16z/jolt/pull/1322) — gnark transpiler architecture
- `zklean-extractor/src/mle_ast.rs` — MleAst CSE and code generation patterns
