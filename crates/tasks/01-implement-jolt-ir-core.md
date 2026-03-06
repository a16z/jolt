# Task 01: implement-jolt-ir-core

**Status:** Done
**Phase:** 1a (parallel with jolt-poly, jolt-instructions)
**Dependencies:** `jolt-field` (done)
**Blocks:** All backend tasks, then `jolt-zkvm`

## Objective

Implement the core `jolt-ir` crate: the expression IR, builder API, claim definition types, visitor trait, and normalization passes. **No backends** — this task is about getting the IR itself right.

The IR is the **single source of truth** for all sumcheck claim formulas. Every backend (evaluation, BlindFold R1CS, Lean4, circuit transpilation) will consume it. Getting this right is prerequisite to everything else.

See spec §4.10 and RFC finding (12).

## Problem statement

Every sumcheck in Jolt has a mathematical claim formula (e.g., booleanity: $\gamma^i \cdot (H^2 - H)$). Today this formula is written 4 times in 4 formats (see RFC finding 12):

1. `SumcheckInstanceParams::input_claim()` — imperative Rust
2. `output_claim_constraint()` — BlindFold `ProductTerm` / `ValueSource`
3. `MleAst` — symbolic field with global arena
4. `ClaimExpr<F>` — expression tree for `SumcheckFrontend`

These must stay in sync. `jolt-ir` replaces all four with one definition.

## Deliverables

### 1. Expression types (`src/expr.rs`)

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Var {
    Opening(u32),
    Challenge(u32),
}

#[derive(Debug, Clone, Copy)]
pub enum ExprNode {
    Constant(i128),
    Var(Var),
    Neg(ExprId),
    Add(ExprId, ExprId),
    Mul(ExprId, ExprId),
    Sub(ExprId, ExprId),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExprId(u32);

pub struct ExprArena {
    nodes: Vec<ExprNode>,
}

pub struct Expr {
    arena: ExprArena,
    root: ExprId,
}
```

Design decisions:
- **No `Div`/`Inv`** — claim expressions never need division.
- **Constants are `i128`** — covers all practical field constants without making the IR generic over `F`.
- **Instance-local arena** — no global state (unlike MleAst's `static OnceLock<RwLock<Vec<Node>>>`).
- **`ExprId(u32)`** — small, Copy, hashable. 4B per reference vs 8B for a pointer.

### 2. Builder API (`src/builder.rs`)

```rust
pub struct ExprBuilder {
    arena: ExprArena,
}

pub struct ExprHandle<'a> {
    builder: &'a ExprBuilder,
    id: ExprId,
}

impl ExprBuilder {
    pub fn new() -> Self;
    pub fn opening(&self, id: u32) -> ExprHandle<'_>;
    pub fn challenge(&self, id: u32) -> ExprHandle<'_>;
    pub fn constant(&self, val: i128) -> ExprHandle<'_>;
    pub fn zero(&self) -> ExprHandle<'_>;
    pub fn one(&self) -> ExprHandle<'_>;
    pub fn build(self, root: ExprHandle<'_>) -> Expr;
}
```

`ExprHandle` implements `Add`, `Sub`, `Mul`, `Neg` so users write:

```rust
let mut b = ExprBuilder::new();
let h = b.opening(0);
let gamma = b.challenge(0);
let expr = b.build(gamma * (h * h - h));
```

Key design question: `ExprBuilder` must use interior mutability (`RefCell` or similar) because `opening()` etc. take `&self` but push to the arena. This allows multiple handles to coexist without borrow conflicts. Alternative: take `&mut self` and return `ExprId` directly (less ergonomic but simpler). **Decide during implementation — ergonomics matter here because developers will write ~20 claim definitions.**

### 3. Claim definition (`src/claim.rs`)

```rust
pub struct ClaimDefinition {
    pub expr: Expr,
    pub opening_bindings: Vec<OpeningBinding>,
    pub challenge_bindings: Vec<ChallengeBinding>,
}

pub struct OpeningBinding {
    pub var_id: u32,
    pub polynomial_tag: u64,
    pub sumcheck_tag: u64,
}

pub struct ChallengeBinding {
    pub var_id: u32,
    pub source: ChallengeSource,
}

pub enum ChallengeSource {
    BatchingCoefficient(usize),
    SumcheckChallenge(usize),
    Derived,
}
```

`ClaimDefinition` bundles the expression with metadata telling downstream code how to resolve the variables. Tags are opaque `u64` — `jolt-ir` never imports from `jolt-zkvm`.

### 4. Visitor trait (`src/visitor.rs`)

```rust
pub trait ExprVisitor {
    type Output;
    fn visit_constant(&mut self, val: i128) -> Self::Output;
    fn visit_var(&mut self, var: Var) -> Self::Output;
    fn visit_neg(&mut self, inner: Self::Output) -> Self::Output;
    fn visit_add(&mut self, lhs: Self::Output, rhs: Self::Output) -> Self::Output;
    fn visit_sub(&mut self, lhs: Self::Output, rhs: Self::Output) -> Self::Output;
    fn visit_mul(&mut self, lhs: Self::Output, rhs: Self::Output) -> Self::Output;
}

impl Expr {
    pub fn visit<V: ExprVisitor>(&self, visitor: &mut V) -> V::Output;
}
```

Bottom-up traversal: visit children before parents. For DAGs (after CSE), each node is visited once; results are cached internally.

### 5. Normalization (`src/normalize.rs`)

```rust
pub struct SumOfProducts {
    pub terms: Vec<SopTerm>,
}

pub struct SopTerm {
    pub coefficient: SopValue,
    pub factors: Vec<SopValue>,
}

pub enum SopValue {
    Constant(i128),
    Opening(u32),
    Challenge(u32),
}

impl Expr {
    pub fn to_sum_of_products(&self) -> SumOfProducts;
    pub fn fold_constants(&self) -> Expr;
    pub fn eliminate_common_subexpressions(&self) -> Expr;
}
```

`to_sum_of_products()` is the key transformation — it mechanically distributes multiplication over addition:
- `(a + b) * c` → `[a*c, b*c]`
- `(a - b) * c` → `[a*c, (-1)*b*c]`
- `-(a * b)` → `[(-1)*a*b]`

This replaces the hand-written `OutputClaimConstraint` construction in every sumcheck. The SoP form maps directly to R1CS multiplication gates.

### 6. Crate root (`src/lib.rs`)

Re-exports: `Expr`, `ExprId`, `ExprArena`, `ExprNode`, `ExprBuilder`, `ExprHandle`, `Var`, `ClaimDefinition`, `OpeningBinding`, `ChallengeBinding`, `ChallengeSource`, `ExprVisitor`, `SumOfProducts`, `SopTerm`, `SopValue`.

## Testing (inline, `#[cfg(test)]`)

Every module gets unit tests:

- **expr.rs**: Arena construction, node insertion, ExprId stability
- **builder.rs**: Operator overloading produces correct DAGs, `build()` consumes builder
- **visitor.rs**: Visitor traversal order is bottom-up, visits each node exactly once
- **normalize.rs**: Distribution correctness for all operator combinations, constant folding, CSE dedup
- **claim.rs**: Construction, binding metadata round-trips

Property tests (using `proptest` or hand-rolled random):
- For random `Expr`: `to_sum_of_products()` evaluated with concrete values matches direct evaluation via visitor
- CSE: `eliminate_common_subexpressions()` produces smaller or equal arena, same evaluation

## Design constraints

- Depends **only** on `jolt-field` (for `Field` trait — used only by backends, but the crate needs the dep)
- Actually: this task has **zero backend code**. The `jolt-field` dependency may not even be needed yet. Consider making it optional or deferring it to backend tasks.
- No global mutable state
- No `Div`/`Inv` nodes
- Constants are `i128`, not generic `F`
- Instance-local arenas
- No files >500 lines, prefer <300
- ~800-1000 LOC target for this task (backends are separate)

## Future work (NOT in this task)

- **Kernel IR** (`SumcheckKernel`, `TableDescriptor`, `TableRole`) — deferred until GPU work begins. Will reuse `Expr` with different variable semantics.
- **All backends** — separate tasks (evaluate, R1CS, Lean4, circuit). Each consumes the IR via `ExprVisitor`.
- **TracingField** — `MleAst`-style symbolic field type. Separate crate (`jolt-ir-trace`).

## Reference files

- `jolt-core/src/subprotocols/blindfold/output_constraint.rs` — the BlindFold SoP IR this normalizes to
- `jolt-core/src/subprotocols/sumcheck_claim.rs:139-145` — `ClaimExpr` this replaces
- `zklean-extractor/src/mle_ast.rs` — MleAst arena (anti-pattern: global state)
- `jolt-core/src/subprotocols/booleanity.rs:87-161` — concrete dual-implementation example
- `jolt-core/src/zkvm/ram/read_write_checking.rs:67-228` — another dual-implementation example
- `jolt-core/src/subprotocols/sumcheck_verifier.rs:49-69` — `SumcheckInstanceParams` trait this feeds into
