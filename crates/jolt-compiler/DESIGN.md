# jolt-ir2: SNARK Protocol Compiler

Minimal compiler framework for specifying and lowering SNARK protocols. Three-level IR with progressive lowering, inspired by ML compilers (XLA/TVM).

## Core Insight

Every SNARK prover computation: `resolve inputs -> dispatch to device -> small result to host -> Fiat-Shamir -> propagate`. The compiler lowers a declarative protocol into a dispatch plan where each vertex is exactly one such roundtrip.

## Type Hierarchy

Four primitives, each building on the last:

```
Scalar / Poly   -- protocol variables, implement arithmetic ops
      |
      v
    Expr         -- sum-of-products, emerges from ops on Scalar/Poly
      |
      v
   Protocol      -- DAG of claims about Exprs
```

### Poly and Scalar

Handles into the protocol. Implement arithmetic ops to produce `Expr`.

```rust
#[derive(Clone, Copy)]
pub struct Poly(pub usize);

#[derive(Clone, Copy)]
pub struct Scalar(pub usize);

/// Committed = PCS-backed. Virtual = derived during proving.
/// Public = deterministic from challenges (eq, eq+1, lt).
pub enum PolyKind { Committed, Virtual, Public(PublicPoly) }
pub enum PublicPoly { Eq, EqPlusOne, Lt }
```

All three kinds are symbolically identical in expressions. Backends inspect `PolyKind` to decide materialization.

The protocol stores names, dimensions, and kinds in flat `Vec`s — no wrapper structs:

```rust
pub struct Protocol {
    pub dim_names: Vec<&'static str>,       // "log_T", "log_K", ...
    pub scalar_names: Vec<&'static str>,    // "gamma", "rho_a", ...
    pub poly_names: Vec<&'static str>,
    pub poly_dims: Vec<Vec<usize>>,         // poly index -> dim indices
    pub poly_kinds: Vec<PolyKind>,
    pub claims: Vec<Claim>,
    pub vertices: Vec<Vertex>,
}
```

Scalar values are **never** concrete in the IR — only at runtime (Fiat-Shamir outputs).

### Expr

Emerges from arithmetic on `Poly` and `Scalar`. Sum-of-products, built eagerly.

```rust
/// Sum-of-products expression.
/// Each term: (coefficient, [factors]).
/// A factor is either a poly or scalar reference.
pub struct Expr(pub Vec<(i128, Vec<Factor>)>);

#[derive(Clone, Copy, Hash, Eq, PartialEq)]
pub enum Factor {
    Poly(usize),
    Scalar(usize),
}
```

That's it. `Poly`, `Scalar`, `Factor`, `Expr`. Arithmetic ops on `Poly`/`Scalar`/`Expr`/`i128` produce `Expr`:

```rust
let eq = proto.poly("eq", &[log_T], Public(PublicPoly::Eq(None)));  // Poly(0)
let az = proto.poly("Az", &[log_T], Virtual);                 // Poly(1)
let bz = proto.poly("Bz", &[log_T], Virtual);                 // Poly(2)
let cz = proto.poly("Cz", &[log_T], Virtual);                 // Poly(3)
let gamma = proto.scalar("gamma");                             // Scalar(0)
let h = proto.poly("H", &[log_T], Virtual);                   // Poly(4)

// Public poly is just another factor in the expression
let outer    = eq * (az * bz - cz);        // Expr: [(1, [P(0), P(1), P(2)]), (-1, [P(0), P(3)])]
let boolcheck = eq * gamma * (h * h - h);  // Expr: [(1, [P(0), S(0), P(4), P(4)]), (-1, [P(0), S(0), P(4)])]
let weighted = rho_a * az + rho_b * bz;    // Expr: [(1, [S(1), P(1)]), (1, [S(2), P(2)])]
```

`Expr` is fully transparent — consumers iterate terms directly. Evaluation, pattern detection, and kernel dispatch live in the compiler and backends, not on the type.

### Claims

First-class. The only newtype ID in the system.

```rust
#[derive(Clone, Copy, Hash, Eq, PartialEq)]
pub struct ClaimId(pub u32);

/// A claim: "polynomial P evaluates to some value at a point
/// determined by the producing vertex."
pub struct Claim {
    pub id: ClaimId,
    pub poly: usize,          // index into Protocol.polynomials
    pub produced_by: usize,   // index into Protocol.vertices
}
```

## L0: Protocol

The mathematical specification. A DAG of vertices connected by claims. `Protocol` struct shown above.

Four vertex types:

```rust
pub enum Vertex {
    /// Prover commits to polynomials.
    Commit {
        polynomials: Vec<usize>,       // poly indices
    },
    /// Proves a polynomial identity via sumcheck.
    ///
    /// The composition Expr includes all polynomial factors — committed,
    /// virtual, and public. The backend inspects PolyKind to determine
    /// how to handle each factor (e.g., materializing eq from challenges).
    Sumcheck {
        composition: Expr,
        input: InputClaim,
        produces: Vec<ClaimId>,
        consumes: Vec<ClaimId>,
        /// Dimensions to bind, in order. Multi-phase = multiple dims.
        binding_order: Vec<usize>,     // dim indices
    },
    /// Batches claims via random linear combination.
    Rlc {
        consumes: Vec<ClaimId>,
        produces: ClaimId,
    },
    /// Terminal PCS evaluation proof.
    Opening {
        consumes: ClaimId,
    },
}

pub enum InputClaim {
    Zero,
    Constant(i64),
    /// Derived from upstream claim evaluations and scalars.
    Derived {
        expr: Expr,
        /// Maps scalar indices used in expr to the upstream claim
        /// whose evaluation they represent.
        claim_evals: Vec<(usize, ClaimId)>,  // (scalar_index, claim)
    },
}
```

### Protocol Builder

```rust
impl Protocol {
    pub fn new() -> Self;

    pub fn dim(&mut self, name: &'static str) -> usize;
    pub fn scalar(&mut self, name: &'static str) -> Scalar;
    pub fn poly(&mut self, name: &'static str, dims: &[usize], kind: PolyKind) -> Poly;

    pub fn commit(&mut self, polys: &[Poly]);
    pub fn sumcheck(&mut self, composition: Expr, input: InputClaim, binding_order: &[usize]) -> Vec<ClaimId>;
    pub fn rlc(&mut self, claims: &[ClaimId]) -> ClaimId;
    pub fn opening(&mut self, claim: ClaimId);
    pub fn open_all_committed(&mut self);
}
```

No separate builder — `Protocol` builds itself.

**~50-100 vertices for Jolt.**

## L1: Staged Protocol

Expands L0 into its Fiat-Shamir-bound constituents. Determines which vertices share a transcript interaction, how Fiat-Shamir challenges are derived, and how scalars resolve. Consumed by the **verifier**.

For v1, fully determined by hints. Future: LP solver.

```rust
pub struct StagedProtocol {
    pub protocol: Protocol,
    pub stages: Vec<Stage>,
    pub commitment_order: Vec<usize>,  // vertex indices
}

pub struct Stage {
    pub vertices: Vec<usize>,          // vertex indices (antichain)
    pub num_challenge_vars: usize,
    pub challenges: Vec<FiatShamirChallenge>,
    pub batches: Vec<Vec<usize>>,      // groups of vertex indices
    pub scalar_bindings: Vec<ScalarBinding>,
}

pub enum FiatShamirChallenge {
    Scalar { label: &'static str },
    GammaPowers { label: &'static str, count: usize },
}

pub struct ScalarBinding {
    pub scalar: usize,                 // scalar index
    pub source: ScalarSource,
}

pub enum ScalarSource {
    Challenge { label: &'static str },
    GammaPower { label: &'static str, exponent: usize },
    External { label: &'static str },
    ClaimEval(ClaimId),
}
```

Hints for v1:
```rust
pub struct StagingHints {
    pub stages: Vec<StageHint>,
    pub commitment_order: Vec<usize>,
}

pub struct StageHint {
    pub vertices: Vec<usize>,
    pub num_challenge_vars: usize,
    pub challenges: Vec<FiatShamirChallenge>,
    pub batches: Vec<Vec<usize>>,
    pub scalar_bindings: Vec<ScalarBinding>,
}
```

## L2: Execution Plan

Each vertex = one device dispatch + one host FS sync. A 20-round sumcheck = 20 ops. A 3-phase sumcheck = 3 kernel runs. Consumed by the **prover**.

All scalar values are symbolic — they reference outputs of prior ops, not concrete field elements. Concrete values exist only at runtime.

```rust
pub struct ExecutionPlan {
    pub ops: Vec<Op>,
    pub kernels: Vec<KernelDef>,
}

pub struct KernelDef {
    pub composition: Expr,
    pub source_vertex: usize,
}

pub enum Op {
    Dispatch       { kernel: usize, inputs: Vec<usize>, output: usize },
    Bind           { buffer: usize, challenge: usize },
    /// Materialize a public polynomial from challenge values.
    /// The compiler inspects PublicPoly to emit the right computation.
    Materialize    { kind: PublicPoly, point: Vec<usize>, output: usize },
    DeriveChallenge { label: &'static str, output: usize },
    Transcript     { source: usize },
    ExtractEval    { buffer: usize, output: usize },
    Upload         { source: HostData, output: usize },
    Free           { buffer: usize },
}

pub enum HostData {
    Witness(usize),               // poly index
    Precomputed(&'static str),
}
```

L2 `usize` fields are indices into the plan's buffer/scalar allocation tables. The plan tracks lifetimes.

**~500-2000 ops for Jolt.**

## Compiler

```rust
pub trait Pass {
    type Input;
    type Output;
    type Hints;
    fn run(input: &Self::Input, hints: &Self::Hints) -> Self::Output;
}
```

```
Protocol (L0)
    |
    +---> [R1CS backend]    Expr.emit_r1cs()
    |
    v  StagePass (hinted for v1, LP solver later)
StagedProtocol (L1)
    |
    +---> [Verifier]
    |
    v  LowerPass (hinted for v1)
ExecutionPlan (L2)
    |
    +---> [Prover]
```

## R1CS Backend

For BlindFold. Poly factors become witness variables, scalar factors bake into matrix coefficients.

```rust
pub struct R1csVar(pub u32);

pub struct R1csConstraint<F> {
    pub a: Vec<(R1csVar, F)>,
    pub b: Vec<(R1csVar, F)>,
    pub c: Vec<(R1csVar, F)>,
}

impl Expr {
    pub fn emit_r1cs<F: Field>(
        &self,
        poly_vars: &[R1csVar],        // poly index -> witness var
        scalar_vals: &[F],            // scalar index -> baked value
        next_var: &mut u32,
    ) -> (Vec<R1csConstraint<F>>, R1csVar);
}
```

## Jolt Protocol Definition

```rust
pub fn build_jolt_protocol() -> Protocol {
    let mut p = Protocol::new();
    use PolyKind::*;

    // Dimensions
    let log_T = p.dim("log_T");
    let log_K = p.dim("log_K");
    let log_M = p.dim("log_M");

    // Scalars
    let gamma = p.scalar("gamma");
    let rho_a = p.scalar("rho_a");
    let rho_b = p.scalar("rho_b");
    let rho_c = p.scalar("rho_c");

    // Public polynomials — materialized from challenges, no witness data
    let eq = p.poly("eq", &[log_T], Public(PublicPoly::Eq(None)));

    // Committed polynomials — PCS-backed
    let witness = p.poly("SpartanWitness", &[log_M], Committed);
    let ram_inc = p.poly("RamInc", &[log_T], Committed);

    // Virtual polynomials — derived during proving
    let az = p.poly("Az", &[log_T], Virtual);
    let bz = p.poly("Bz", &[log_T], Virtual);
    let cz = p.poly("Cz", &[log_T], Virtual);
    let h  = p.poly("H",  &[log_T], Virtual);

    p.commit(&[witness, ram_inc]);

    // Spartan outer: eq · (Az · Bz - Cz) = 0
    let outer = p.sumcheck(eq * (az * bz - cz), InputClaim::Zero, &[log_T]);

    // Booleanity: eq · gamma · (H² - H) = 0
    let _bool = p.sumcheck(eq * gamma * (h * h - h), InputClaim::Zero, &[log_T]);

    // ... remaining ~40 vertices

    p.open_all_committed();
    p
}
```

## Crate Structure

```
jolt-ir2/src/
    lib.rs          re-exports                                ~30 lines
    expr.rs         Poly, Scalar, Factor, Expr, ops, patterns ~300 lines
    protocol.rs     L0: Protocol, Vertex, Claim, Builder      ~300 lines
    staged.rs       L1: StagedProtocol, Stage, Hints          ~200 lines
    execution.rs    L2: ExecutionPlan, Op                      ~150 lines
    compiler.rs     Pass trait + stubs                         ~80 lines
    r1cs.rs         R1csVar, R1csConstraint, emit_r1cs         ~200 lines
    jolt/
      mod.rs                                                   ~10 lines
      protocol.rs   build_jolt_protocol()                      ~300 lines
      hints.rs      jolt_staging_hints()                       ~200 lines
      claims.rs     claim formula helpers                      ~400 lines
```

**~2,200 lines total** (vs jolt-ir's ~14,600).

## ID Summary

One newtype: `ClaimId(u32)`. Everything else is `usize` indices into Vecs owned by their parent struct.

| Thing | How it's identified | Why |
|-------|-------------------|-----|
| Claim | `ClaimId(u32)` | First-class, referenced across vertices |
| Polynomial | `usize` index + `Poly` handle | Index into `Protocol.polynomials` |
| Scalar | `usize` index + `Scalar` handle | Index into `Protocol.scalars` |
| Dimension | `usize` index | Index into `Protocol.dims` |
| Vertex | `usize` index | Index into `Protocol.vertices` |
| Stage | `usize` index | Index into `StagedProtocol.stages` |
| Buffer/Kernel/etc. | `usize` index | Index into `ExecutionPlan` tables |

## Dependencies

```toml
[dependencies]
jolt-field = { path = "../jolt-field", default-features = false }

[dev-dependencies]
jolt-field = { path = "../jolt-field", features = ["bn254"] }
rand_chacha = { workspace = true }
rand_core = { workspace = true }
```
