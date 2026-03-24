# Kernel Shapes and Evaluator Architecture

This document describes how every sumcheck composition in the Jolt protocol maps
to one of two kernel shapes, and how those shapes relate to the three evaluator
implementations that execute them.

## Two Shapes

Every sumcheck formula in Jolt maps to exactly one of two `KernelShape` variants.
No `Custom` fallback exists.

### SumOfProducts

Weighted sum of monomial products over payload polynomial buffers.

```
kernel(t) = Σ_k  weight[k] · Π_{j ∈ terms[k]}  payload[j](t)
```

Evaluated on the **standard grid** `{0, 2, 3, ..., degree}` (skipping `t=1`, which
is derived from the sumcheck claimed sum). The weighting polynomial (eq, LT, eq+1)
is applied externally by the evaluator — it is NOT one of the payload inputs.

```rust
SumOfProducts {
    /// Each inner Vec lists payload input indices participating in that term.
    /// Example: [[0,1], [0,2]] means weight[0]·payload[0]·payload[1] + weight[1]·payload[0]·payload[2].
    /// An empty inner Vec is an empty product (= 1), contributing a constant term.
    /// Repeated indices encode powers: [0,0] means payload[0]².
    terms: Vec<Vec<usize>>,
    /// Number of distinct payload input buffers.
    num_inputs: usize,
}
```

**Inner degree** = max term arity. **Total degree** = inner degree + 1 (weighting poly).

Backend optimizations (not visible in the shape):
- **All-arity-≤1**: backend pre-combines `g = Σ wₖ · polyₖ` into a single buffer,
  reducing to a degree-2 `weighting · g` multiply (the "EqProduct" fast path).
- **Gruen split-eq**: when `KernelDescriptor.tensor_split` is `Some`, the evaluator
  factors eq into outer × inner tables and uses Gruen cubic recovery to avoid
  explicit eq evaluation at grid points.

### ToomCookProduct

Product of D linear interpolants across P groups, summed.

```
kernel(t_k) = Σ_{g=0}^{P-1}  Π_{j=0}^{D-1}  payload[g·D + j](t_k)
```

Evaluated on the **Toom-Cook grid** `{1, 2, ..., D-1, ∞}`. Eq is always factored
out via `TensorSplit` — it is never a payload input.

```rust
ToomCookProduct {
    num_inputs_per_product: usize,  // D
    num_products: usize,            // P
}
```

Uses balanced binary splitting with extrapolation for O(D log D) multiplications
per pair position (vs O(D²) on the standard grid). D-specialized eval_prod
routines (D ∈ {4, 8, 16, 32}) are compiled AOT by each backend.

## Separation of Concerns

Three independent axes govern sumcheck execution. Shape is only one of them.

```
┌─────────────────────────────────────────────────────────┐
│  KernelShape                                            │
│  WHAT formula:  SumOfProducts or ToomCookProduct        │
│  Lives in:      jolt-ir/src/kernel.rs                   │
│  Determined by: ClaimDefinition.compile_descriptor()    │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  KernelDescriptor metadata                              │
│  HOW eq is traversed:  tensor_split (flat vs factored)  │
│  Lives in:      jolt-ir/src/kernel.rs                   │
│  Orthogonal to shape — any shape can carry TensorSplit  │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  Evaluator                                              │
│  HOW the domain is iterated:  dense, sparse, streaming  │
│  Lives in:      jolt-zkvm/src/evaluators/               │
│  Chosen by:     vertex metadata (phases, variable_group)│
└─────────────────────────────────────────────────────────┘
```

The shape describes the formula for kernel compilation and verifier checking.
The evaluator decides how to iterate over the domain — dense pair-wise reduction,
sparse checkpoint iteration, or streaming sparse matrix multiplication.

## Three Evaluators

All implement `SumcheckCompute<F>` from `jolt-sumcheck`. The prover dispatches
to the right one based on `SumcheckVertex` metadata.

### KernelEvaluator\<F, B: ComputeBackend\>

**Location**: `jolt-zkvm/src/evaluators/kernel.rs`

Backend-agnostic dense evaluator. Takes pre-materialized polynomial buffers as
`B::Buffer<F>`, compiles the `KernelDescriptor` into a backend-specific kernel,
and evaluates it at every pair position via `pairwise_reduce`.

Handles both shapes:
- `SumOfProducts` → standard grid, optional Gruen via `tensor_split`
- `ToomCookProduct` → Toom-Cook grid, eq as external weight buffer

Used by **11 single-phase vertices** (claim reductions, booleanity, RAF evaluation,
bytecode, instruction input, product virtual remainder, etc.).

### SparseRWEvaluator\<F\>

**Location**: `jolt-zkvm/src/evaluators/sparse_rw.rs`

CPU-only sparse evaluator for RAM and register read-write checking. Exploits the
fact that the K×T polynomial matrix is sparse (only ~T non-zero entries out of
K·T positions).

Three-phase execution:
1. **CycleMajor** — bind cycle variables over sparse entries sorted by (cycle, address).
   Gruen split-eq over non-zero entries only. Pairs of rows merge on `bind()`.
2. **AddressMajor** — bind address variables over sparse entries sorted by (address, cycle).
   Checkpoint-inferred zeros: absence of an entry at cycle j means the value equals
   the last written value for that address (the checkpoint).
3. **Dense** — after phases 1-2 shrink the table, materialize the small remaining
   K'×T' matrix into dense buffers and evaluate inline.

NOT generic over `ComputeBackend` — phases 1-2 have irregular access patterns and
data-dependent control flow that don't map to GPU thread grids.

The formula is the same SumOfProducts across all three phases:
`(1+γ)·ra·val + γ·ra·inc` → `SoP { terms: [[0,1], [0,2]], num_inputs: 3 }`.

Used by **2 multi-phase vertices** (RAM RW checking, register RW checking).

### Spartan (internal)

**Location**: `jolt-spartan/`

Bespoke `SumcheckCompute` implementations for the Spartan IOP: outer sumcheck
(`eq · Az · Bz`) and product virtual sumcheck. Uses univariate skip for the first
round, streaming sparse matrix evaluation for R1CS, and cached Az/Bz binding for
linear-time remaining rounds.

Not part of `prove_from_graph` — Spartan stages are handled by `jolt-spartan`
directly, before the protocol graph's sumcheck stages begin.

## Complete Claim Mapping

### ToomCookProduct (3 claims)

Eq factored via TensorSplit. O(D log D) per pair.

| Claim | D | P | Evaluator |
|-------|---|---|-----------|
| `instruction_ra_virtual(n, m)` | m | n | KernelEvaluator |
| `ram_ra_virtual(d)` | d | 1 | KernelEvaluator |
| `bytecode_ra_virtual(d)` | d | 1 | KernelEvaluator |

### SumOfProducts, inner degree 2 (8 claims)

Total degree 3 with weighting. Evaluator applies eq/LT externally.

| Claim | Payload | Terms | Weights | Weighting | Evaluator |
|-------|---------|-------|---------|-----------|-----------|
| `hamming_booleanity()` | [H] | `[[0,0],[0]]` | [1, -1] | Eq | KernelEvaluator |
| `ram_read_write_checking()` | [ra, val, inc] | `[[0,1],[0,2]]` | [(1+γ), γ] | Eq | SparseRWEvaluator |
| `ram_val_check()` | [inc, wa] | `[[0,1]]` | [1] | Lt | KernelEvaluator |
| `registers_read_write_checking()` | [val, rs1, rs2, wa, inc] | `[[3,4],[3,0],[1,0],[2,0]]` | [1, 1, γ, γ²] | Eq | SparseRWEvaluator |
| `registers_val_evaluation()` | [inc, wa] | `[[0,1]]` | [1] | Lt | KernelEvaluator |
| `instruction_input()` | [ris2, rs2, rimm, imm, lrs1, rs1, lpc, pc] | `[[0,1],[2,3],[4,5],[6,7]]` | [1, 1, γ, γ] | Eq | KernelEvaluator |
| `product_virtual_remainder()` | [left, right, rd_nz, wl, jump, lookup, branch, noop] | `[[0,1],[2,3],[2,4],[5,6],[4],[4,7]]` | [1, γ, γ², γ³, γ⁴, -γ⁴] | Eq | KernelEvaluator |
| `ra_booleanity(n)` | [ra₀..ra_{n-1}] | `[[i,i],[i]]` × n | [cᵢ, -cᵢ] × n | Eq | KernelEvaluator |

### SumOfProducts, inner degree 1 (5 claims)

Total degree 2 with weighting. Backend pre-combines all payload polynomials into
a single buffer → degree-2 `weighting · g` (EqProduct fast path).

| Claim | Payload | Terms | Weights | Weighting | Evaluator |
|-------|---------|-------|---------|-----------|-----------|
| `shift()` | [upc, pc, is_virt, is_first, noop] | `[[0],[1],[2],[3],[4],[]]` | [c₀..c₅] | EqPlusOne | KernelEvaluator |
| `ram_output_check()` | [val_final] | `[[0],[]]` | [mask, -mask·val_io] | Eq | KernelEvaluator |
| `ram_val_check_input(n)` | [val_rw, val_final, adv₀..] | `[[0],[1],..[1+n],[]]` | [...] | Eq | KernelEvaluator |
| `ram_raf_evaluation()` | [ra] | `[[0]]` | [unmap] | Eq | KernelEvaluator |
| `bytecode_read_raf(n)` | [val₀..val_{n-1}] | `[[0],..[n-1]]` | [c₀..c_{n-1}] | Eq | KernelEvaluator |

## How prove_from_graph Dispatches

```rust
fn build_witness(sv: &SumcheckVertex, ...) -> Box<dyn SumcheckCompute<F>> {
    let has_address_phase = sv.phases.iter()
        .any(|p| p.variable_group == VariableGroup::Address);

    if has_address_phase {
        // Multi-phase sparse vertex (RAM/register RW)
        Box::new(SparseRWEvaluator::new(sparse_matrix, ...))
    } else {
        // Single-phase dense vertex — compile kernel from claim definition
        let (desc, weights) = claim.compile_descriptor(&challenges);
        let kernel = backend.compile(&desc);
        Box::new(KernelEvaluator::from_descriptor(&desc, inputs, kernel, backend))
    }
}
```

Both return `Box<dyn SumcheckCompute<F>>` — the prover's stage loop is uniform
regardless of which evaluator sits behind the trait object.
