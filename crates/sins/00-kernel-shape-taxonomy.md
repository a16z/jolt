# Sin 0: KernelShape Conflates "What" With "How"

## The Problem

`KernelShape` has 4 variants:

```rust
enum KernelShape {
    ProductSum { num_inputs_per_product, num_products },
    EqProduct,
    HammingBooleanity,
    Custom { expr, num_inputs },
}
```

These are not 4 different computations. They're 1 computation (sum-of-products) with 3
hand-tuned special cases. `EqProduct` is `Custom { expr: a*b }` with a fast kernel.
`HammingBooleanity` is `Custom { expr: a*b*(b-1) }` with a fast kernel. The frontend
(jolt-ir) is making optimization decisions that belong to the backend (jolt-cpu/jolt-metal).

This forces a 4-arm match in `build_witness()` that mixes buffer preparation logic with
kernel dispatch — each arm does different things with eq, weights, and polynomial tables.

## What the Inventory Shows

Every sumcheck composition in jolt-core (19 instances, ~5000 lines) falls into exactly
two algorithmic categories:

### Algorithm 1: Dense Weighted Reduce

```
round_poly(X) = Σ_i weight[i] · kernel(inputs_at_pair_i, X)
```

Every dense kernel is a **sum of products**:

| Sumcheck Instance           | Formula (kernel only, eq factored out)           | Degree |
|-----------------------------|--------------------------------------------------|--------|
| RA virtual (instruction)    | `Π_{j=0}^{D-1} ra_j`                            | D      |
| RA virtual (RAM, bytecode)  | `Σ_t γ^t · Π_j ra_{t·m+j}`                      | m      |
| RAM read-write              | `ra · ((1+γ)·val + γ·inc)`                       | 2      |
| Hamming booleanity          | `h · (h - 1)`                                    | 2      |
| RA booleanity               | `Σ_i γ^i · (ra_i² - ra_i)`                      | 2      |
| Claim reductions            | `Σ_i c_i · poly_i`  (linear)                     | 1      |
| RAF evaluation              | `ra · unmap`                                     | 1      |
| Output check                | `io_mask · (val_final - val_io)`                  | 2      |
| Spartan outer               | `A·B - C`                                         | 2      |
| Product virtual             | `left·right` + batched corrections                | 2      |
| Shift                       | `Σ c_i · poly_i` (linear)                        | 1      |
| Instruction input           | `(ris2·rs2 + rimm·imm) + γ·(lrs1·rs1 + lpc·pc)` | 2      |

**Every single one** is expressible as `Σ_t coeff_t · Π_j input_{t,j}`.

### Algorithm 2: Sparse Entry-List Reduce

```
round_poly(X) = Σ_entry formula(entry_values, X)
```

Used by:
- RAM read-write phase 1-2 (SparseRwEvaluator — walks sorted entries)
- Register read-write phase 1-2

Sparse entries are (position, values) tuples walked in sorted order, merged/split
on bind. Not parallelizable in the same way as dense reduce.

### That's It

There are no other algorithms. Every sumcheck in Jolt is one of these two. The 4
`KernelShape` variants, the `InterpolationMode` enum, the hand-coded
EqProduct/HammingBooleanity kernels — all of that is optimization within Algorithm 1.

## Proposed Solution: Two Shapes, Backend Chooses Strategy

### New IR Types (jolt-ir frontend)

```rust
/// What the kernel computes. Pure description, no optimization hints.
pub struct CompositionFormula {
    /// The kernel as a normalized sum-of-products.
    /// Each term: coefficient × product of input references.
    pub terms: Vec<ProductTerm>,
    /// Number of distinct input polynomial buffers.
    pub num_inputs: usize,
}

pub struct ProductTerm {
    /// Structural coefficient (small integer, field-agnostic).
    pub coeff: i128,
    /// Challenge variable indices that multiply this term.
    /// Runtime values supplied at kernel compile time.
    pub challenge_indices: Vec<u32>,
    /// Input buffer indices in the product. May repeat (e.g., h·h for h²).
    pub input_indices: Vec<usize>,
}
```

`CompositionFormula` replaces `KernelShape`. It describes WHAT to compute with no
opinion about HOW.

### New KernelDescriptor

```rust
pub struct KernelDescriptor {
    pub formula: CompositionFormula,
    pub degree: usize,
    pub eq_handling: EqHandling,   // Flat | TensorSplit
    pub binding_order: BindingOrder,
}
```

Binding order and eq handling are now part of the descriptor (they were scattered
across the orchestration layer before).

### Backend Pattern Recognition (jolt-cpu, jolt-metal)

The BACKEND, not the frontend, recognizes optimization opportunities:

```rust
// In jolt-cpu/src/compile.rs
fn compile<F: Field>(desc: &KernelDescriptor, challenges: &[F]) -> CpuKernel<F> {
    let formula = &desc.formula;

    // Pattern: pure product of D inputs, no challenge mixing
    if let Some(d) = formula.as_pure_product() {
        return compile_toom_cook::<F>(d, formula.num_products());
    }

    // Pattern: single degree-1 term per distinct input (weighted linear combo)
    if formula.is_weighted_linear() {
        return compile_eq_product::<F>(formula, challenges);
    }

    // Pattern: x*(x-1) with single input (booleanity)
    if let Some(_) = formula.as_booleanity() {
        return compile_hamming::<F>(formula, challenges);
    }

    // General: compile SoP expression with baked challenges
    compile_general_sop::<F>(formula, challenges)
}
```

Metal does the same — recognizes patterns and generates specialized MSL. The CPU
backend might recognize patterns that Metal doesn't, or vice versa. Each backend
optimizes independently.

### What Changes

| Before | After |
|--------|-------|
| Frontend produces `KernelShape::EqProduct` | Frontend produces `CompositionFormula { terms: [{coeff:1, inputs:[0,1]}] }` |
| Frontend produces `KernelShape::HammingBooleanity` | Frontend produces `CompositionFormula { terms: [{coeff:1, inputs:[0,0]}, {coeff:-1, inputs:[0]}] }` |
| Frontend produces `KernelShape::ProductSum{D,P}` | Frontend produces `CompositionFormula { terms: P × [{coeff:1, inputs:[t*D..t*D+D]}] }` |
| Frontend produces `KernelShape::Custom{expr}` | Frontend produces `CompositionFormula` from SoP normalization |
| `build_witness()` has 4-arm match | `build_witness()` has 0 match — just uploads buffers and compiles |
| `compile_descriptor()` returns untyped `Vec<F>` | Challenge values are part of the compilation API, not side-channel data |

### Sparse Shape

```rust
pub enum SumcheckAlgorithm {
    /// Dense weighted reduce over all 2^n pair positions.
    Dense(CompositionFormula),
    /// Sparse entry-list reduce over sorted entries.
    Sparse(SparseFormula),
}

pub struct SparseFormula {
    /// Per-entry composition (same SoP structure).
    pub entry_formula: CompositionFormula,
    /// Entry value layout (how many values per entry, types).
    pub entry_layout: EntryLayout,
}
```

### What This Eliminates

1. **No more `KernelShape` enum** — replaced by `CompositionFormula` (a data structure, not a tagged union)
2. **No more `InterpolationMode`** — backend decides standard grid vs Toom-Cook
3. **No more `compile_descriptor()` returning untyped `Vec<F>`** — challenges are a compile-time input to the backend
4. **No more 4-arm match in `build_witness()`** — one path: build buffers from formula, compile kernel, done
5. **No more stack-machine interpreter for "Custom"** — general SoP compilation replaces it (still possible to specialize via pattern matching in the backend)
6. **EqProduct/HammingBooleanity hand-coded kernels still exist** — but as backend optimizations, not IR concepts

## Decisions

- IR frontend owns pure `CompositionFormula` (SoP) only — no execution plan, no optimization hints
- Sparsity is a property of the polynomial, declared in the IR, interpreted by the compute backend
- Backend impl of the jolt-ir consumer has enough metadata from formula + sparsity annotations to generate correct kernels
- Both sides need to be expressive enough for their roles — IR captures mathematical structure, backend exploits it
