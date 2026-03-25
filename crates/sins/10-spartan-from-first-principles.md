# Analysis 10: Spartan From First Principles

## What Spartan Actually Does

Spartan proves R1CS satisfaction: given matrices A, B, C and witness w, prove that
`Az ⊙ Bz = Cz` where `z = (1, x, w)` and ⊙ is Hadamard (element-wise) product.

This reduces to a sumcheck:

```
Σ_x eq(τ, x) · [Az(x) · Bz(x) - Cz(x)] = 0
```

That's a **zero-check weighted by eq**. The composition is `Az · Bz - Cz` which is
degree 2 in the witness variables (each of Az, Bz, Cz is multilinear in x).

## Breaking Down the "Special" Parts

### Part 1: The outer sumcheck

Formula: `Σ_x eq(τ, x) · q(x)` where `q(x) = Az(x)·Bz(x) - Cz(x)`

This is **exactly** a dense weighted reduce with:
- Weight polynomial: `eq(τ, ·)`
- Composition: `Az · Bz - Cz`
- Degree: 3 (eq is degree 1, Az·Bz is degree 2)

Wait — actually the degree is 2 in the evaluation, because `Az(x)`, `Bz(x)`, `Cz(x)`
are each multilinear in x. The product `Az · Bz` is degree 2. With eq weighting, the
round polynomial is degree 3.

This is the SAME pattern as RAM read-write checking (`eq · ra · val`), just with
different polynomials.

### Part 2: Az, Bz, Cz are matrix-vector products

`Az(x) = Σ_y A(x,y) · z(y)` — this is a dot product of a sparse matrix row with the
witness vector. The "bespoke" part of Spartan is that the polynomial `Az(x)` is not
stored as a table — it's computed on-the-fly from the sparse matrix A and the dense
witness z.

In the current code, this is why Spartan uses streaming sumcheck: instead of
materializing the full `Az` table (which would be huge), it streams over the sparse
matrix entries and computes contributions to the round polynomial incrementally.

But this is **not** fundamentally different from how `SparseRwEvaluator` works for RAM.
RAM read-write also has sparse entries that contribute to the round polynomial. The
difference is just the formula and entry structure:

| | Spartan outer | RAM read-write |
|--|--------------|----------------|
| Entries | Sparse matrix (row, col, val) | Sparse memory ops (addr, val, inc) |
| Formula per entry | `eq(τ,x) · A_val · z(col)` | `eq(r,x) · ra(addr) · val` |
| Accumulation | Sum over entries at each x | Sum over entries at each x |

### Part 3: The univariate skip

The first round uses a Lagrange polynomial `L(τ_high, Y)` to handle the "outer"
dimension efficiently. This is an optimization for when the sumcheck domain has a
known structure (uniform R1CS with repeated constraint patterns).

But univariate skip is already a general concept — `SumcheckCompute::first_round_polynomial()`
exists for exactly this. It's not Spartan-specific.

### Part 4: The inner sumcheck (witness opening)

After the outer sumcheck produces point `r_x`, Spartan needs to evaluate `Az(r_x)`,
`Bz(r_x)`, `Cz(r_x)`. Each is a dot product `Σ_y M(r_x, y) · z(y)`. This is another
sumcheck:

```
Σ_y eq(τ_y, y) · [ρ_A · A(r_x, y) + ρ_B · B(r_x, y) + ρ_C · C(r_x, y)] · z(y) = claimed
```

This is a dense weighted reduce:
- Weight: `eq(τ_y, ·)`
- Composition: `[ρ_A·A_row(y) + ρ_B·B_row(y) + ρ_C·C_row(y)] · z(y)`
- Degree: 2

Completely standard. The matrix row `A(r_x, ·)` can be computed from the sparse matrix
and the bound point `r_x`.

## So What's Actually Special?

**Nothing, algorithmically.** Spartan is:

1. A sparse-data dense-weighted sumcheck (outer) — like RAM read-write
2. Followed by a dense-data dense-weighted sumcheck (inner) — like claim reductions
3. With a univariate skip optimization on the first round — already supported generically

The reason it's special-cased today is **implementation history**, not mathematical
necessity. The streaming sumcheck infrastructure was built specifically for Spartan
before the generic `SumcheckCompute` / `ComputeBackend` abstractions existed. Now that
those abstractions exist, Spartan can be expressed as two graph vertices.

## How Spartan Fits the Graph Model

### Outer Vertex

```
SumcheckVertex {
    input: Constant(0),  // zero-check
    formula: ClaimDefinition {
        // Az(x) · Bz(x) - Cz(x)
        // Where Az, Bz, Cz are virtual polys computed from sparse matrices
        expr: opening(0) * opening(1) - opening(2),
        opening_bindings: [Az, Bz, Cz],
    },
    weighting: Eq,  // eq(τ, ·)
    degree: 3,
    num_vars: log_num_constraints,
    phases: [
        PhasePlan {
            algorithm: Sparse,  // matrix entries are sparse
            num_rounds: log_num_constraints,
        }
    ],
}
```

The "virtual polynomials" Az, Bz, Cz are computed from sparse matrix + witness, just
like RAM's virtual polynomials are computed from trace. The `PolynomialSource` (from
Sin 4) handles this: `source.get(PolynomialId::Az)` materializes on demand from the
sparse matrix.

### Inner Vertex

```
SumcheckVertex {
    input: Formula { ρ_A·az_eval + ρ_B·bz_eval + ρ_C·cz_eval },
    formula: ClaimDefinition {
        // [ρ_A·A_row(y) + ρ_B·B_row(y) + ρ_C·C_row(y)] · z(y)
        expr: (challenge(0)*opening(0) + challenge(1)*opening(1) + challenge(2)*opening(2)) * opening(3),
        opening_bindings: [A_row, B_row, C_row, Witness],
    },
    weighting: Eq,
    degree: 2,
    num_vars: log_witness_size,
    phases: [
        PhasePlan {
            algorithm: Dense,  // matrix rows are dense after binding r_x
            num_rounds: log_witness_size,
        }
    ],
}
```

### What About the Univariate Skip?

The outer vertex can declare `first_round: UnivariateSkip` in its execution plan. The
evaluator handles this via `SumcheckCompute::first_round_polynomial()` — no special
Spartan code needed.

### What About Streaming?

The sparse algorithm in the backend (from Analysis 8) handles this. Sparse matrix entries
are uploaded to `SparseBuffer`, and `sparse_reduce` iterates them. This IS streaming —
it's just expressed as a backend primitive instead of a bespoke prover.

## What Would Change

| Before | After |
|--------|-------|
| `prove_from_graph` has S1 special-cased outside the loop | S1 is two vertices in the graph, processed by the same generic loop |
| `jolt-spartan` crate has its own prover with streaming sumcheck | Spartan outer uses `Algorithm::Sparse` via the backend |
| Spartan inner has its own sumcheck implementation | Spartan inner uses `Algorithm::Dense` via the backend |
| `UniformSpartanKey` preprocessing is separate | Spartan matrices become part of the `PolynomialSource` |
| Graph stages start at S2 | Graph stages start at S1 |

## Open Questions for Discussion

1. **Spartan preprocessing**: The `UniformSpartanKey` precomputes matrix structure (row
   indices, column indices, values). In the new model, this becomes part of the witness
   source — the `PolynomialSource` knows how to compute `Az(x)` from the key. Is this
   the right place, or should preprocessing remain separate?

2. **Performance**: Spartan's streaming sumcheck is highly optimized (fused eq·A·z
   evaluation with delayed reduction). Would expressing it as `sparse_reduce` through
   the backend lose performance? The backend would need to be as efficient as the current
   hand-written code.

3. **The witness commitment**: Spartan's inner sumcheck opens the committed witness at
   `r_y`. This is currently handled by Spartan internally. In the graph model, it would
   be an `Opening` vertex like any other committed polynomial. This is clean but means
   the witness commitment flow must be wired through the graph.

4. **Uniform R1CS structure**: Jolt's R1CS is "uniform" — all constraints have the same
   sparsity pattern per cycle. This enables massive optimization (single constraint
   template, applied T times). Does the sparse backend primitive need to know about
   uniformity, or can it be encoded as repeated sparse entries?

## Decisions — DISSOLVED by Analysis 11

Spartan is not infrastructure — it's a recipe encoded as graph vertices.

- `jolt-spartan` as a prover crate is eliminated
- R1CS data types (`UniformSpartanKey` → `UniformR1cs`) extracted to new `jolt-r1cs` crate
- The Spartan "recipe" becomes graph vertices:
  - `UnivariateSkip` vertex (1 round, first round of outer sumcheck)
  - `Dense` vertex (log_m - 1 rounds, remaining outer sumcheck rounds)
  - Edge computation: `combined_partial_evaluate` (materializes M(r_x, ·))
  - `Dense` vertex (log_n rounds, inner sumcheck)
  - `Opening` vertex: z(r_y) via standard PCS infrastructure
- No S1 special case in `prove_from_graph` — same generic loop as all other stages
- UniSkip is a backend algorithm primitive alongside Dense and Sparse
- For external consumers wanting a standalone Spartan SNARK: `jolt-r1cs` + `jolt-sumcheck` + any PCS = complete recipe in ~50 lines
- See [Analysis 11](11-unified-execution-model.md) for the unified model
