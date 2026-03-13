1# CPU Kernel Parity Report

**Goal:** Bring `jolt-cpu-kernels` + `jolt-compute` to full performance parity with the bespoke `compute_message` / `bind` implementations in `jolt-core`.

---

## 1. Gap Analysis: What Jolt-Core Does That The Kernel Compiler Doesn't

### 1.1 Evaluation Grid

| Aspect | Current kernel compiler | Jolt-core |
|--------|----------------------|-----------|
| **Grid** | `[0, 1, ..., degree]` | `[1, 2, ..., d-1, infinity]` (Toom-Cook) |
| **Product eval** | Linear interpolation per point: `lo + t*(hi - lo)` | Hierarchical factoring: d=16 as two d=8 half-products with sliding-window extrapolation |
| **Mul count** | D muls per grid point, D+1 grid points = O(D^2) | O(D log D) via balanced binary splitting + O(D) extrapolation |
| **Infinity point** | Not supported | `P(inf) = prod_i (hi_i - lo_i)` (free, no interpolation) |

The Toom-Cook approach is **the single largest performance difference**. For d=16: current = ~272 muls, Toom-Cook = ~120 muls.

### 1.2 Split-Eq Integration

The current `pairwise_reduce` treats the eq polynomial as a flat weights buffer:

```
for i in 0..half:
    evals = kernel.evaluate(lo_pairs[i], hi_pairs[i])
    for k: sums[k] += weights[i] * evals[k]
```

Jolt-core uses `SplitEqEvaluator::par_fold_out_in` with a fundamentally different parallelism topology:

```
parallel for x_out in E_out:
    for x_in in E_in:        // sequential inner loop (cache locality)
        g = x_out * |E_in| + x_in
        vals = kernel(g)
        for k: inner_acc[k].fmadd(E_in[x_in], vals[k])  // delayed reduction
    for k: outer_acc[k].fmadd(E_out[x_out], reduce(inner_acc[k]))
merge outer accumulators
```

**Differences:**
1. Two-level parallelism (outer parallel, inner sequential) vs flat parallel
2. Gruen's formula recovers cubic `s(X) = eq(X, w) * q(X)` from only the quadratic coefficients `[q(0), e]`, avoiding explicit eq multiplication per grid point
3. Cached prefix tables (`E_out_vec`, `E_in_vec`) popped lazily as variables bind
4. `current_scalar` tracks the accumulated product of bound eq evaluations

### 1.3 RaPolynomial: Lazy Materialization

The kernel compiler assumes dense interleaved buffers where `get_bound_coeff(j)` is `buf[j]`. The RA polynomial uses a state machine with indirect lookups:

```
Round1: F[lookup_indices[j].unwrap_or(0)]           // O(1) but indirect
Round2: F_0[indices[2j]] + F_1[indices[2j+1]]       // 2 lookups + add
Round3: F_00[...] + F_01[...] + F_10[...] + F_11[...] // 4 lookups + 3 adds
RoundN: dense[j]                                     // direct
```

This is a memory-efficient representation for sparse lookup polynomials where most entries are zero. The kernel compiler cannot express this — it sees uniform `Vec<F>` buffers.

### 1.4 Sparse Matrix Phases (RAM/Register Checking)

RAM/register read-write checking uses three phases with different data structures:

| Phase | Data structure | Binding method | Kernel shape |
|-------|---------------|----------------|-------------|
| Phase 1 | `ReadWriteMatrixCycleMajor` (sparse, sorted by row) | Merge-sort style: `bind_rows()` merges adjacent rows | `ra * (val + gamma*(val + inc))` with Gruen eq |
| Phase 2 | `ReadWriteMatrixAddressMajor` (sparse, sorted by col) | Same merge pattern on columns | Same formula, address-major access |
| Phase 3 | Dense `ra`, `val`, `inc` polynomials | Standard `sumcheck_evals_array` + parallel bind | `eq * ra * (val + gamma*(val + inc))` |

The sparse phases use **zero-allocation merge**: dry-run to count output size, then fill with `MaybeUninit`. Parallelized via recursive splitting at pivot points with `rayon::join`.

### 1.5 Spartan Outer: Fused R1CS Evaluation

Not a composition over pre-materialized polynomials at all. Instead, it evaluates the R1CS constraints **on-the-fly from trace data**:

```
for each (x_out, x_in) pair:
    step_idx = f(x_out, x_in)
    row_inputs = R1CSCycleInputs::from_trace(trace, step_idx)
    eval = R1CSEval::from_cycle_inputs(row_inputs)
    product = eval.extended_azbz_product_first_group(j)  // Az * Bz at extended point
    acc[j].fmadd(e_in, product)
```

Key features:
- **Two constraint groups** with different accumulator widths (`SmallAccumU` for booleans, `MedAccumS`/`WideAccumS` for magnitudes)
- **Univariate skip** first round: degree-9 polynomial over extended domain `[-D..D]`, multiplied by Lagrange kernel
- **Multiquadratic expansion**: streaming rounds expand binary evaluation grid to tertiary via `MultiquadraticPolynomial::expand_linear_grid_to_multiquadratic`

### 1.6 Booleanity: `h^2 - h` Optimization

The booleanity check computes `h(j)^2 - h(j)` which decomposes into two values per pair:

```
h0 = H.get_bound_coeff(2*g)
h1 = H.get_bound_coeff(2*g+1)
delta = h1 - h0
[c0, e] = [h0^2 - h0, delta^2]   // only 2 values needed
```

Then Gruen's formula gives the cubic `eq(X, w) * q(X)` from just `c0` and `e`. No grid evaluation needed at all.

### 1.7 Claim Reductions: eq * poly Product

Most claim reductions are degree-2 sumchecks of the form `eq(r, x) * g(x)`. The current `EqProductCompute` in `jolt-zkvm` already handles this with a direct loop, but doesn't use split-eq or delayed reduction.

### 1.8 Streaming Sumcheck (Window Schedule)

The Spartan outer uses a two-phase streaming sumcheck:
1. **Streaming phase** (first ~n/2 rounds): Geometrically growing windows over the trace. Each window evaluates on a 3^w multiquadratic grid, avoids materializing full polynomial.
2. **Linear phase** (remaining rounds): Standard dense polynomial evaluation.

The window schedule is cost-optimal: `window_ratio = ln(2) / ln((d+1)/2)`.

---

## 2. Sumcheck Algorithms to Support

### Algorithm 1: RA Virtual ProductSum (dominant cost, ~40% of prover time)

**Formula:** `g(X) = sum_j eq(r', X, j, r) * prod_i ra_i(X, j)`

**Current state in jolt-zkvm:** `mles_product_sum.rs` — already has Toom-Cook kernels (d=4,8,16,32) with split-eq. This is the **most mature** implementation and is close to parity.

**Gap:**
- Uses `SplitEqEvaluator::par_fold_out_in` directly, bypasses `ComputeBackend::pairwise_reduce`
- Product evaluation kernels (`eval_prod_d_assign`) use Toom-Cook factoring, not the kernel compiler's linear interpolation
- `RaPolynomial` state machine is deeply integrated into the inner loop via `get_bound_coeff`

**Action:** The RA virtual sumcheck should remain as bespoke `SumcheckCompute` implementations with the existing Toom-Cook kernels and `SplitEqEvaluator`. The kernel compiler cannot match this without becoming a full symbolic execution engine. Mark as **"bespoke, not kernel-compiled"**.

### Algorithm 2: Booleanity Check (stage 6, ~10% of prover time)

**Formula:** `0 = sum_{k,j} eq(r_addr, k) * eq(r_cycle, j) * sum_i gamma^{2i} * (ra_i(k,j)^2 - ra_i(k,j))`

**Two phases:**
- Phase 1 (address binding): `par_fold_out_in_unreduced` over address variables with G tables
- Phase 2 (cycle binding): `par_fold_out_in_unreduced` over cycle variables with SharedRaPolynomials

**Gap:** Two-phase structure with pre-computed G tables, shared RA polynomial tables, gamma-weighted booleanity formula, Gruen's cubic recovery. The G computation is a separate pushforward pass over the trace.

**Action:** Bespoke `SumcheckCompute` implementation. The two-phase G-then-H structure, SharedRaPolynomials state machine, and Gruen integration are too interleaved to factor through kernel compilation.

### Algorithm 3: RAM/Register Read-Write Checking (stage 4, ~15% of prover time)

**Formula:** `eq * ra * (val + gamma*(val + inc))`

**Three phases** with fundamentally different data structures (sparse matrix → address-major → dense).

**Gap:** Sparse matrix merge-sort binding, Gruen eq integration in phase 1, phase transitions that materialize dense polynomials, per-entry `compute_evals` with match on (even, odd, both, neither).

**Action:** Bespoke. The three-phase sparse→dense transition pattern is irreducible to dense-buffer kernel execution.

### Algorithm 4: Spartan Outer (stage 1, ~20% of prover time)

**Formula:** `eq(tau, x) * Az(x) * Bz(x)` where Az, Bz are derived from R1CS constraints applied to the execution trace.

**Two sub-stages:**
1. **Univariate skip** (first round): Extended domain evaluation → Lagrange interpolation → degree-27 polynomial
2. **Streaming + linear** (remaining rounds): Windowed multiquadratic evaluation with split-eq

**Gap:** On-the-fly trace evaluation, two constraint groups with typed accumulators, multiquadratic grid expansion, streaming window schedule.

**Action:** Bespoke. The computation derives polynomial values from trace data at each point, not from pre-materialized buffers. No buffer-based kernel can express this.

### Algorithm 5: Claim Reductions (stage 3 + 7, ~10% of prover time)

**Formula:** `eq(r, x) * g(x)` (degree 2) or `eq(r, x) * poly_0(x) * poly_1(x)` (degree 3)

Six reduction types:
1. **Advice** (two-phase P*Q → dense)
2. **Hamming weight + address** (fused 3-claim)
3. **Increments** (4 gamma-batched claims)
4. **Instruction lookups** (5 virtual polys)
5. **RAM RA** (3 RA claims, cycle-only)
6. **Registers** (3 virtual register polys)

**Gap:** Some use P*Q prefix-suffix decomposition (phase 1), some use direct polynomial evaluation (phase 2). All use `sumcheck_evals_array` which extrapolates from two endpoints via linear interpolation.

**Action:** These are the **best candidates for kernel compilation**, specifically:
- Phase 2 (dense) of all claim reductions: `eq * g` products with standard buffers
- Degree-2 and degree-3 compositions over pre-materialized dense polynomials
- The `EqProductCompute` pattern (`eq_product.rs`) is a clean kernel target

### Algorithm 6: Hamming Booleanity (stage 6, ~5% of prover time)

**Formula:** `eq(r_cycle, j) * (H(j)^2 - H(j))` where H is a boolean indicator.

**Gap:** Uses Gruen's cubic recovery from only two quadratic coefficients. The `h^2 - h` decomposition into `[h0^2 - h0, delta^2]` is a unique optimization.

**Action:** Can be a specialized Custom kernel shape, but the Gruen recovery step lives outside the kernel (in the split-eq evaluator). The kernel only needs to compute `[h0^2 - h0, (h1-h0)^2]`.

---

## 3. Plan: What To Change

### Phase 0: Accept the architecture split (no code changes)

Not every sumcheck should be kernel-compiled. The clean split is:

| Category | Compilation | Why |
|----------|------------|-----|
| RA virtual (alg 1) | Bespoke | RaPolynomial state machine, Toom-Cook kernels already optimal |
| Booleanity (alg 2) | Bespoke | Two-phase G/H structure, SharedRaPolynomials |
| RAM/register (alg 3) | Bespoke | Sparse matrix phases, merge-sort binding |
| Spartan outer (alg 4) | Bespoke | On-the-fly trace evaluation, streaming windows |
| Claim reductions (alg 5) | **Kernel-compiled** (phase 2) | Dense eq*poly products |
| Hamming booleanity (alg 6) | **Kernel-compiled** | `h^2 - h` is a simple Custom expr |

The kernel compiler targets **~15% of prover time** (claim reductions + hamming booleanity). The remaining 85% uses bespoke `SumcheckCompute` implementations that directly use `SplitEqEvaluator` and specialized polynomial types.

### Phase 1: Toom-Cook evaluation grid in `jolt-cpu-kernels`

**Problem:** The current ProductSum kernels evaluate on `[0, 1, ..., D]` using D+1 linear interpolations per point. Jolt-core evaluates on `[1, 2, ..., D-1, infinity]` using balanced binary factoring + extrapolation.

**Change `jolt-cpu-kernels/src/product_sum.rs`:**

Replace the current D=4,8,16 specializations and generic fallback with the Toom-Cook approach from `mles_product_sum.rs`:

```
compile_d4: Use eval_linear_prod_2_internal → balanced tree (as in eval_prod_4_assign)
compile_d8: Use eval_linear_prod_4_internal → batch_helper → extrapolate (as in eval_prod_8_assign)
compile_d16: Use eval_linear_prod_8_internal → sliding window ex8 (as in eval_prod_16_assign)
compile_d32: Use eval_half_16_base → expand16_to_u32 (as in eval_prod_32_assign)
```

**Grid change:** Output becomes `[P(1), P(2), ..., P(D-1), P(infinity)]` instead of `[P(0), P(1), ..., P(D)]`. The consumer (`finish_mles_product_sum_from_evals`) recovers `P(0)` from the claim.

**New specializations:** Add d=2,3,5,6,7 kernels (already written in `mles_product_sum.rs`).

**Kernel signature change:**
```rust
// Before:
fn(lo: &[F], hi: &[F], degree: usize) -> Vec<F>  // [P(0), ..., P(D)]

// After:
fn(lo: &[F], hi: &[F], degree: usize) -> Vec<F>  // [P(1), ..., P(D-1), P(inf)]
```

The `pairwise_reduce` caller reconstructs `P(0)` from the claim sum.

**Extrapolation helpers to port:**
- `ex2(f: &[F; 2], f_inf: &F) -> F` — degree-2 extrapolation
- `ex4(f: &[F; 4], f_inf6: &F) -> F` — degree-4 extrapolation
- `ex4_2(f: &[F; 4], f_inf6: &F) -> (F, F)` — degree-4 double extrapolation
- `ex8(f: &[F; 8], f_inf40320: F) -> F` — degree-8 extrapolation
- `ex16(f: &[F; 16], f_inf16_fact: F) -> F` — degree-16 extrapolation

These use small-scalar multiplications (`mul_u64`) and binomial coefficient identities.

**Estimated impact:** 2-3x speedup on ProductSum kernels for d >= 8.

### Phase 2: Split-eq aware pairwise_reduce

**Problem:** The current `pairwise_reduce` uses a flat `weights` buffer (eq table materialized as `Vec<F>`). Jolt-core's `par_fold_out_in` uses cached prefix tables with two-level parallel-sequential topology.

**Option A: SplitEqEvaluator integration in `jolt-compute`**

Add a new method to `ComputeBackend`:

```rust
fn split_eq_reduce<F: Field>(
    &self,
    inputs: &[&Self::Buffer<F>],
    eq_out: &Self::Buffer<F>,
    eq_in: &Self::Buffer<F>,
    kernel: &Self::CompiledKernel<F>,
    degree: usize,
) -> Vec<F>;
```

This preserves the two-level topology: parallel over `eq_out`, sequential over `eq_in`, with `FieldAccumulator` delayed reduction at each level.

**Option B: Keep SplitEqEvaluator in jolt-sumcheck, bespoke witness uses it directly**

Since most sumcheck instances that use split-eq are bespoke anyway (RA virtual, booleanity, RAM, Spartan), the `SplitEqEvaluator` doesn't need to go through `ComputeBackend`. Only claim reductions (which use flat eq tables) go through `pairwise_reduce`.

**Recommendation:** Option B. The split-eq topology is a protocol-level optimization, not a backend primitive. Bespoke witnesses call `SplitEqEvaluator` directly. `pairwise_reduce` stays simple for the cases that use it.

### Phase 3: Gruen's cubic recovery in jolt-sumcheck

**Problem:** The `gruen_poly_deg_3` method reconstructs a cubic round polynomial from only two quadratic coefficients + the previous claim. This avoids evaluating at all D+1 grid points when the composition has degree 2 and eq adds degree 1.

**Already done:** `SplitEqEvaluator` in `jolt-sumcheck/src/split_eq.rs` already has `gruen_poly_deg_3`. The bespoke witnesses already use it.

**For kernel-compiled witnesses:** When the kernel's degree is 2 (e.g., `eq * g`), the round polynomial is cubic. Instead of evaluating at 4 grid points, the kernel can return only `[q(0), e]` (the constant and quadratic coefficient of the quotient) and let Gruen's formula do the rest.

Add a `KernelShape::QuadraticForGruen` variant that produces only 2 values per pair position, paired with `SplitEqEvaluator::gruen_poly_deg_3` to recover the cubic polynomial.

### Phase 4: FieldAccumulator in kernel evaluation

**Problem:** The current `pairwise_reduce` uses `FieldAccumulator::fmadd` for the outer reduction (weighting by eq), but the kernel evaluation itself (`kernel.evaluate(lo, hi, degree)`) uses standard field arithmetic internally. Jolt-core uses `mul_to_product_accum` / `UnreducedProductAccum` inside the kernel to defer Montgomery reductions.

**Change `jolt-cpu-kernels`:** The compiled closures should use `FieldAccumulator` internally for the accumulation across product groups (in ProductSum) and for the stack operations (in Custom).

This requires the kernel to be generic over the accumulator type, or to use `F::Accumulator` directly. The signature stays the same but internal arithmetic defers reduction.

### Phase 5: Compact buffer support in pairwise_reduce

**Already partially done:** `CpuBackend::pairwise_reduce_mixed` and `AnyBuffer` support heterogeneous element types. The kernel reads through `AnyBuffer::pair(i)` which promotes compact scalars to `F` inline.

**Remaining:** Ensure `AnyBuffer` covers all compact types used in jolt-core: `bool`, `u8`, `u16`, `u32`, `u64`, `i64`, `i128`, `u128`, `S128`.

### Phase 6: Batch interpolate optimization

**Problem:** After each sumcheck round, all polynomial buffers are bound via `interpolate_pairs`. The current implementation processes each buffer independently. When many small buffers exist (late rounds), Rayon overhead dominates.

**Already partially done:** `interpolate_pairs_batch` parallelizes across buffers.

**Remaining:** Ensure the batch method is used by all bespoke witnesses for their dense polynomial binding.

---

## 4. Bespoke Witness Implementations Needed

Each bespoke sumcheck needs a `SumcheckCompute` implementation in `jolt-zkvm`:

### 4.1 RA Virtual (`ra_virtual.rs`) — **Already implemented**

Uses `compute_mles_product_sum` / `compute_mles_weighted_sop` with Toom-Cook kernels and `SplitEqEvaluator`. Round polynomial via `finish_mles_product_sum_from_evals`.

Status: Done, needs testing against jolt-core for correctness.

### 4.2 Booleanity (`booleanity.rs`) — Needs implementation

Two-phase `SumcheckCompute`:
- Phase 1: Pre-compute G tables via trace pushforward, `par_fold_out_in_unreduced` with `[h0^2 - h0, delta^2]`
- Phase 2: SharedRaPolynomials with gamma-weighted `h*(h - rho)` and `delta^2`

Depends on: `SharedRaPolynomials` type (port from jolt-core), G computation from trace.

### 4.3 RAM/Register Read-Write Checking — Needs implementation

Three-phase `SumcheckCompute`:
- Phase 1: `ReadWriteMatrixCycleMajor` with Gruen eq and sparse entry evaluation
- Phase 2: `ReadWriteMatrixAddressMajor` with merged eq
- Phase 3: Dense polynomials with standard evaluation

Depends on: Sparse matrix types, merge-sort binding infrastructure, phase transition logic.

### 4.4 Spartan Outer — Needs implementation

Two-stage `SumcheckCompute`:
- Stage 1 (univariate skip): Extended domain evaluation from trace, Lagrange interpolation
- Stage 2 (streaming + linear): Windowed multiquadratic evaluation with split-eq

Depends on: `R1CSCycleInputs`, `R1CSEval`, accumulator types, streaming schedule, `MultiquadraticPolynomial`.

### 4.5 Claim Reductions — Partially implemented

`EqProductCompute` handles the simplest case (eq * g, degree 2). Needs:
- Prefix-suffix (P*Q) decomposition for phase 1
- Multiple-polynomial products (eq * poly_0 * poly_1, degree 3)
- Gamma-weighted batching across claim types
- Split-eq integration for the address/cycle variable decomposition

### 4.6 Hamming Weight + Address Fusion — Needs implementation

Fused three-claim `SumcheckCompute`:
- `G_i * (gamma_hw + gamma_bool * eq_bool + gamma_virt * eq_virt_i)`
- Shared `eq_bool`, per-family `eq_virt`

---

## 5. Implementation Priority

### Immediate (blocks e2e testing)

1. **Phase 1: Toom-Cook kernels** — Port `eval_prod_{2..32}_assign` and extrapolation helpers from `mles_product_sum.rs` into `jolt-cpu-kernels`. Change grid from `[0..D]` to `[1..D-1, inf]`. Update `pairwise_reduce` callers.

2. **4.5: Claim reduction witnesses** — Extend `EqProductCompute` with split-eq support, P*Q phases, and gamma batching. These are the cleanest kernel targets.

### Short-term (enables full prover pipeline)

3. **4.2: Booleanity witness** — Port SharedRaPolynomials and G computation.
4. **4.3: RAM/register witness** — Port sparse matrix types and three-phase structure.
5. **4.6: Hamming fusion witness** — Port fused three-claim reduction.

### Medium-term (performance parity)

6. **Phase 3: Gruen integration** — Add `QuadraticForGruen` kernel shape for degree-2 compositions.
7. **Phase 4: FieldAccumulator in kernels** — Defer Montgomery reductions inside kernel closures.
8. **4.4: Spartan outer witness** — Port streaming sumcheck, R1CS evaluation, multiquadratic expansion.

### Deferred (GPU readiness)

9. **Phase 2: Split-eq in ComputeBackend** — Only if GPU backends need it.
10. **Phase 5/6: Compact buffers + batch interpolate** — Already partially done.

---

## 6. What The Kernel Compiler Should and Should Not Do

### Should do (sweet spot)

- Compile `ProductSum` with Toom-Cook evaluation for d=2..32
- Compile `Custom` expressions with stack-machine execution
- Support `QuadraticForGruen` shape that returns only 2 values for Gruen recovery
- Handle challenge baking for Custom kernels
- Produce zero-overhead CPU closures via monomorphization

### Should not do (leave to bespoke witnesses)

- Split-eq topology (parallel outer, sequential inner)
- RaPolynomial lazy materialization state machine
- Sparse matrix merge-sort binding
- On-the-fly trace evaluation (Spartan outer)
- Streaming/windowed sumcheck scheduling
- Phase transitions between data structures
- SharedRaPolynomials memory sharing

The kernel compiler is a **leaf-node evaluator**: given dense paired values at a single position, evaluate the composition. Everything above that (parallelism topology, eq factoring, sparse access, phasing) is the witness implementation's responsibility.

---

## 7. Performance Targets

| Algorithm | Current (est.) | Target | Bottleneck |
|-----------|---------------|--------|------------|
| RA virtual d=16 | 1.0x (already Toom-Cook) | 1.0x | `par_fold_out_in` + `eval_prod_16_assign` |
| RA virtual d=4 | 1.0x | 1.0x | Same |
| Claim reduction (eq * g) | 0.5x (no split-eq, no Gruen) | 0.9x | Add split-eq + Gruen recovery |
| Booleanity | 0.3x (not implemented) | 0.9x | Port bespoke witness |
| RAM phase 1 | N/A (not implemented) | 0.9x | Port sparse matrix phases |
| RAM phase 3 | N/A | 0.95x | Dense buffers, straightforward |
| Spartan outer | N/A | 0.85x | Streaming schedule complexity |

"1.0x" means parity with jolt-core. Target overall: **>0.9x parity** on all stages.

---

## Appendix A: Toom-Cook Kernel Evaluation Reference

The Toom-Cook approach evaluates `P(x) = prod_i (lo_i + x * (hi_i - lo_i))` on the grid `[1, 2, ..., d-1, infinity]` by factoring the product into balanced halves and extrapolating missing points.

### d=4 Example

```
// Split into 2 pairs: {p0, p1} and {p2, p3}
(a1, a2, a_inf) = eval_linear_prod_2_internal(p0, p1)  // A at [1, 2, inf]
a3 = ex2([a1, a2], a_inf)                                // A at 3 via extrapolation

(b1, b2, b_inf) = eval_linear_prod_2_internal(p2, p3)  // B at [1, 2, inf]
b3 = ex2([b1, b2], b_inf)                                // B at 3

outputs = [a1*b1, a2*b2, a3*b3, a_inf*b_inf]           // P at [1, 2, 3, inf]
```

**Cost:** 6 muls (3 per half) + 2 adds (extrapolation) + 4 muls (products) = 12 muls total.
**Naive:** 4 grid points * 4 muls per point = 16 muls. **Savings: 25%.**

### d=16 Example

```
a = eval_linear_prod_8_internal(p[0..8])   // A at [1..8, inf]: 9 values
b = eval_linear_prod_8_internal(p[8..16])  // B at [1..8, inf]: 9 values

// Direct products for first 8 points
for i in 0..8: outputs[i] = a[i] * b[i]

// Sliding window extrapolation for points 9..15
a_inf40320 = a[8] * 40320              // scale for ex8 formula
for i in 0..7:
    a[8+i] = ex8(a[i..i+8], a_inf40320)  // A at 9+i
    b[8+i] = ex8(b[i..i+8], b_inf40320)  // B at 9+i
    outputs[8+i] = a[8+i] * b[8+i]

outputs[15] = a[8] * b[8]              // P(infinity)
```

**Cost:** ~80 muls for 16 evaluations. **Naive:** 16 * 16 = 256 muls. **Savings: 69%.**

### Extrapolation Formula (ex8)

Uses finite differences (binomial coefficients with alternating signs):

```
P(9) = 8*(f[1]+f[7]) + 56*(f[3]+f[5]) - 28*(f[2]+f[6]) - 70*f[4] - f[0] + f_inf*40320
```

All multiplications are by small constants (`mul_u64`), which are 2-5x faster than general field multiplication.

## Appendix B: Gruen's Cubic Recovery

Given a degree-2 quotient `q(X)` where only `q(0)` and the quadratic coefficient `e` are computed by the kernel, and the previous claim `s(0) + s(1)`:

```
eq(0) = current_scalar * (1 - w)
eq(1) = current_scalar * w
eq_m = eq(1) - eq(0)

cubic(0) = eq(0) * q(0)
cubic(1) = claim - cubic(0)      // from s(0) + s(1) = claim
q(1) = cubic(1) / eq(1)

// Extrapolate quadratic via finite differences:
q(2) = q(1) + q(1) - q(0) + 2*e
q(3) = q(2) + q(1) - q(0) + 4*e

// Recover cubic at all 4 points:
s = [eq(0)*q(0), cubic(1), eq(2)*q(2), eq(3)*q(3)]
```

**Cost:** 1 division + ~10 muls + ~10 adds. Replaces evaluating the kernel at 4 grid points, which for degree-3 compositions would be 4 * (kernel cost).
