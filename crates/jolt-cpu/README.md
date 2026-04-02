# jolt-cpu

CPU compute backend and kernel compiler for the Jolt zkVM.

Part of the [Jolt](https://github.com/a16z/jolt) zkVM.

## Overview

This crate implements the `ComputeBackend` trait from `jolt-compute` for CPU execution. It compiles `jolt-compiler` composition formulas into optimized closures and runs sumcheck reduce/bind operations using Rayon parallelism.

### Compilation (`compile<F>(spec) -> CpuKernel<F>`)

The `compile` function inspects the formula's structure and dispatches to the best evaluator:

- **ProductSum** (Toom-Cook) — When `formula.as_product_sum()` matches (all terms are products of `D` inputs with coefficient 1), evaluates on the Toom-Cook grid `{1, ..., D-1, inf}`. Covers ~80% of prover time. Uses hand-unrolled specializations for D=4,8,16,32 via `toom_cook::eval_prod_*_assign`, with a generic loop fallback for other D values.

- **Generic** — Fallback for formulas containing challenges, non-unit coefficients, or repeated inputs. Evaluates on the standard grid `{0, 2, 3, ..., degree}` (skipping `t=1`). Compiles the formula into a `Vec<(coeff, Vec<CompiledFactor>)>` at compile time; the closure walks this at eval time.

### Backend operations

`CpuBackend` implements `ComputeBackend` with `Buffer<T> = Vec<T>`:

- **`reduce`** — Sums kernel evaluations over all pairs. Three iteration modes:
  - *Dense*: contiguous `(lo, hi)` pairs via `LowToHigh` or `HighToLow` layout
  - *DenseTensor*: split-eq weighted pairs with outer/inner weight tables
  - *Sparse*: sorted key column defines pairs; adjacent `(2k, 2k+1)` entries merge, unmatched entries pair with zero

- **`bind`** — Interpolates each buffer at the sumcheck challenge, halving buffer size. Dense mode interpolates in-place; sparse mode also halves the key space.

- **`eq_table` / `lt_table` / `eq_plus_one_table`** — Evaluation tables for eq and lt polynomials.

All operations use Rayon `par_iter().fold().reduce()` when buffer size exceeds `PAR_THRESHOLD` (1024 pairs), falling back to sequential loops below that threshold. Dense reduce dispatches to const-generic specializations for common `(num_inputs, num_evals)` pairs (2-32), using stack-allocated `[F; N]` scratch arrays instead of heap Vecs.

### Toom-Cook module (`pub mod toom_cook`)

Re-exported for use by other crates. Provides:
- `eval_prod_{4,8,16,32}_assign` — unrolled specializations
- `eval_linear_prod_assign` — generic loop for any D
- `eval_prod_generic_assign` — generic via `DenseMatrix` Vandermonde inversion

## Dependency Position

```
jolt-field    --+
jolt-compiler --+--> jolt-cpu
jolt-compute  --+
```

Downstream: `jolt-sumcheck`, `jolt-zkvm`, `jolt-metal` (parity testing).

## Feature Flags

| Flag | Default | Effect |
|------|---------|--------|
| `parallel` | yes | Enables Rayon parallelism in reduce/bind/eq_table |

## Benchmarks

Run with `cargo bench -p jolt-cpu`.

### Benchmark groups

- **`product_sum_kernel_eval`** — Single-pair Toom-Cook kernel evaluation (D=4,8,16 x P=1,4)
- **`custom_kernel_eval`** — Generic formula evaluation (booleanity, product)
- **`dense_reduce`** — Full dense reduce pipeline (D=4,8,16 x 2^14,2^18)
- **`sparse_reduce`** — Full sparse reduce pipeline (D=4,8 x 2^14,2^18)
- **`bind`** — Dense and sparse bind (2^14, 2^18)
- **`rayon_vs_backend`** — Direct Rayon fold vs `ComputeBackend::reduce` abstraction overhead

## License

MIT
