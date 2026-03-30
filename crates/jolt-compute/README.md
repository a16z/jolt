# jolt-compute

Backend-agnostic compute device abstraction for the Jolt zkVM.

Part of the [Jolt](https://github.com/a16z/jolt) zkVM.

## Overview

This crate defines the `ComputeBackend` trait — a protocol-agnostic interface over typed buffer management and parallel primitives (pairwise interpolation, composition-reduce, product tables).

Concrete backends live in separate crates:
- `jolt-cpu` — CPU backend (`CpuBackend`) using `Vec<T>` buffers with Rayon parallelism. After monomorphization every trait method compiles to a direct function call — identical codegen to hand-written Rayon code.
- `jolt-metal` — Apple Metal GPU backend (`MetalBackend`) with compiled MSL shader kernels.

CUDA and WebGPU backends will follow the same pattern.

## Public API

### Types

- **`Scalar`** — Marker trait for types storable in device buffers. Blanket-implemented for primitive integers, `bool`, and all `Field` types.

- **`BindingOrder`** — Variable binding order for polynomial interpolation:
  - `LowToHigh` (default) — Interleaved pairs `(buf[2i], buf[2i+1])`. Binds least-significant variable first.
  - `HighToLow` — Split-half pairs `(buf[i], buf[i + n/2])`. Binds most-significant variable first.

### Hybrid Backend

- **`HybridBackend<P, Fb>`** — Wraps a primary (GPU) and fallback (CPU) backend. Buffers start on the primary and migrate to the fallback when they shrink below a size threshold during interpolation. The transition is one-way.
- **`HybridBuffer<T, P, Fb>`** — Enum over `Primary(P::Buffer<T>)` or `Fallback(Fb::Buffer<T>)`.
- **`HybridKernel<F, P, Fb>`** — Holds both backends' compiled kernels so dispatch works after migration.

### Trait: `ComputeBackend`

Core device abstraction with associated types:
- `Buffer<T: Scalar>` — typed device buffer handle
- `CompiledKernel<F: Field>` — opaque compiled kernel for composition-reduce

| Method | Description |
|--------|-------------|
| `upload` / `download` / `alloc` / `len` | Buffer management |
| `interpolate_pairs` | Pairwise linear interpolation, halving buffer (supports type promotion) |
| `interpolate_pairs_batch` | Batched `interpolate_pairs` across multiple buffers |
| `interpolate_pairs_inplace` | In-place pairwise interpolation with `BindingOrder` |
| `interpolate_pairs_batch_inplace` | Batched in-place interpolation |
| `pairwise_reduce` | Kernel evaluation over pairs with `EqInput` weighting and `BindingOrder` → `Vec<F>` |
| `pairwise_reduce_fixed<const D>` | Const-generic reduce with stack-allocated `[F; D]` |
| `pairwise_reduce_multi` | Single-pass multi-kernel evaluation over shared inputs |
| `product_table` | Multiplicative product table over Boolean hypercube (eq polynomial) |
| `sum` | Sum all buffer elements → scalar |
| `dot_product` | Inner product of two buffers → scalar |
| `scale` | In-place scalar multiplication |
| `add` / `sub` | Element-wise addition/subtraction → new buffer |
| `accumulate` | Fused multiply-add: `buf[i] += scalar * other[i]` |
| `accumulate_weighted` | Multi-input weighted accumulation: `buf[i] += Σ_k s_k * inputs_k[i]` |
| `scale_batch` | In-place scalar multiplication across multiple buffers |

## Dependency Position

```
jolt-field ─► jolt-compute
jolt-ir    ─►
```

Used by `jolt-sumcheck` and `jolt-zkvm`.

## Benchmarks

BN254 Fr on Apple Silicon (M-series). Run with `cargo bench -p jolt-compute`.

### `interpolate_pairs` — pairwise linear interpolation

| Size | Fr→Fr | u8→Fr | `Polynomial::bind` (ref) |
|------|-------|-------|--------------------------|
| 2^16 | 99 Melem/s | 172 Melem/s | 117 Melem/s |
| 2^18 | 94 Melem/s | 175 Melem/s | 94 Melem/s |
| 2^20 | 94 Melem/s | 138 Melem/s | 92 Melem/s |

### `interpolate_pairs_inplace` — in-place vs allocating bind

| Size | Allocating | InPlace LowToHigh | InPlace HighToLow |
|------|------------|--------------------|--------------------|
| 2^16 | 45 Melem/s | 93 Melem/s | 106 Melem/s |
| 2^18 | 134 Melem/s | 119 Melem/s | 133 Melem/s |
| 2^20 | 133 Melem/s | 123 Melem/s | 103 Melem/s |

At small sizes in-place saves allocation overhead (2× faster). At large sizes both
converge as the interpolation cost dominates. HighToLow is faster because
`split_at_mut` enables true zero-allocation parallel in-place (no aliasing).

### `interpolate_pairs_batch` — batched bind across multiple polynomials

| Configuration | Throughput |
|---------------|------------|
| 8 × 2^18 | 123 Melem/s |
| 32 × 2^16 | 138 Melem/s |
| 128 × 2^14 | 137 Melem/s |

### `pairwise_reduce` — weighted kernel evaluation over pairs

| D (inputs) | 2^16 pairs | 2^18 pairs | 2^20 pairs |
|------------|------------|------------|------------|
| 4 | 9.6 Mpair/s | 8.7 Mpair/s | 8.5 Mpair/s |
| 8 | 4.9 Mpair/s | 4.2 Mpair/s | 4.2 Mpair/s |
| 16 | 2.5 Mpair/s | 2.1 Mpair/s | 2.1 Mpair/s |

### `pairwise_reduce_fixed` — const-generic D vs dynamic

| D | Size | Dynamic | Fixed | Speedup |
|---|------|---------|-------|---------|
| 4 | 2^18 | 5.0 Mpair/s | 6.9 Mpair/s | 1.4× |
| 4 | 2^20 | 7.5 Mpair/s | 8.2 Mpair/s | 1.1× |
| 16 | 2^16 | 394 Kpair/s | 523 Kpair/s | 1.3× |
| 16 | 2^20 | 537 Kpair/s | 571 Kpair/s | 1.1× |

Fixed-D helps most at D=4 (medium sizes) and D=16 (stack arrays replace heap).
At D=8 the benefit is marginal — the dynamic version already amortizes Vec
allocation across the Rayon fold.

### `pairwise_reduce` with `EqInput::Tensor` — split-eq factored weights vs flat

| Split (outer+inner) | Flat | Tensor | Tensor Fixed |
|---------------------|------|--------|--------------|
| 5+13 (~2^18 pairs) | 8.9 Mpair/s | 8.4 Mpair/s | 8.6 Mpair/s |
| 9+9 (~2^18 pairs) | 8.6 Mpair/s | 9.0 Mpair/s | 9.0 Mpair/s |
| 13+5 (~2^18 pairs) | 9.0 Mpair/s | 8.9 Mpair/s | 9.0 Mpair/s |

Tensor matches flat throughput while avoiding O(outer×inner) weight
materialization. At balanced splits (9+9) tensor is slightly faster due to
better cache locality (inner weights stay in L1).

### `pairwise_reduce_multi` — multi-kernel single pass vs individual

| Size | Individual 2× | Multi 2× | Speedup |
|------|---------------|----------|---------|
| 2^16 | 4.0 Mpair/s | 3.9 Mpair/s | ~1× |
| 2^18 | 4.3 Mpair/s | 4.5 Mpair/s | 1.04× |
| 2^20 | 4.5 Mpair/s | 4.2 Mpair/s | ~1× |

On CPU the multi-kernel path is comparable to individual calls — Rayon keeps
data warm in L1/L2 between calls. The benefit will be larger on GPU where
kernel launch overhead and memory bus bandwidth dominate.

### `product_table` — eq-polynomial evaluation table

| Variables | `product_table` | `EqPolynomial` (ref) |
|-----------|-----------------|----------------------|
| 16 (64K) | 65 Melem/s | 71 Melem/s |
| 20 (1M) | 136 Melem/s | 91 Melem/s |
| 24 (16M) | 140 Melem/s | 170 Melem/s |

## License

MIT
