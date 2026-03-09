# jolt-compute

Backend-agnostic compute device abstraction for the Jolt zkVM.

Part of the [Jolt](https://github.com/a16z/jolt) zkVM.

## Overview

This crate defines the `ComputeBackend` trait — a protocol-agnostic interface over typed buffer management and parallel primitives (pairwise interpolation, composition-reduce, product tables).

The `CpuBackend` implementation uses `Vec<T>` buffers with Rayon parallelism. After monomorphization every trait method compiles to a direct function call — identical codegen to hand-written Rayon code.

GPU backends (Metal, CUDA, WebGPU) live in separate crates and implement the same trait with device memory buffers and compiled shader kernels.

## Public API

### Traits

- **`Scalar`** — Marker trait for types storable in device buffers. Blanket-implemented for primitive integers, `bool`, and all `Field` types.

- **`ComputeBackend`** — Core device abstraction with associated types:
  - `Buffer<T: Scalar>` — typed device buffer handle
  - `CompiledKernel` — opaque compiled kernel for `pairwise_reduce`
  - Methods: `upload`, `download`, `alloc`, `len`, `is_empty`, `interpolate_pairs`, `interpolate_pairs_batch`, `pairwise_reduce`, `product_table`

### Backends

- **`CpuBackend`** — CPU backend using `Vec<T>` buffers and Rayon parallelism.
- **`CpuKernel`** — Compiled kernel type for the CPU backend.

## Dependency Position

```
jolt-field ─► jolt-compute
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

### `product_table` — eq-polynomial evaluation table

| Variables | `product_table` | `EqPolynomial` (ref) |
|-----------|-----------------|----------------------|
| 16 (64K) | 65 Melem/s | 71 Melem/s |
| 20 (1M) | 136 Melem/s | 91 Melem/s |
| 24 (16M) | 140 Melem/s | 170 Melem/s |

## Feature Flags

| Flag | Default | Description |
|------|---------|-------------|
| `parallel` | yes | Enables Rayon parallelism in `CpuBackend` |

## License

MIT
