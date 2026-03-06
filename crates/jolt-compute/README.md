# jolt-compute

Backend-agnostic compute device abstraction for the Jolt zkVM.

Part of the [Jolt](https://github.com/a16z/jolt) zkVM.

## Overview

This crate defines the `ComputeBackend` trait — a protocol-agnostic interface over typed buffer management and parallel primitives (pairwise interpolation, composition-reduce, product tables). All methods are named for what they compute, not what protocol uses them.

The `CpuBackend` implementation uses `Vec<T>` buffers with Rayon parallelism. After monomorphization every trait method compiles to a direct function call — identical codegen to hand-written Rayon code.

GPU backends (Metal, CUDA, WebGPU) live in separate crates and implement the same trait with device memory buffers and compiled shader kernels.

## Public API

### Traits

- **`Scalar`** — Marker trait for types storable in device buffers. Blanket-implemented for primitive integers, `bool`, and all `Field` types.

- **`ComputeBackend`** — Core device abstraction with associated types:
  - `Buffer<T: Scalar>` — typed device buffer handle
  - `CompiledKernel` — opaque compiled kernel for `pairwise_reduce`
  - Methods: `upload`, `download`, `alloc`, `len`, `is_empty`, `interpolate_pairs`, `pairwise_reduce`, `product_table`

### Backends

- **`CpuBackend`** — CPU backend using `Vec<T>` buffers and Rayon parallelism.
- **`CpuKernel`** — Compiled kernel type for the CPU backend.

## Dependency Position

```
jolt-field ─► jolt-compute
```

Used by `jolt-sumcheck` and `jolt-zkvm`.

## Feature Flags

| Flag | Default | Description |
|------|---------|-------------|
| `parallel` | yes | Enables Rayon parallelism in `CpuBackend` |

## License

MIT
