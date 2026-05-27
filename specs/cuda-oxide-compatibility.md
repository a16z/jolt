# Spec: cuda-oxide Compatibility

| Field     | Value      |
| --------- | ---------- |
| Author(s) | TBD        |
| Created   | 2026-05-26 |
| Status    | proposed   |
| PR        | TBD        |

## Summary

Add an opt-in CUDA prover acceleration backend using [NVlabs/cuda-oxide](https://github.com/NVlabs/cuda-oxide).
The compatibility target is Jolt prover hot paths, not guest execution and not compiling the whole Jolt workspace with `cargo oxide`.

The backend must live behind an explicit feature and a CUDA-specific build path. Jolt's normal Rust toolchain, CI, and CPU/rayon prover path must keep working without CUDA, LLVM NVPTX, clang, `rustc-dev`, or NVIDIA drivers installed.

Initial hardware target: rented NVIDIA A100. Treat it as an Ampere `sm_80` device with compute capability 8.0 and either 40 GB HBM2 or 80 GB HBM2e memory, depending on the rented SKU.

## Intent

### Goal

Introduce a cuda-oxide backend that can run selected prover kernels on NVIDIA GPUs while preserving byte-identical proof semantics.
The first real target is BN254 scalar-field vector work in Dory, because it has a narrow data shape, existing Montgomery limb constants in `jolt-field`, and low coupling to elliptic-curve code.

Stage 1 should implement and integrate:

1. A cuda-oxide package with its own `rust-toolchain.toml`, built only with `cargo oxide`.
2. Device-side BN254 Fr Montgomery arithmetic over fixed `[u32; 8]` limbs.
3. A fused Dory field-vector fold kernel for `left[i] = left[i] * scalar + right[i]`.
4. A host wrapper that transfers limb buffers with `cuda-core::DeviceBuffer`, launches the typed `#[cuda_module]` kernel, and copies results back.
5. A `jolt-dory` integration point that uses the CUDA kernel only when the CUDA feature is enabled, the input length clears a benchmarked threshold, and device initialization succeeds.

Stage 2 should add G1 row MSM acceleration for streaming Dory commitments.
This requires a pure device-side BN254 G1 implementation rather than calling arkworks from kernels.

### Non-Goals

1. Proving CUDA/PTX guest programs in Jolt. Jolt's guest ISA remains RV64IMAC.
2. Making `cargo oxide build` compile the whole Jolt workspace.
3. Adding cuda-oxide dependencies to default workspace resolution.
4. Calling arkworks, dory, rayon, or allocation-heavy Rust code from CUDA kernels.
5. Removing the CPU path. CUDA must be an optional accelerator.

## Constraints

cuda-oxide currently requires a separate environment from Jolt's normal root toolchain: Linux, NVIDIA GPU, CUDA Toolkit 12.x+, LLVM/Clang 21+, and a pinned nightly with `rust-src` and `rustc-dev`.
Jolt currently pins `rust-toolchain.toml` to stable `1.95`.
The CUDA backend therefore must be isolated so normal development and CI do not inherit cuda-oxide's toolchain or system dependencies.

The A100 path should compile kernels for `sm_80`.
Use `CUDA_OXIDE_TARGET=sm_80` or `cargo oxide --arch sm_80` for cross-builds; `cargo oxide run` may auto-detect the local device, but explicit targeting keeps benchmark artifacts comparable.
The Stage 1 BN254 field-fold kernel should need only baseline 32-bit integer arithmetic and global-memory bandwidth.

Device code is also a restricted Rust environment.
Kernels should use `core`, fixed-size arrays, slices, and plain structs.
No `std`, no heap allocation, no panics on expected paths, and no dependency graph that pulls host-only crates into device code.

## Crate Layout

Use a standalone package under `gpu/cuda-oxide/` with its own `[workspace]` table and `rust-toolchain.toml`.
Do not add it to the root workspace until cuda-oxide can be resolved and checked on non-CUDA machines without fetching or building CUDA host bindings.

Recommended layout:

```text
gpu/cuda-oxide/
  Cargo.toml
  rust-toolchain.toml
  src/
    lib.rs
    fr.rs
    kernels.rs
    host.rs
```

`fr.rs` owns device-compatible BN254 Fr arithmetic.
It should duplicate only the minimum constants needed by kernels, generated from `jolt-field::MontgomeryConstants` or checked against those constants in tests.

`kernels.rs` owns `#[cuda_module]` kernels.
`host.rs` owns `CudaContext`, `DeviceBuffer`, launch configuration, error mapping, and threshold decisions.

## Integration Points

### Dory Field Fold

Current CPU path:

- `jolt-core/src/poly/commitment/dory/jolt_dory_routines.rs`
- `fold_field_vectors(left, right, scalar)`
- `JoltG1Routines::fold_field_vectors`
- `JoltG2Routines::fold_field_vectors`

Shared-crate path:

- `crates/jolt-dory/src/scheme.rs`
- Dory proof rounds via `DoryRoutines::fold_field_vectors`

The CUDA backend should expose a synchronous API equivalent to:

```rust
pub fn fold_field_vectors_cuda(
    left: &mut [Fr],
    right: &[Fr],
    scalar: &Fr,
) -> Result<(), CudaBackendError>;
```

The caller must fall back to the existing CPU implementation on initialization failure, unsupported platform, or sub-threshold input length.
Fallback must be explicit at the callsite so benchmarks can distinguish CPU and GPU runs.

### Streaming Row MSM

Current row MSM path:

- `crates/jolt-dory/src/streaming.rs`
- `DoryScheme::feed`
- `G1Routines::msm(g1_bases, scalars)`

This is the first elliptic-curve CUDA target.
It should not be attempted until Stage 1 has correctness and transfer-overhead benchmarks.

The device representation should use affine bases and projective bucket accumulators.
A Pippenger implementation should be benchmarked against the existing CPU GLV/rayon path before integration.

## Feature Policy

Root workspace defaults must not change.
Do not add cuda-oxide git dependencies to root `[workspace.dependencies]` until Cargo can resolve them without CUDA-specific host builds.

The eventual feature shape should be:

```toml
[features]
cuda-oxide = []
```

The feature should enable only Jolt-side callsites and host wrappers.
The standalone cuda-oxide package remains the owner of cuda-oxide crate dependencies and kernel compilation.

## Correctness Requirements

1. CPU and CUDA outputs match byte-for-byte for every tested Fr vector length.
2. Inputs and outputs remain in Montgomery form; no canonical-byte conversion in hot loops.
3. In-place aliasing rules are explicit: `left` may be mutated, `right` must not alias `left` unless the kernel is written and tested for that case.
4. CUDA fallback must never change transcript order, proof bytes, or verifier behavior.
5. Any CUDA path used in ZK mode must pass the same deterministic proof checks as the CPU path.

## Validation

Stage 1 local CUDA validation:

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv
cargo oxide doctor
cd gpu/cuda-oxide
CUDA_OXIDE_TARGET=sm_80 cargo oxide run
```

Jolt correctness after integration:

```bash
cargo nextest run -p jolt-core muldiv --cargo-quiet --features host
cargo nextest run -p jolt-core muldiv --cargo-quiet --features host,zk
```

Performance validation:

- Benchmark `fold_field_vectors` over powers of two from 2^10 through the largest Dory fold size used in representative proofs.
- Report host-to-device, kernel, device-to-host, and total wall time separately.
- Pick the CUDA threshold from measured crossover, not a constant guessed in code.

## Acceptance Criteria

- `gpu/cuda-oxide` builds and runs under `cargo oxide run` on a supported CUDA host.
- A100 validation reports compute capability 8.0 and runs kernels compiled for `sm_80`.
- Default `cargo clippy --all --features host -q --all-targets -- -D warnings` does not require cuda-oxide, CUDA, clang, LLVM NVPTX, or NVIDIA drivers.
- `fold_field_vectors_cuda` matches the CPU fold implementation across randomized vectors, edge values, zero/one scalar, and non-power-of-two tails.
- Integrated `muldiv` passes in both `host` and `host,zk` with CUDA enabled and with CUDA unavailable.
- Benchmarks prove a positive crossover for at least one Dory field-fold size before enabling automatic CUDA dispatch.

## Follow-Up Work

1. CUDA G1 arithmetic and row MSM.
2. Batched row MSM for streaming commitments.
3. Field-polynomial kernels for large `bind_parallel` and equality-polynomial table generation.
4. Async cuda-oxide execution using `cuda-async` once multiple independent prover kernels exist.
5. CI job in a CUDA container, separate from normal CPU CI.
