# WebGPU Backend

## What

A new `ComputeBackend` implementation targeting WebGPU (via `wgpu`), enabling browser-based and cross-platform GPU-accelerated proving.

## Why

WebGPU is the emerging standard for GPU compute in browsers and is also available natively via `wgpu-rs`. A WebGPU backend enables:
- **Browser proving:** users prove in-browser without installing anything
- **Cross-platform GPU:** single backend works on Metal (macOS), Vulkan (Linux/Windows), and D3D12 (Windows), falling back gracefully
- **Wasm target:** `wgpu` compiles to Wasm+WebGPU, enabling the full prover to run in a web worker

## Scope

**New crate: `jolt-webgpu`**

Implements `ComputeBackend` for `WgpuBackend`:
- `compile(spec: &KernelSpec) → WgpuKernel` — generates WGSL compute shaders from the KernelSpec's composition formula and iteration pattern
- `execute(kernel, inputs, output)` — dispatches GPU compute, handles buffer management
- `upload(data) → GpuBuffer` / `download(buf) → Vec<F>` — host↔device transfer

**Shader generation:**
The `KernelSpec` describes the computation abstractly (formula, iteration, binding order). The Metal backend already does MSL codegen from this. The WebGPU backend does the same but targets WGSL.

Key differences from Metal:
- WGSL has no SIMD intrinsics — vectorization is implicit via the GPU
- Workgroup size / dispatch dimensions need tuning per GPU vendor
- No shared memory atomics in WebGPU 1.0 — reduction patterns must use workgroup barriers
- 32-bit integers only in WGSL — BN254 field arithmetic needs multi-limb representation (same as Metal)

**Field arithmetic in WGSL:**
BN254's 254-bit field elements need 8×32-bit limb representation with Montgomery multiplication. This is the core performance-critical shader code. Can potentially share the limb arithmetic implementation with the Metal backend (both use 32-bit limbs).

## How It Fits

Plugs in exactly where `jolt-metal` and `jolt-cpu` plug in:

```
jolt-compute (ComputeBackend trait)
     |
     +-- jolt-cpu
     +-- jolt-metal
     +-- jolt-webgpu   ← new
     +-- jolt-hybrid
```

The hybrid combinator (`jolt-hybrid`) can also work with WebGPU: small kernels on CPU, large kernels on GPU.

The prover runtime (`jolt-zkvm`) is backend-generic — it works with WebGPU without changes.

## Dependencies

None — this is an independent track. Only requires the `ComputeBackend` trait (stable in `jolt-compute`).

## Unblocks

- Browser-based proving
- Cross-platform GPU proving without Metal/CUDA specificity
- Wasm compilation target for the full prover

## Open Questions

- **Performance ceiling:** WebGPU adds abstraction overhead vs. native Metal/CUDA. For sumcheck kernels (relatively small, many dispatches), will the overhead dominate? Need benchmarks.
- **Memory model:** WebGPU buffer mapping is async. The current `ComputeBackend` trait assumes synchronous `upload`/`download`. Does the trait need async variants, or do we block on the GPU?
- **wgpu version stability:** wgpu moves fast. Pin a specific version and abstract behind the trait to isolate churn.
- **Fallback:** If WebGPU is unavailable (old browser, no GPU), should `jolt-webgpu` gracefully degrade to CPU, or should the caller use `jolt-hybrid` with `jolt-cpu` as fallback?
- **Shared shader code:** Can the WGSL field arithmetic be generated from the same source as the MSL field arithmetic (e.g., a shared IR that emits both)?
