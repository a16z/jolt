# CUDA Port

## What

Port the existing CUDA GPU backend to the new modular crate architecture as `jolt-cuda`, implementing the `ComputeBackend` trait.

## Why

CUDA is the highest-performance GPU compute platform, available on NVIDIA hardware which dominates data center and high-end proving infrastructure. The existing Jolt codebase likely has CUDA code in jolt-core that needs to be extracted and adapted to the `ComputeBackend` interface.

## Scope

**New crate: `jolt-cuda`**

Implements `ComputeBackend` for `CudaBackend`:
- `compile(spec: &KernelSpec) → CudaKernel` — generates CUDA kernels (PTX or NVRTC runtime compilation) from KernelSpec
- `execute(kernel, inputs, output)` — launches CUDA kernels with appropriate grid/block dimensions
- `upload` / `download` — cudaMemcpy host↔device

**Porting scope:**
1. **Extract** existing CUDA code from jolt-core (MSM, NTT, field arithmetic kernels)
2. **Adapt** to the `ComputeBackend` trait interface — the existing CUDA code may use different abstractions
3. **Kernel generation** from `KernelSpec` — either runtime compilation (NVRTC) or pre-compiled PTX with specialization
4. **Memory management** — CUDA memory pools, async transfers, stream-based execution

**CUDA-specific advantages over Metal/WebGPU:**
- Wider SIMD (warp = 32 threads vs. SIMD group = 32 on Metal)
- Shared memory with configurable L1/shared split
- Tensor cores for potential acceleration of large multiplications
- Mature async compute with CUDA streams and events
- `cuBLAS`/`cuFFT` for MSM and NTT if applicable

## How It Fits

```
jolt-compute (ComputeBackend trait)
     |
     +-- jolt-cpu
     +-- jolt-metal
     +-- jolt-webgpu
     +-- jolt-cuda     ← new
     +-- jolt-hybrid
```

Same pattern as all other backends. The hybrid combinator works with CUDA.

## Dependencies

None — independent track. Requires stable `ComputeBackend` trait.

## Unblocks

- High-performance server-side proving (data center deployments)
- Benchmarking against Metal backend
- Multi-GPU proving (CUDA has better multi-device APIs)

## Open Questions

- **Existing CUDA code:** What state is the CUDA code in jolt-core? Is it production-quality or experimental? How much can be reused vs. rewritten?
- **Runtime vs. ahead-of-time compilation:** NVRTC (runtime) is flexible but adds compilation overhead. Pre-compiled PTX requires knowing the SM architecture at build time. Which approach for `compile()`?
- **Multi-GPU:** Should `CudaBackend` support multiple GPUs transparently, or should that be a separate `MultiGpuBackend` combinator?
- **Memory pressure:** Large proofs can exceed GPU memory. The hybrid combinator helps (offload to CPU), but CUDA also has unified memory. Which strategy?
- **CI/CD:** CUDA requires NVIDIA hardware for testing. How to handle CI without GPU runners?
