# Follow-ups

Scoping documents for work after the jolt-zkvm pipeline is fully working for the complete Jolt protocol. Each document covers what the track is, why it matters, how it interfaces with the current architecture, and open design questions.

## Dependency Graph

```
01 Compiler Full Protocol ──┬──→ 02 R1CS from Module ──┬──→ 04 BlindFold Rewrite ──→ 07 Composition
                            │                          │                              ↑
                            │                          ├──→ 05 Wrapper Protocol ──────┤
                            │                          │         ↓                    │
                            │                          │    06 Gnark Codegen ─────────┘
                            │                          │
                            └──→ 14 Extended P/V ──────┘
                            
03 Kernel Optimization                     (parallel, independent)
08 WebGPU Backend                          (parallel, independent)
09 CUDA Port                               (parallel, independent)
10 Hash Jolt                               (research, independent)
11 Hachi (Lattice Jolt)                    (research, independent)
12 Lattice BlindFold                       (depends on 04 + 11)
13 Hash BlindFold                          (research, depends on 04 + 10)
```

## Critical Path

01 → 02 → 04/05 → 07 (compiler → R1CS derivation → ZK + wrapper → composition)

## Documents

### Core Pipeline
- [01 — Compiler Full Protocol](01-compiler-full-protocol.md) — express all 7 Jolt stages in Protocol IR, replace hand-coded Module
- [02 — R1CS from Module](02-r1cs-from-module.md) — generic Module → R1CS pass for BlindFold and wrapper
- [03 — Kernel Optimization](03-kernel-optimization.md) — CPU/Metal/GPU kernel performance

### ZK and Wrapping
- [04 — BlindFold Rewrite](04-blindfold-rewrite.md) — generic ZK layer consuming Module-derived R1CS
- [05 — Spartan-HyperKZG Wrapper](05-wrapper-protocol.md) — native Jolt protocol (Protocol IR → compile → runtime) that wraps the inner proof
- [06 — Gnark Groth16 Wrapper](06-gnark-codegen.md) — Go codegen from wrapper Module for constant-size on-chain verification
- [07 — jolt-zkvm Composition](07-zkvm-composition.md) — orchestrate full pipeline: prove → ZK → wrap → Groth16

### Backends
- [08 — WebGPU Backend](08-webgpu-backend.md) — browser + cross-platform GPU proving via wgpu
- [09 — CUDA Port](09-cuda-port.md) — NVIDIA GPU backend for server-side proving

### Alternative Instantiations
- [10 — Hash Jolt](10-hash-jolt.md) — hash-based PCS (transparent, post-quantum), research track
- [11 — Hachi](11-hachi.md) — lattice field + PCS + BlindFold (post-quantum with homomorphism)
- [12 — Lattice BlindFold](12-lattice-blindfold.md) — BlindFold with lattice commitments (Nova folding preserved)
- [13 — Hash BlindFold](13-hash-blindfold.md) — ZK without homomorphism, research track

### Deferred
- [14 — Extended P/V](14-extended-pv.md) — auxiliary SNARK for Dory verification via curve 2-cycle (BN254/Grumpkin)
