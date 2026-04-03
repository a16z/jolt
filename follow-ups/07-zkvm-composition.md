# jolt-zkvm Composition

## What

Make the jolt-zkvm runtime a composable pipeline where protocols chain together naturally. Each layer (inner proof, BlindFold, extended P/V, Spartan-HyperKZG wrapper, Groth16) is a protocol that runs on the same runtime, and users compose them freely — choosing any backend, toggling ZK, selecting which wrapping layers to apply.

The key insight: the jolt runtime already executes any compiled Module. Composition is just running multiple Modules in sequence, where each Module's inputs come from the previous Module's outputs. The runtime doesn't need special composition logic — it just runs protocols.

## Why

Different deployment targets need different pipeline configurations:

| Use Case | Pipeline |
|----------|----------|
| Testing / benchmarking | inner proof only |
| Off-chain privacy | inner + BlindFold |
| On-chain (Dory) | inner + extended P/V + Spartan-HyperKZG wrapper + Groth16 |
| On-chain (Hachi) | inner + Spartan-HyperKZG wrapper + Groth16 |
| On-chain (no privacy) | inner + wrapper + Groth16 |
| Aggregation | inner + wrapper + wrapper (recursive) |

The user should be able to select any combination without the runtime caring about the specifics. Each layer is a Module; the runtime runs Modules.

## Scope

**Composition model:**

Every protocol in the pipeline is a Module that the runtime executes. The output of one protocol feeds the input of the next:

```
inner Module  →  execute()  →  JoltProof
                                   ↓  (proof is witness data for next layer)
zk Module     →  execute()  →  BlindFoldProof        [optional]
                                   ↓
aux Module    →  execute()  →  ExtendedProof          [optional, Dory-specific]
                                   ↓
wrapper Module → execute()  →  WrapperProof           [optional]
                                   ↓
                          gnark codegen  →  Groth16    [optional, external]
```

Each arrow is "extract witness from previous proof, build buffers, execute next Module." The runtime's `execute()` function is the same at every layer — it doesn't know whether it's proving a RISC-V trace or proving that a proof was verified.

**R1CS chaining:**

The Spartan-HyperKZG wrapper (05) takes its R1CS from the R1CS-from-Module pass (02) applied to the inner Module. But the wrapper itself produces a Module, and that Module can also be fed through R1CS-from-Module to get *the wrapper's* R1CS. This wrapper R1CS is what gnark consumes.

```
inner Module  →  R1CS-from-Module  →  R1CS₁  (wrapper proves this)
wrapper Module  →  R1CS-from-Module  →  R1CS₂  (gnark proves this)
```

Each layer's R1CS is mechanically derived from its Module. The R1CS-from-Module pass doesn't care what protocol the Module represents — it just encodes the verifier schedule as constraints. This is how wrapping composes: the system can wrap a wrapper, or wrap a wrapper of a wrapper.

**Backend flexibility:**

Each layer can use a different backend and PCS:

```rust
// Inner: CPU + Dory
let inner_proof = prove::<CpuBackend, Fr, T, Dory>(&inner_module, ...);

// Wrapper: GPU + HyperKZG
let wrapper_proof = prove::<MetalBackend, Fr, T, HyperKZG>(&wrapper_module, ...);
```

Or with Hachi:
```rust
// Inner: CPU + Hachi
let inner_proof = prove::<CpuBackend, HachiField, T, Hachi>(&inner_module, ...);

// Wrapper: CPU + HyperKZG (still BN254 for on-chain)
let wrapper_proof = prove::<CpuBackend, Fr, T, HyperKZG>(&wrapper_module, ...);
```

The field and PCS can even change between layers (Hachi field for inner, BN254 for wrapper targeting on-chain).

**Configuration:**

A `PipelineConfig` describes the composition:

```rust
struct PipelineConfig {
    backend: BackendChoice,          // CPU, Metal, CUDA, WebGPU, Hybrid
    pcs: PcsChoice,                  // Dory, HyperKZG, Hachi
    zk: bool,                        // BlindFold layer
    extended_pv: bool,               // auxiliary Dory cert (only with Dory PCS)
    wrapper: Option<WrapperConfig>,  // Spartan-HyperKZG wrapping
    groth16: bool,                   // final Groth16 layer (gnark)
}
```

The pipeline orchestrator validates the config (e.g., extended P/V only makes sense with Dory) and chains the appropriate Modules.

## How It Fits

`jolt-zkvm` already owns `execute()` and the prover entry points. Composition is a thin orchestration layer on top:

1. Compile the inner Module (or load from disk)
2. Execute inner proof
3. For each subsequent layer: derive R1CS from previous Module, compile wrapper Protocol, execute
4. Optionally emit gnark codegen for the final layer

Each layer's crate (`jolt-blindfold`, `jolt-hyperkzg`, `jolt-hyrax`, `jolt-wrapper`) provides its Module definition and witness extraction logic. `jolt-zkvm` chains them.

## Dependencies

- All prior follow-ups: compiler (01), R1CS (02), BlindFold (04), wrapper (05), gnark (06), extended P/V (14)

## Unblocks

- End-to-end on-chain verification with any PCS/backend combination
- SDK integration (`jolt-sdk` exposes `PipelineConfig` to users)
- Proof aggregation (recursive wrapping)

## Open Questions

- **Witness extraction:** Each layer needs to extract a witness from the previous proof. This is protocol-specific (the wrapper needs to extract transcript state, challenges, evaluations from the inner proof). Should witness extraction be a trait that each layer implements, or is it simple enough to hardcode per layer?
- **Proof types:** The final proof type depends on the pipeline depth. A layered enum? A generic `ComposedProof<Inner, Outer>`? Or just the outermost proof (since inner proofs are consumed and not needed by the verifier)?
- **Proving time budget:** What's the target end-to-end time for sha256? Each layer adds overhead — how much is acceptable?
- **Verification key composition:** The on-chain verifier needs the Groth16 VK, which embeds the wrapper circuit, which embeds the inner protocol's R1CS. Is the VK generated once per protocol version and deployed as a contract constant?
- **Field bridging:** When the inner proof uses Hachi (lattice field) but the wrapper uses BN254 (for on-chain), how do field elements cross the boundary? The wrapper's R1CS witness includes inner proof evaluations — do they need field conversion?
