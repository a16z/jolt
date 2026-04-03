# Wrapper Protocol — Spartan-HyperKZG

## What

A Jolt protocol (`wrapper.jolt`) expressed in the Protocol IR, compiled by jolt-compiler, and executed by the jolt-zkvm runtime — just like the inner Jolt proof. It proves "I verified a Jolt proof correctly" by taking the R1CS derived from the inner Module (02) and proving it via Spartan with HyperKZG as the PCS. The wrapper is a native Jolt protocol, not external tooling.

This is distinct from the gnark Groth16 wrapper (06), which is Go codegen targeting a different proving system.

## Why

Jolt proofs with Dory are large. On-chain verification of the full proof is impractical. The wrapper compresses the proof by proving the verifier's computation via Spartan-HyperKZG, which has smaller proofs and cheaper verification (a few pairings).

Because Spartan is sumcheck-based, the wrapper can be expressed as a Protocol IR definition — the same IR, compiler, and runtime that runs the inner Jolt proof. This is recursive use of the architecture: the system proves proofs about itself.

## Scope

**The wrapper as a Protocol:**

The wrapper's Protocol IR defines a sumcheck-based proof of R1CS satisfaction. The R1CS comes from the Module-derived pass (02), encoding the inner Jolt verifier's computation (transcript simulation, sumcheck verification, claim formulas, PCS verification via primitives).

```
inner Module  →  R1CS-from-Module (02)  →  R1CS constraints
                                                ↓
                              wrapper Protocol IR (Spartan over this R1CS)
                                                ↓
                              compile()  →  wrapper Module
                                                ↓
                              link() + prove()  →  WrapperProof
```

The wrapper prover:
1. Extracts the R1CS witness from the inner proof (transcript state, challenges, evaluations)
2. Commits to the witness via HyperKZG
3. Runs the Spartan sumcheck (outer + inner) as compiled by jolt-compiler
4. Outputs a wrapper proof: HyperKZG commitments + sumcheck proof + opening proof

The wrapper verifier:
1. Runs the compiled verifier schedule (same `verify()` function as the inner proof, different Module)
2. Checks HyperKZG openings

**What makes this native:**
- The wrapper Module is produced by `compile()` — same compiler passes (staging, batching, emit)
- The wrapper runs on `execute()` — same runtime, same `ComputeBackend`
- The wrapper's verifier schedule is interpreted by the same `verify()` function
- The only difference: different R1CS, different PCS type parameter (HyperKZG instead of Dory)

## How It Fits

```
protocol.jolt  →  Module  →  prove::<B, Fr, T, Dory>()  →  JoltProof
                     ↓
              R1CS-from-Module (02)
                     ↓
wrapper.jolt  →  Module  →  prove::<B, Fr, T, HyperKZG>()  →  WrapperProof
                                                                    ↓
                                                         gnark codegen (06) for on-chain
```

The `jolt-hyperkzg` crate provides the PCS. Everything else is existing infrastructure.

## Dependencies

- R1CS from Module (02) — the wrapper's R1CS IS the inner verifier encoded as constraints
- Compiler full protocol (01) — the inner Module must be complete
- `jolt-hyperkzg` — PCS for the wrapper

## Unblocks

- Gnark codegen (06) — gnark wraps the wrapper verifier (small circuit) into Groth16
- jolt-zkvm composition (07)
- On-chain verification

## Open Questions

- The inner PCS verification (Dory pairings) is the most expensive part of the wrapper's R1CS. This is where primitives (02) matter most — the Dory pairing check constraints dominate the wrapper circuit size. Can we amortize or batch pairing checks?
- HyperKZG requires a trusted setup (universal SRS). What size? How does it relate to the wrapper's R1CS size?
- Can the wrapper itself be wrapped (recursive wrapping) for proof aggregation?
- Is there a benefit to expressing Spartan's outer/inner sumcheck as Protocol IR vertices, or is Spartan simple enough to hardcode in the wrapper's Protocol definition?
