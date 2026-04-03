# Hash BlindFold — ZK Without Homomorphism

## What

A fundamentally different approach to making Jolt zero-knowledge, using hash-based commitments that lack additive homomorphism. This is the ZK layer for Hash Jolt (10).

## Why

Hash commitments (Merkle trees, hash chains) are the simplest and most battle-tested post-quantum commitment scheme. But they're not additively homomorphic — you can't add two commitments and get the commitment to the sum. This means BlindFold's Nova folding (which relies on homomorphism) doesn't work.

Hash BlindFold needs a different protocol for achieving zero-knowledge over sumcheck proofs.

## Scope

This is primarily a research track: designing a ZK sumcheck protocol that works with non-homomorphic commitments.

**Possible approaches:**

**1. Masking polynomials:**
Instead of committing to round polynomial coefficients and folding, the prover adds random masking polynomials to each sumcheck polynomial before evaluation. The sumcheck is over (f + mask), where mask sums to zero. The verifier checks the masked sumcheck; ZK follows from the mask hiding f's coefficients.

This requires:
- Committing to masking polynomials (hash commitment works)
- Proving that the mask sums to zero without revealing it
- Adjusting the sumcheck protocol to handle the masking

**2. GKR-style ZK:**
Use the GKR protocol's ZK variant, where the prover commits to a random polynomial and uses it to mask the entire sumcheck interaction. This has been formalized in the literature (e.g., Libra, zkGKR).

**3. MPC-in-the-head:**
Convert the sumcheck verifier into an MPC protocol, then use the "MPC-in-the-head" paradigm (as in ZKBoo, Limbo). The prover simulates multiple parties running the MPC protocol, commits to their views with Merkle trees, and the verifier checks a random subset.

**4. IOPs with hash queries:**
Model the sumcheck proof as an IOP (Interactive Oracle Proof) and compile it with a hash-based compiler (FRI-style). The prover commits to round polynomial evaluations via Merkle trees, and the verifier queries random positions.

**What this is NOT:**
- It's not "BlindFold with hash commitments" — the protocol structure changes fundamentally
- It's not a STARK wrapper around Jolt — it's ZK at the sumcheck level

## How It Fits

The interface to `jolt-zkvm` would be the same: a ZK layer that takes a sumcheck proof and makes it zero-knowledge. The internal protocol is different, but the API is:

```rust
// Standard BlindFold
zk_prove(inner_proof, module, transcript) → ZkProof

// Hash BlindFold  
zk_prove(inner_proof, module, transcript) → ZkProof
```

The ZK layer is feature-gated. Hash BlindFold would be a different feature flag (e.g., `zk-hash` vs. `zk`).

Whether this lives in `jolt-blindfold` (as an alternative mode) or in a separate crate (e.g., `jolt-hash-zk`) depends on how much code is shared.

## Dependencies

- BlindFold rewrite (04) — to understand the interface and what needs to be preserved
- Hash Jolt (10) — the PCS layer that this pairs with

## Unblocks

- Post-quantum ZK Jolt with hash PCS (the simplest post-quantum configuration: hash PCS + hash ZK)
- Potentially the fastest proving (no group operations at all — pure hashing and field arithmetic)

## Open Questions

- **Which approach?** The four approaches above have very different performance/proof-size tradeoffs. Need research to determine which is viable for Jolt's specific sumcheck structure (many stages, varying degree, many polynomials per stage).
- **Proof size:** Hash-based ZK typically has larger proofs (Merkle paths, multiple party views). What's the overhead over non-ZK proofs? Is it acceptable?
- **Prover overhead:** Masking approaches require additional polynomial evaluations. MPC-in-the-head requires simulating multiple parties. What's the concrete prover time overhead?
- **Compatibility with the verifier schedule:** The verifier schedule encodes sumcheck verification steps. Does hash BlindFold change what the verifier does, or does it only change what's committed/revealed?
- **Is this even necessary?** If lattice BlindFold (12) works well, it provides post-quantum ZK with homomorphism (same protocol as Pedersen BlindFold). Hash BlindFold is only needed if we want ZK without ANY algebraic structure — is there a compelling use case beyond "minimizing assumptions"?
