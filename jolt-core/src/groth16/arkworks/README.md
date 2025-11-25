# Groth16 Transpilation for Stage 1 Verification (Arkworks)

This module implements a Groth16 circuit using **arkworks** that verifies Jolt's Stage 1 (Spartan outer sumcheck), enabling EVM-efficient on-chain verification.

> **Note**: This is the arkworks-based approach located at `jolt-core/src/groth16/arkworks/`. Future implementations (Circom, Halo2, SP1) will be added as sibling modules under `groth16/`.

## Background

Jolt's native verifier uses Dory polynomial commitments and interactive sumcheck protocols. While efficient for native execution, these are expensive on-chain (~1.2B gas). By transpiling Stage 1 into a Groth16 circuit, we achieve:

- **Constant-size proof**: 2 G1 + 1 G2 elements (~256 bytes)
- **O(1) verification**: Single pairing check (~200k gas on EVM)

## Running the Example

```bash
# Run fibonacci example with Groth16 verification
RUST_LOG=info cargo run --release -p fibonacci --features groth16 -- --groth16
```

Expected output:
```
INFO  fibonacci > === Running Groth16 Verification ===
INFO  fibonacci > Extracting circuit data from Stage 1 proof...
INFO  fibonacci > Circuit data extracted:
INFO  fibonacci >   - tau challenges: 10
INFO  fibonacci >   - r0 challenge: present
INFO  fibonacci >   - sumcheck challenges: 9
INFO  fibonacci >   - uni-skip poly coeffs: 28
INFO  fibonacci >   - sumcheck round polys: 9
INFO  fibonacci >   - r1cs input evals: 30
INFO  fibonacci >   - trace length: 256
INFO  fibonacci >   - total public inputs: 102
INFO  fibonacci > Running Groth16 setup...
INFO  fibonacci > Groth16 setup completed in 0.12s
INFO  fibonacci > Generating Groth16 proof...
INFO  fibonacci > Groth16 proof generated in 0.20s
INFO  fibonacci > Verifying Groth16 proof...
INFO  fibonacci > === Groth16 Results ===
INFO  fibonacci >   Setup time:   0.12s
INFO  fibonacci >   Prove time:   0.20s
INFO  fibonacci >   Verify time:  0.016000s
INFO  fibonacci >   Verification: PASSED ✓
```

## Architecture

```
┌─────────────────────┐
│  Jolt Full Proof    │
│  (Stage1OnlyProof)  │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Witness Extractor  │  ← Replays Fiat-Shamir transcript
│  (Stage1CircuitData)│  ← Decompresses sumcheck polynomials
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   Stage1Circuit     │  ← R1CS constraints (arkworks)
│   (ConstraintSynth) │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   Groth16 Proof     │  ← 2 G1 + 1 G2 (~256 bytes)
│   (ark-groth16)     │
└─────────────────────┘
```

## Module Structure

- `mod.rs` - Module exports
- `circuit.rs` - `Stage1Circuit` implementing `ConstraintSynthesizer`
- `gadgets.rs` - Constraint gadgets (polynomial evaluation, power sum check)
- `witness.rs` - `Stage1CircuitData` witness extraction from Jolt proof
- `tests.rs` - Integration tests

## What the Circuit Verifies

### 1. Uni-skip First Round (Power Sum Check)

Instead of the standard sumcheck first round over {0,1}, Jolt uses a high-degree polynomial over a symmetric domain D = {-4, -3, ..., 5}. The verification check is:

```
Σ_j a_j * S_j == 0
```

Where:
- `a_j` are the polynomial coefficients (28 total)
- `S_j = Σ_{t∈D} t^j` are precomputed power sums

### 2. Sumcheck Round Consistency

For each subsequent round i:
```
g_i(0) + g_i(1) == claim_{i-1}
claim_i = g_i(r_i)
```

Where `r_i` is the Fiat-Shamir challenge for round i.

## Current Limitations

The final claim check is not yet implemented. Complete Stage 1 verification requires:

- **Lagrange kernel**: `L(τ_high, r0)` evaluation
- **eq polynomial**: `eq(τ_low, r_tail)` multilinear equality
- **Inner sum product**: `A(rx, r) * B(rx, r)` from R1CS evaluations

For now, polynomial opening verification is handled by Dory commitments in the native Jolt verifier.

## Running Tests

```bash
# Run Groth16 integration tests
cargo test -p jolt-core --features groth16-stable groth16

# Run with output
cargo test -p jolt-core --features groth16-stable groth16 -- --nocapture
```

## Dependencies

This implementation uses the arkworks ecosystem:
- `ark-groth16` - Groth16 proving system
- `ark-bn254` - BN254 pairing curve (EVM-compatible)
- `ark-r1cs-std` - R1CS constraint gadgets
- `ark-relations` - R1CS constraint system

Enable with either:
- `groth16-stable` - Uses stable arkworks releases (v0.5)
- `groth16-git` - Uses git dependencies (for bleeding edge)

## Why Arkworks?

Arkworks provides a mature, well-audited Rust implementation of Groth16 with:
- Native Rust performance
- Direct integration with Jolt's existing arkworks dependencies (ark-bn254, ark-ff)
- Flexible R1CS constraint API via `ConstraintSynthesizer` trait
- Built-in support for EVM-compatible curves (BN254)

Alternative approaches being considered:
- **Circom**: DSL-based, widely used, excellent tooling (snarkjs, circomlib)
- **Halo2**: PLONKish arithmetization, no trusted setup required
- **SP1**: zkVM-based approach for recursive verification
