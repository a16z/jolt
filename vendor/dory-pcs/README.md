# Dory PCS

A high-performance, modular implementation of the Dory polynomial commitment scheme in Rust.

[![Crates.io](https://img.shields.io/crates/v/dory-pcs.svg)](https://crates.io/crates/dory-pcs)
[![Documentation](https://docs.rs/dory-pcs/badge.svg)](https://docs.rs/dory-pcs)

## Overview

Dory is a transparent polynomial commitment scheme with excellent asymptotic performance, based on the work of Jonathan Lee ([eprint 2020/1274](https://eprint.iacr.org/2020/1274)). This implementation provides a clean, modular architecture with strong performance characteristics and comprehensive test coverage.

**Key Features:**
- **Transparent setup**: No trusted setup ceremony required with optional disk persistence
- **Logarithmic proof size**: O(log n) group elements
- **Logarithmic verification**: O(log n) GT exps and 1 multi-pairing
- **Modular design**: Pluggable backends for curves and cryptographic primitives
- **Performance-optimized**: Vectorized operations, optional prepared point caching, and parallelization with Rayon
- **Flexible matrix layouts**: Supports both square and non-square matrices (nu ≤ sigma)
- **Homomorphic properties**: Commitment linearity enables proof aggregation
- **Zero-knowledge mode**: Toggleable hiding proofs

## Installation

Add `dory-pcs` to your `Cargo.toml`:

```toml
[dependencies]
dory-pcs = "0.4"
```

Or with specific features:

```toml
[dependencies]
dory-pcs = { version = "0.4", features = ["backends", "disk-persistence"] }
```

For maximum performance with all optimizations:

```toml
[dependencies]
dory-pcs = { version = "0.4", features = ["backends", "cache", "parallel", "disk-persistence"] }
```

## Architecture

### Core Modules

- **`primitives`** - Core trait abstractions
  - `arithmetic` - Field, group, and pairing curve traits
  - `poly` - Multilinear polynomial traits and operations
  - `transcript` - Fiat-Shamir transcript trait
  - `serialization` - Serialization abstractions

- **`setup`** - Transparent setup generation for prover and verifier

- **`evaluation_proof`** - Evaluation proof creation and verification

- **`reduce_and_fold`** - Inner product protocol (prover/verifier)

- **`messages`** - Protocol message structures (VMV, reduce rounds, scalar product)

- **`proof`** - Complete proof data structure

- **`error`** - Error types

### Backend Implementations

- **`backends::arkworks`** - Modular Arkworks backend with BN254 curve (requires `arkworks` feature)
  - Field wrappers (`ArkFr`)
  - Group wrappers (`ArkG1`, `ArkG2`, `ArkGT`)
  - Polynomial implementation
  - Optimized MSM routines
  - Blake2b transcript
  - Serialization bridge

## How It Works

Dory uses a two-tier (also known as AFGHO) homomorphic commitments:

1. **Polynomial Representation**: Coefficients are arranged as a 2^ν × 2^σ matrix (constraint: ν ≤ σ)
2. **Row Commitments** (Tier 1): Each row is committed using multi-scalar multiplication in G1
3. **Final Commitment** (Tier 2): Row commitments are combined via pairings with G2 generators
4. **Evaluation Proof**: Uses a VMV (Vector-Matrix-Vector) protocol with reduce-and-fold rounds

The protocol leverages the algebraic structure of bilinear pairings to achieve logarithmic proof sizes and verification times.

### Homomorphic Properties

Dory commitments are linearly homomorphic, meaning:
```
Com(r₁·P₁ + r₂·P₂ + ... + rₙ·Pₙ) = r₁·Com(P₁) + r₂·Com(P₂) + ... + rₙ·Com(Pₙ)
```

This property enables efficient proof aggregation and batch verification. See `examples/homomorphic.rs` for a demonstration.

## Usage

```rust
use dory_pcs::{setup, prove, verify, Transparent};
use dory_pcs::backends::arkworks::{
    BN254, G1Routines, G2Routines, ArkworksPolynomial, ArkFr, Blake2bTranscript
};
use dory_pcs::primitives::arithmetic::Field;
use dory_pcs::primitives::poly::Polynomial;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Generate setup for polynomials up to 2^10 coefficients
    let max_log_n = 10;
    let (prover_setup, verifier_setup) = setup::<BN254>(max_log_n);

    // 2. Create a polynomial with 256 coefficients (nu=4, sigma=4)
    let coefficients: Vec<ArkFr> = (0..256).map(|_| ArkFr::random()).collect();
    let polynomial = ArkworksPolynomial::new(coefficients);

    // 3. Define evaluation point (length = nu + sigma = 8)
    let point: Vec<ArkFr> = (0..8).map(|_| ArkFr::random()).collect();

    let nu = 4;     // log₂(rows) = 4 → 16 rows
    let sigma = 4;  // log₂(cols) = 4 → 16 columns

    // 4. Commit to polynomial to get tier-2 commitment and row commitments
    let (tier_2, row_commitments, commit_blind) = polynomial
        .commit::<BN254, Transparent, G1Routines>(nu, sigma, &prover_setup)?;

    // 5. Create evaluation proof using row commitments
    let mut prover_transcript = Blake2bTranscript::new(b"dory-example");
    let (proof, _) = prove::<_, BN254, G1Routines, G2Routines, _, _, Transparent>(
        &polynomial,
        &point,
        row_commitments,
        commit_blind,
        nu,
        sigma,
        &prover_setup,
        &mut prover_transcript,
    )?;

    // 6. Verify: check that the proof is valid
    let evaluation = polynomial.evaluate(&point);
    let mut verifier_transcript = Blake2bTranscript::new(b"dory-example");
    verify::<_, BN254, G1Routines, G2Routines, _>(
        tier_2,
        evaluation,
        &point,
        &proof,
        verifier_setup,
        &mut verifier_transcript,
    )?;

    println!("Proof verified successfully!");
    Ok(())
}
```

## Examples

The repository includes six comprehensive examples demonstrating different aspects of Dory:

1. **`basic_e2e`** - Standard end-to-end workflow with square matrix (nu=4, sigma=4)
   ```bash
   cargo run --example basic_e2e --features backends
   ```

2. **`homomorphic`** - Demonstrates homomorphic combination of 5 polynomials
   ```bash
   cargo run --example homomorphic --features backends
   ```

3. **`non_square`** - Non-square matrix layout (nu=3, sigma=4) demonstrating flexibility
   ```bash
   cargo run --example non_square --features backends
   ```

4. **`homomorphic_mixed_sizes`** - Combining polynomials with different matrix dimensions
   ```bash
   cargo run --example homomorphic_mixed_sizes --features backends
   ```

5. **`zk_e2e`** - Zero-knowledge end-to-end workflow with hiding proofs
   ```bash
   cargo run --example zk_e2e --features backends,zk
   ```

6. **`zk_statistical`** - Chi-squared uniformity and witness-independence tests for ZK proofs
   ```bash
   cargo run --release --example zk_statistical --features backends,zk,parallel
   ```

## Development Setup

After cloning the repository, install Git hooks to ensure code quality:

```bash
./scripts/install-hooks.sh
```

This installs a pre-commit hook that:
- Auto-formats code with `cargo fmt`
- Runs `cargo clippy` in strict mode
- Checks that documentation builds without warnings

## Performance Considerations

This implementation is optimized for performance:

- **Vectorized Operations**: The `DoryRoutines` trait provides optimized methods for batch operations:
  - `fixed_base_vector_scalar_mul` - Vectorized scalar multiplication
  - `fixed_scalar_mul_bases_then_add` - Fused multiply-add with base vectors
  - `fixed_scalar_mul_vs_then_add` - Fused multiply-add for vector folding

- **Prepared Point Caching** (optional `cache` feature): Pre-computes prepared points for setup generators, providing ~20-30% speedup on pairing operations. Initialize with:
  ```rust
  use dory_pcs::backends::arkworks::init_cache;
  init_cache(&prover_setup.g1_vec, &prover_setup.g2_vec);
  ```

- **Parallelization** (optional `parallel` feature): Uses Rayon for parallel multi-scalar multiplications and multi-pairings, providing significant speedup on multi-core systems.

- **Pluggable Polynomial Implementations**: The `Polynomial` and `MultilinearLagrange` traits are exposed, allowing users to provide custom polynomial representations optimized for their specific use cases (e.g., sparse polynomials)

- **Backend Flexibility**: The modular design allows backend implementations to leverage hardware acceleration and custom high performance arithmetic libraries

## Building and Testing

```bash
# Build with arkworks backend
cargo build --release --features backends

# Build with all optimizations (cache + parallel)
cargo build --release --features backends,cache,parallel

# Run tests
cargo nextest run --features backends

# Run all tests with all features
cargo nextest run --all-features

# Run clippy
cargo clippy --features backends -- -D warnings

# Generate documentation
cargo doc --features backends --open

# Run examples (see Examples section above)
cargo run --example basic_e2e --features backends

# Run benchmark with optimizations
cargo bench --features backends,cache,parallel
```

## Features

- `backends` - Enable concrete backends. Currently supports Arkworks BN254.
- `cache` - Enable prepared point caching for ~20-30% pairing speedup. Requires `arkworks` and `parallel`.
- `parallel` - Enable parallelization using Rayon for MSMs and pairings. Works with both `arkworks` backend and enables parallel features in `ark-ec` and `ark-ff`.
- `zk` - Enable zero-knowledge mode. Adds the `ZK` mode type for generating hiding proofs with blinded protocol messages, sigma proofs, and scalar-product sub-proofs.
- `disk-persistence` - Enable automatic setup caching to disk. When enabled, `setup()` will load from OS-specific cache directories if available, avoiding regeneration.

## Project Structure

```
src/
├── lib.rs                          # Main API (setup, prove, verify)
├── primitives/
│   ├── arithmetic.rs              # Core arithmetic traits
│   ├── poly.rs                    # Polynomial traits and operations
│   ├── transcript.rs              # Fiat-Shamir transcript trait
│   └── serialization.rs           # Serialization abstractions
├── backends/
│   ├── mod.rs                     # Backend module exports
│   └── arkworks/                  # Arkworks BN254 backend
│       ├── mod.rs                 # Module exports
│       ├── ark_field.rs           # Field wrapper (ArkFr)
│       ├── ark_group.rs           # Group wrappers (ArkG1, ArkG2, ArkGT)
│       ├── ark_pairing.rs         # Pairing curve (BN254)
│       ├── ark_poly.rs            # Polynomial implementation
│       ├── ark_proof.rs           # Proof type alias and serialization
│       ├── ark_cache.rs           # Prepared point caching
│       ├── ark_setup.rs           # Setup wrapper with disk persistence
│       ├── ark_serde.rs           # Serialization bridge
│       └── blake2b_transcript.rs  # Blake2b transcript
├── mode.rs                        # Transparent and ZK mode types
├── setup.rs                       # Transparent setup generation
├── evaluation_proof.rs            # Proof creation and verification
├── reduce_and_fold.rs             # Inner product protocol
├── messages.rs                    # Protocol messages
├── proof.rs                       # Proof structure
└── error.rs                       # Error types

tests/arkworks/
├── mod.rs                         # Test utilities
├── setup.rs                       # Setup tests
├── commitment.rs                  # Commitment tests
├── evaluation.rs                  # Evaluation tests
├── integration.rs                 # End-to-end tests
├── homomorphic.rs                 # Homomorphic combination tests
├── non_square.rs                  # Non-square matrix tests
├── serialization.rs               # Proof serialization round-trip tests
├── cache.rs                       # Prepared point caching tests
├── soundness.rs                   # Soundness tests
└── zk.rs                          # Zero-knowledge mode and ZK soundness tests
```

## Test Coverage

The implementation includes comprehensive tests covering:
- Setup generation and disk persistence
- Polynomial commitment (square and non-square matrices)
- Evaluation proofs
- End-to-end workflows
- Homomorphic combination
- Non-square matrix support (nu < sigma, nu = sigma - 1, and very rectangular cases)
- Zero-knowledge mode (hidden evaluations, sigma proofs, scalar-product proofs, soundness)
- Statistical indistinguishability of ZK proofs (witness independence)
- Soundness (tampering resistance for all proof components across 20+ attack vectors)
- Prepared point caching correctness

## Acknowledgments

This implementation was inspired by the proof of concept Dory PCS implementation by Space and Time Labs:
- Original repository: https://github.com/spaceandtimelabs/sxt-dory
- Research paper: Jonathan Lee, "Dory: Efficient, Transparent arguments for Generalised Inner Products and Polynomial Commitments" ([eprint 2020/1274](https://eprint.iacr.org/2020/1274))

## License

Dory is dual licensed under the following two licenses at your discretion: the MIT License (see [LICENSE-MIT](LICENSE-MIT)), and the Apache License (see [LICENSE-APACHE](LICENSE-APACHE)).

Dory is Copyright (c) a16z 2025. However, certain portions of the Dory codebase are modifications or ports of third party code, as indicated in the applicable code headers for such code or in the copyright attribution notices we have included in the directories for such code.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.

### Disclaimer

_This code is being provided as is. No guarantee, representation or warranty is being made, express or implied, as to the safety or correctness of the code. It has not been audited and as such there can be no assurance it will work as intended, and users may experience delays, failures, errors, omissions or loss of transmitted information. Nothing in this repo should be construed as investment advice or legal advice for any particular facts or circumstances and is not meant to replace competent counsel. It is strongly advised for you to contact a reputable attorney in your jurisdiction for any questions or concerns with respect thereto. a16z is not liable for any use of the foregoing, and users should proceed with caution and use at their own risk. See a16z.com/disclosures for more info._
