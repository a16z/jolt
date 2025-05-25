# Dory: A Rust Implementation of the Dory Polynomial Commitment Scheme

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

A flexible and efficient Rust implementation of the **Dory polynomial commitment scheme**, based on the work by Jonathan Lee: ["Dory: Efficient, Transparent arguments for Generalised Inner Products and Polynomial Commitments"](https://eprint.iacr.org/2020/1274.pdf).

Dory is a polynomial commitment scheme with excellent asymptotic performance as well as practical efficiency, designed to be used as a building block for other zero-knowledge (ZK) and SNARK protocols.

## 📑 Table of Contents

- [Dory: A Rust Implementation of the Dory Polynomial Commitment Scheme](#dory-a-rust-implementation-of-the-dory-polynomial-commitment-scheme)
  - [📑 Table of Contents](#-table-of-contents)
  - [🎯 Features](#-features)
  - [🛠️ Installation](#️-installation)
  - [🚀 Usage](#-usage)
    - [Basic Example](#basic-example)
    - [Setup and SRS Management](#setup-and-srs-management)
  - [🧪 Testing](#-testing)
  - [📊 Performance](#-performance)
  - [🔧 Technical Overview](#-technical-overview)
    - [Core Components](#core-components)
    - [Cryptographic Assumptions](#cryptographic-assumptions)
  - [🤝 Contributing](#-contributing)
      - [📝 Commit Messages](#-commit-messages)
  - [📧 Contact](#-contact)
  - [📚 Additional Resources](#-additional-resources)

## <a name="features"></a>🎯 Features

- **Efficient Polynomial Commitments**: Optimized implementation of the Dory PCS with excellent asymptotic performance
- **Multilinear Polynomial Support**: Full support for multilinear polynomials with evaluation proofs
- **SRS Management**: Structured Reference String (SRS) generation, saving, and loading with file caching
- **Flexible Backend**: Supports multiple elliptic curve backends via the Arkworks ecosystem

## <a name="installation"></a>🛠️ Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
dory = "0.0.1"
```

Or install directly from the repository:

```bash
cargo add --git https://github.com/spaceandtimelabs/sxt-dory
```

## <a name="usage"></a>🚀 Usage

### Basic Example

Here's a complete example of committing to a polynomial and creating an evaluation proof:

```rust
use dory::*;
use dory::curve::{test_rng, ArkBn254Pairing, OptimizedMsmG1, OptimizedMsmG2, DummyMsm};
use ark_bn254::Fr;
use ark_ff::UniformRand;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = test_rng();
    let domain = b"my_application";
    
    // Setup parameters
    let num_variables = 10;  // 2^10 = 1024 coefficients
    let sigma = 5;           // Reduction parameter
    
    // Generate or load SRS (Structured Reference String)
    let (prover_setup, verifier_setup) = 
        setup_with_srs_file::<ArkBn254Pairing, _>(
            &mut rng, 
            num_variables, 
            Some("./k_10.srs")
        );
    
    // Create a random multilinear polynomial
    let num_coeffs = 1 << num_variables;
    let coeffs: Vec<Fr> = (0..num_coeffs)
        .map(|_| Fr::rand(&mut rng))
        .collect();
    
    // Random evaluation point
    let point: Vec<Fr> = (0..num_variables)
        .map(|_| Fr::rand(&mut rng))
        .collect();
    
    // Commit to the polynomial
    let commitment = commit::<ArkBn254Pairing, OptimizedMsmG1>(
        &coeffs, 
        0, 
        sigma, 
        &prover_setup
    );
    
    // Create evaluation proof
    let transcript = create_transcript::<Fr>(domain);
    let (evaluation, proof) = evaluate::<
        ArkBn254Pairing, 
        _, 
        OptimizedMsmG1, 
        OptimizedMsmG2
    >(
        &coeffs,
        &point,
        sigma,
        &prover_setup,
        transcript,
    );
    
    // Verify the proof
    let verification_result = verify::<
        ArkBn254Pairing, 
        _, 
        OptimizedMsmG1, 
        OptimizedMsmG2, 
        DummyMsm<_>
    >(
        commitment,
        evaluation,
        &point,
        proof,
        sigma,
        &verifier_setup,
        domain,
    );
    
    assert!(verification_result.is_ok());
    println!("✓ Polynomial commitment and evaluation verified!");
    
    Ok(())
}
```

### Setup and SRS Management

The library provides flexible SRS (Structured Reference String) management:

```rust
// Generate new SRS and save to file
let filename = generate_srs::<ArkBn254Pairing, _>(&mut rng, 12)?;
println!("SRS saved to: {}", filename);

// Load existing SRS
let prover_setup = load_prover_setup::<ArkBn254Pairing>("k_12.srs")?;
let verifier_setup = load_verifier_setup::<ArkBn254Pairing>("k_12.srs")?;

// Setup with automatic file handling
let (prover_setup, verifier_setup) = setup_with_srs_file::<ArkBn254Pairing, _>(
    &mut rng, 
    12, 
    Some("k_12.srs")  // Will load if exists, generate and save if not
);
```

The core operations work with multilinear polynomials:

```rust
// Polynomial represented as coefficient vector
let coeffs: Vec<Fr> = vec![/* polynomial coefficients */];

// Commit to polynomial (returns group element in GT)
let commitment = commit::<ArkBn254Pairing, OptimizedMsmG1>(
    &coeffs, 
    0,      // offset 
    sigma,  // reduction parameter
    &prover_setup
);

// Evaluate at point and create proof
let point: Vec<Fr> = vec![/* evaluation point */];
let transcript = create_transcript::<Fr>(b"domain_separator");
let (evaluation, proof) = evaluate::<ArkBn254Pairing, _, OptimizedMsmG1, OptimizedMsmG2>(
    &coeffs,
    &point,
    sigma,
    &prover_setup,
    transcript,
);

// Verify the evaluation proof
let result = verify::<ArkBn254Pairing, _, OptimizedMsmG1, OptimizedMsmG2, DummyMsm<_>>(
    commitment,
    evaluation,
    &point,
    proof,
    sigma,
    &verifier_setup,
    b"domain_separator",
);
```

## <a name="testing"></a>🧪 Testing

Run the test suite:

```bash
cargo test
```

Run specific PCS tests:

```bash
cargo test pcs
```

Run with performance timing:

```bash
cargo test --release test_pcs_api_workflow -- --nocapture
```

## <a name="performance"></a>📊 Performance

The implementation includes several performance optimizations:

- **Parallel Processing**: Leverages Rayon for parallel computations
- **SRS Caching**: Avoids redundant setup generation through file caching

Typical performance for BN254 curve on modern hardware:
- **Setup (n=2^20)**: ~150 seconds (one-time cost, cached) for k = 20
- **Commit**: ~1s for k = 20
- **Prove**: ~3s for k = 20
- **Verify**: ~50 ms for k = 20

## <a name="technical-overview"></a>🔧 Technical Overview

This implementation follows the Dory paper's construction:

### Core Components

- **Inner Product Arguments**: The foundation of the Dory protocol (`src/core/`)
- **VMV (Vector-Matrix-Vector) Module**: Converts inner products to polynomial commitments (`src/vmv/`)
- **Interactive Protocol**: Fiat-Shamir transformed interactive proofs
- **Setup Generation**: Structured reference string creation and management

### Cryptographic Assumptions

- **Discrete Logarithm**: Security relies on discrete log hardness in bilinear groups
- **Pairing-Based**: Uses Type-3 pairings (e.g., BN254, BLS12-381)
- **Transparent Setup**: No trusted setup required, only public randomness

## <a name="contributing"></a>🤝 Contributing

We welcome contributions! Please follow these guidelines:

- **Issues**: Report bugs or request features via GitHub Issues
- **Pull Requests**: Submit PRs with clear descriptions and tests
- **Code Style**: Follow Rust conventions and run `cargo fmt`
- **Testing**: Ensure all tests pass with `cargo test`
- **Documentation**: Update docs for public APIs

#### <a name="commit-messages"></a>📝 Commit Messages

Please ensure your commits follow the [Conventional Commit](https://www.conventionalcommits.org/en/v1.0.0/) standards. Doing so allows our CI system to properly categorize changes during the release process.

## <a name="contact"></a>📧 Contact

For questions on this repository, please reach out to [markosg04](https://github.com/markosg04).

## <a name="additional-resources"></a>📚 Additional Resources

- [📋 Changelog](CHANGELOG.md)
- [📜 License](LICENSE)
- [👨‍💻 Code Owners](CODEOWNERS)