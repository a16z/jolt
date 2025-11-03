# Modexp Chain Example

This example demonstrates modular exponentiation (modexp) operations in Jolt, similar to the EVM's MODEXP precompile.

## Configuration

The example is configurable in several ways:

### Bit Length (Compile-time)

The bit length of the base, exponent, and modulus can be configured by changing the `BITLEN_BYTES` constant in `guest/src/lib.rs`:

```rust
// For 256-bit values (default)
const BITLEN_BYTES: usize = 32;

// For 512-bit values
const BITLEN_BYTES: usize = 64;

// For 1024-bit values
const BITLEN_BYTES: usize = 128;
```

### Number of Iterations (Runtime)

The number of iterations can be configured at runtime by passing the `iters` parameter to the `modexp_chain` function in `src/main.rs`:

```rust
let iters = 10; // Number of times to perform modexp
```

## Running the Example

```bash
cargo run --release -p modexp-chain
```

## Benchmarking

The example is also integrated into the e2e_profiling benchmark suite:

```bash
# Run the modexp-chain benchmark
cargo bench --bench e2e_profiling -- --bench-type modexp-chain
```

## Implementation

The modexp operation is implemented using the `num-bigint` crate's `modpow` method, which performs efficient modular exponentiation using the square-and-multiply algorithm.
