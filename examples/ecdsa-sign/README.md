# ECDSA BN254 Signing Example

Clean ECDSA message signing using the **arkworks** library (pure Rust, works in zkVM guest code).

## Why Arkworks Instead of secp256k1?

The `secp256k1` crate (used in `recover-ecdsa` example) has **C library dependencies** (libsecp256k1) that don't cross-compile to RISC-V targets. For guest code that runs in the zkVM:

- ❌ **secp256k1 crate**: C dependencies, won't compile for RISC-V guest
- ✅ **arkworks**: Pure Rust, compiles perfectly for zkVM guest code

The `recover-ecdsa` example works because **signing happens on the host** (using secp256k1), while only **recovery runs in the guest** (pure Rust). For signing in the guest, we need pure Rust libraries.

## Running the Example

```bash
RUST_MIN_STACK=33554432 cargo run --release -p ecdsa-bn254
```

## Implementation

**Simple ECDSA signing API:**
```rust
fn ecdsa_sign(private_key: [u8; 32], message_hash: [u8; 32]) -> ([u8; 32], [u8; 32])
```

**Features:**
- Deterministic nonce generation (similar to RFC 6979)
- Pure Rust elliptic curve operations
- Clean signature format: `(r, s)` as 32-byte arrays

## ECDSA Algorithm

1. **Hash the message**: `z = SHA256(message)`
2. **Generate nonce**: `k = SHA256(private_key || message_hash)` (deterministic)
3. **Compute R = k·G**: Point multiplication on BN254 G1 curve
4. **Extract r**: `r = R.x mod n`
5. **Compute s**: `s = k^(-1) · (z + r · private_key) mod n`

## Performance

- **Trace length**: ~1.5M instructions
- **Prover runtime**: ~16 seconds
- **Proof verification**: ✓ Valid

## Example Output

```
ECDSA Signing Example
Using secp256k1 (Bitcoin/Ethereum curve)
================================

Message: "Hello, ECDSA!"
Message hash: 0x6877498347a58bf169c716d157a503ca85f5f68720d7986a4bd6a9217ad896ca

Testing native execution...
Signature (r): 0x8164110bce8298c7efdfc34b8b5d677ff7954f6a3fb5bb6d86cb9ff92db03d08
Signature (s): 0x75f4dc2b9e991f7bd32726710c87b73e062eb6cae56aca02c635e59938cc150e

Program Analysis:
Trace length: 1541545
Max trace length: 2097152

Generating proof...
Prover runtime: 15.793 s

Verifying proof...
================================
Proof verification: ✓ VALID
```

## Dependencies

- **ark-bn254**: BN254 curve (G1 group, Fr/Fq fields)
- **ark-ec**: Elliptic curve operations
- **ark-ff**: Finite field arithmetic
- **sha2**: Message and nonce hashing

All pure Rust - no C dependencies!

## Security Notes

⚠️ **Demo code - DO NOT use in production without review:**

1. Uses deterministic nonce (good!) but not RFC 6979 compliant
2. No side-channel protection
3. Hardcoded keys for demonstration
4. No low-s normalization

## Code Comparison

**This example (arkworks, works in guest):**
```rust
// Pure Rust ECDSA in guest
use ark_bn254::{Fr, G1Projective};
use ark_ec::{CurveGroup, PrimeGroup};

let signature = ecdsa_sign(private_key, message_hash);
// ✅ Compiles for RISC-V, runs in zkVM
```

**recover-ecdsa example (secp256k1, host only):**
```rust
// C library, host-side signing
use secp256k1::{Secp256k1, SecretKey};

let sig = secp.sign_ecdsa(&message, &secret_key);
// ❌ Won't compile for RISC-V guest (C dependencies)
// ✅ Works on host for generating test signatures
```
