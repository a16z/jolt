# Lane J — compressed BN254 GT wire format

Date: 2026-09-02
Implementation: `6cdb8fbcb` (`feat: add compressed Dory GT wire format`)

## Codec survey

Source resolution:

- `dory-pcs` is crates.io `0.4.1`, not a vendored path dependency.
- The codecs live in the replaced a16z arkworks fork at commit `76bb3a45`: `curves/bn254/src/fields/compression.rs` and `ff/src/fields/models/quadratic_extension.rs`.

192-byte torus form:

- Stores one `Fq6`, the Cayley/torus coordinate for a norm-one `Fq12` element.
- Raw fork operations perform field algebra only. They do not check order-r membership or define a wire-level identity encoding; the compression division has a zero denominator at identity.
- Decode cost before validation: one `Fq12` inversion. Strict untrusted-byte decode also needs canonical `Fq6` parsing and `x^r = 1`.

128-byte `CompressedFq12` form, selected:

- Stores `(Fq2, Fq2)`, the first two coordinates of the torus `Fq6`.
- Drops the third `Fq2`; Proposition 1's cyclotomic relation recovers it as `c2 = (3c0² + ξ) / (3c1ξ)`. No square root or sign bit.
- Raw fork decode calls `inverse().unwrap()` when `c1 = 0`, and derived canonical deserialization checks only the four base-field limbs. It does not check the cyclotomic image, order-r membership, or uniqueness.
- Decode cost before validation: one `Fq2` inversion to recover `c2`, then one `Fq12` inversion for the torus map.

The Jolt wrapper makes the 128-byte form strict:

1. All-zero is the sole identity encoding; ordinary encodings require nonzero `c1`.
2. Each base-field limb uses canonical arkworks parsing.
3. The reconstructed element must satisfy `x^r = 1`.
4. Re-encoding must reproduce the input bytes.

Malformed input returns `Err`; no compression decoder path panics. Existing `Bn254GT` and Dory native subgroup validation remains unchanged; the compressed boundary performs the same order-r check before handing values to native verification.

## Wire sizes

Real modular Fibonacci proof, padded trace `2^18`: `L = 18`, `sigma = ceil((L + 4) / 2) = 11`.

The proof has 42 commitments, not the discovery note's 41: `2 + 32 instruction_ra + 4 ram_ra + 4 bytecode_ra`. Its 68 Dory-proof GTs give exactly 110 compressed GT elements.

| Component | Native bincode 2 standard | Compressed | Saved |
|---|---:|---:|---:|
| 42 commitments | 16,255 B | 5,377 B | 10,878 B |
| Dory proof, sigma=11 | 29,429 B | 12,021 B | 17,408 B |
| Total | 45,684 B | 17,398 B | 28,286 B (61.9%) |

For `N <= 250` commitments and proof payloads below 65,536 bytes:

- Native: `918 + 387N + 2592sigma` bytes.
- Compressed: `406 + 128N + 1056sigma` bytes.
- Dory-proof element counts: `6sigma + 2` GT, `3sigma + 2` G1, `3sigma + 1` G2.

`JoltProof` and all existing fixture encodings are untouched. `CompressedDoryArtifacts` is a separate future-wrapper wire type.

## Decode timing

Criterion release build, this host:

| Operation | Median | 95% interval |
|---|---:|---:|
| One strict GT decode | 340.84 us | 331.14–353.85 us |
| 110 strict GT decodes | 36.488 ms | 34.302–39.120 ms |

An earlier run measured 252.90 us per element; CPU load caused visible variance. The table records the latest full 110-element run. The order-r exponentiation dominates both torus reconstruction and parsing.

## Verification

- `cargo fmt -q --message-format=short`
- `cargo clippy --all --features host -q --all-targets --message-format=short -- -D warnings`
- `cargo clippy --all --features host,zk -q --all-targets --message-format=short -- -D warnings`
- `cargo nextest run -p jolt-crypto -p jolt-dory -p jolt-prover --cargo-quiet` — 194 passed
- `cargo nextest run -p jolt-prover fibonacci_2_18_dory_artifacts_compress_and_verify --features prover-fixtures --cargo-quiet` — passed; decompressed full proof verified natively
- `cargo bench -p jolt-crypto --bench crypto -- gt_decompress`
