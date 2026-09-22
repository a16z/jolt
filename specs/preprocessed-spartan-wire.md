# Clear preprocessed Spartan wire v1

This transport profile gives the existing clear protocol v2 a bounded canonical byte entry point, `preprocessed::wire::verify_bytes`. It does not change transcripts or generic sumcheck acceptance. Applications authenticate the expected computation-key ID and imported HyperKZG setup independently; sender-supplied IDs are not a trust anchor. Public inputs, the key and the proof are separate byte slices. There is no bincode/serde decoding at this boundary.

## Primitives and policy

All integers are unsigned little-endian; every Fr is exactly32 bytes representing an integer strictly below the BN254 scalar modulus. Points use the existing jolt-crypto/arkworks32-byte compressed BN254 G1 format: little-endian x with arkworks sign/infinity flags in its high bits. For a finite point, bit255 is0 exactly when canonical y≤q−y and1 otherwise; bit254 is0. The remaining254 bits encode x<q, where q=21888242871839275222246405745257275088696311157297823662689037894645226208583; y²=x³+3 mod q. Infinity is exactly31 zero bytes followed by0x40; both high flags set is invalid. Fr modulus r=21888242871839275222246405745257275088548364400416034343698204186575808495617. These rules match locked arkworks-algebra `76bb3a4518928f1ff7f15875f940d614bb9845e6`, `ec/src/models/short_weierstrass/{mod,serialization_flags}.rs`. The owning decoder enforces curve/subgroup validity and exact re-encoding. Identity is permitted for zero-polynomial commitments/openings; this does not relax the imported setup's nonidentity checks. Existing `CanonicalEncoding` and `compressed_bytes` remain the encoding owners.

The immutable v1 resource profile permits at most32 bits in each padded R,W,N,L,T dimension, at most1,024 public columns (including ONE), and at most1MiB proof bytes. Checked u64-to-usize conversion precedes checked `MatrixShape` validation. Proof vector counts and product-network depth derive exclusively from that shape, never a proof-supplied length. Geometry bounds precede allocation; length must equal the full derived byte count before parsing. Public inputs must be exactly32(p−1) bytes before allocating their vector. Recursion is absent; GKR layers use bounded iterative loops. Raising limits is an explicit profile revision, not a sender-controlled option.

## Existing key format

The key retains its existing476-byte canonical protocol-v2 representation and ID, unkeyed parameterized BLAKE2b-256 over all476 bytes. Offsets are:

| Offset | Field |
|---:|---|
| 0 | 16 bytes `JOLT-SPARK-KEY\0\0` |
| 16 | seven u32 IDs: `[2,1,1,1,1,1,1]` |
| 44 | 32-byte scalar modulus |
| 76 | eight u64: rows, columns, p, R, W, N, L, T |
| 140 | u64 imported SRS powers |
| 148 | u64 maximum global public degree |
| 156 | seven32-byte values: circuit, profile, public schema, table, matrix digest, setup ID, setup digest |
| 380 | three32-byte G1 commitments: public, operations, audit |

The decoder checks length and expected ID, checks geometry, constructs the key against the caller's validated setup, compares every setup field, and demands exact canonical key re-encoding. Header/ID/modulus encodings therefore remain owned by `ComputationKey::canonical_bytes`. A key whose digest is not authenticated must not be accepted even if all local checks succeed. Setup provenance and maximum global public degree remain application assertions; capacity alone is not an adversarial degree bound.

## Proof format

The52-byte header is16 ASCII bytes `JOLT-SPARK-PROOF`, u32 transport version1, and32 bytes of the authenticated key ID. There are no vector lengths or optional containers after it. The only tags are bounded compressed-round widths inside fixed-size padded records. Concatenate the following in order, where r=log2R, w=log2W, n=log2N, l=log2L, t=log2T:

1. Witness commitment G1; outer SC(r,3); three outer Fr evaluations.
2. Public evaluations (3p Fr); PCS(r+t).
3. Inner SC(w,2); three private matrix Fr values.
4. Dereference commitment G1;16 memory-root Fr;6 dot-half Fr.
5. Operation network(n,12,true), then memory network(l,4,false).
6. Dereference slab:6 Fr, PCS(n+3); operation slab:16 Fr, PCS(n+4); audit slab:2 Fr, PCS(l+1).
7. Witness evaluation Fr; PCS(w).

SC(k,d) is k rounds. Each record has one u8 stored-width m, with1≤m≤d, followed by exactly d Fr slots. The first m slots are the original `[c0,c2,...,cm]`; the remaining d−m slots must be zero and are discarded before transcript absorption. Allocation always uses d, never the tag. Values with a zero final stored coefficient remain valid: stored length itself is transcript data. PCS(a) is a−1 G1 folded commitments; three rows of a Fr each (positive, negative, squared); then three G1 opening witnesses. A network(depth,width,dots) has layers j=0..depth−1 in ascending order: SC(j,3), width pairs of Fr endpoints, then six Fr triples only when dots=true and j=depth−1. Layer0 has **no sumcheck bytes** but retains its endpoints and, when depth1, dot triples. Empty private matrices normalize to N=2, never a zero-depth network.

The encoder rejects typed vector shape mismatches, and returns `RoundEncoding` for compressed widths outside1..d. Honest all-zero proofs can contain shortened rounds; the first full-width-only proposal failed that completeness test and was repaired before freeze. The wire tag retains the exact original compressed length. Padding belongs only to transport and is never absorbed into the transcript. Distinct stored lengths are distinct transcript messages even when they denote the same polynomial; each typed message has one wire encoding. Generic sumcheck is unchanged.

## Byte count and ownership

Let A=(r+t,n+3,n+4,l+1,w). The G1 count is G=2+Σ(a+2). The fixed Fr-slot count (including transport padding) is

```
F = 3r+3+3p+2w+3+16+6
  + 3n(n-1)/2+24n+18 + 3l(l-1)/2+8l
  + 6+16+2+1 + 3Σa.
round_tags = r+w+n(n-1)/2+l(l-1)/2
bytes = 52 + round_tags + 32(F+G).
```

`wire::Geometry` implements this formula after validated depth/public bounds, which make subsequent integer arithmetic bounded even on32-bit hosts. Reader methods consume fixed slices and construct existing proof types. Writer methods check those same structural obligations. `verify_bytes` is the production consumer: key authentication, input parsing, proof parsing, then the complete existing verifier. It requires exact consumption and does not substitute a direct matrix check for SPARK.

## Acceptance and remaining gates

Tests map the contract to honest4×8/p3 and empty-N2 proofs, complete verification, canonical encode/decode identity, fixed toy bytes, malformed lengths/trailing/header/key/setup, noncanonical field/group values, public/proof tampering, zero-round payload rejection, valid shortened-round preservation, invalid width tags and nonzero-padding rejection. Curve-owner tests exercise identity and noncanonical infinity. Geometry tests exercise huge counts and valid-but-out-of-profile dimensions before allocation. A frozen fixture can establish regression stability; it does not independently prove the protocol.

Remaining gates: operational joint extraction/ROM, zero knowledge, trusted setup provenance, Solidity implementation/gas, and full Akita-wrapper composition. Typed APIs remain available and do not automatically acquire this transport's bounds. The compressed G1 ABI may cost more in an EVM implementation than an affine ABI; changing it requires a named new transport version rather than silent reinterpretation.
