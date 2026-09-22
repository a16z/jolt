# Constrained Blake2b over BN254

This slice implements unkeyed Blake2b-256 and Blake2b-512, plus initialization
and append transitions for Jolt's `LegacyBlake2bTranscript`. It does not implement
Akita's spongefish duplex transcript, SHAKE256, challenge field decoding, or a
complete verifier circuit. Native transcript behavior is unchanged.

## Ownership and source mapping

`jolt-r1cs::bn254_bits` owns Boolean bytes and 64-bit modular word operations.
`jolt-transcript::r1cs` owns Blake2b hashing and legacy transcript framing and is
the first production caller of those word operations. This direction avoids a
cycle: `jolt-crypto` already depends on `jolt-transcript`. Both modules are opt-in;
no new crate or generic field abstraction is introduced.

[RFC 7693](https://www.rfc-editor.org/rfc/rfc7693.html), sections 2.5–3.3,
is the algorithm source. `IV`, `SIGMA`, `mix`, `compress`, and `hash` correspond
to its constants, permutation, G function, compression, and full hash. Initialization
includes the digest-length parameter, so 256-bit hashing is not truncation of
512-bit hashing. Blocks use little-endian words, actual cumulative byte counters,
and the final-block flag. Empty input emits one final zero block with counter zero;
exact block multiples do not append another block. Keyed and tree modes are absent.

`LegacyBlake2bVar` follows `src/digest.rs`: initialization hashes the public label
padded to 32 bytes; append hashes `state[32] || zeros[28] || round_be_u32 || payload`,
then increments the public round. Round overflow returns a typed error. Label,
message length, block count, round and finalization schedule are circuit parameters.
They are not unconstrained witnesses. Native modern `Blake2bTranscript` is a
spongefish transcript and is intentionally not claimed to match this legacy framing.

## Constraint argument and boundary contracts

Allocated bits satisfy `b(b-1)=0`. XOR allocates `z` and emits `(2x)y=x+y-z`. For Boolean inputs the
four cases force the unique field value `z` to be respectively 0, 1, 1, 0;
Boolean output therefore follows without another Boolean row. Materializing `z`
keeps every bit expression sparse across additions and compression blocks. Rotations and byte/word conversion only permute certified bits.

Two-word addition allocates 64 Boolean output bits and one Boolean carry bit;
three-word addition allocates 64 output bits and two Boolean carry bits. One row
binds the integer input sum to output plus `2^64 * carry`. All possible residuals
have magnitude below `2^67`, below the BN254 scalar modulus. Field equality thus
implies integer equality, including overflow. For three operands, carry 3 cannot
satisfy that equality even though its two-bit representation permits it. Induction
through the RFC schedule pins every returned output bit to the hash of input bits.

Returned digest bytes are constrained handles, not automatically public inputs.
The consumer must bind them to its statement or consume them in subsequent gates.
The diagnostic includes private claimed digest allocation and equality binding.
As with existing R1CS gadgets, the outer statement must fix the constant-one column.
Handles must remain in their allocating builder; index checks reject out-of-range
handles but do not prove provenance when indices happen to overlap. These APIs
preserve that existing same-builder contract.

Known and unknown witness construction emit the same matrices. Witness bits,
intermediate words and carries can remain private under an eventual ZK outer
SNARK; this module itself supplies neither a ZK proof nor constant-time witness
generation. Message length, label and round schedule remain public. The complete
acceptance circuit must additionally constrain proof encoding and challenge decoding.

## Evidence

Permanent tests use the published RFC `abc` 512-bit digest and the native `blake2`
implementation at lengths 0, 3, 127, 128 and 129 for both digest sizes. A chained
transition test compares the actual legacy transcript at initialization and two
appends (including a multiblock append and nonzero round). Completed-witness attacks
flip input/digest Boolean bits, inject a non-Boolean input and corrupt overflow
carry. Additional tests compare known/unknown matrices and check typed public-shape
and invalid-index failures before constraint emission.

Commands (no field-inline):

```
cargo nextest run -p jolt-r1cs -p jolt-transcript --features jolt-transcript/r1cs --cargo-quiet
cargo nextest run -p jolt-r1cs -p jolt-transcript --no-default-features --features jolt-transcript/r1cs,jolt-transcript/transcript-blake2b,jolt-transcript/bn254 --cargo-quiet
cargo clippy -p jolt-r1cs -p jolt-transcript --features jolt-transcript/r1cs --all-targets -- -D warnings
cargo run -q -p jolt-transcript --example blake2b_r1cs_cost --features r1cs
```

The fully minimal all-target test invocation without transcript/bn254 hits an
existing unguarded `native_traits_tests` dependency on ark-bn254; use the explicit
bn254 feature above. This slice does not repair unrelated test feature wiring.
Constraint counts are structural diagnostics, not proving-time or gas evidence.
No complete-wrapper or workspace-wide host/ZK suite was run for this optional slice.

Historical baseline `51f6c8adf` diagnostic (variables include ONE; nonzeros sum A/B/C after row
canonicalization; no witness-dependent constant folding):

| Hash | Input bytes | Rows | Variables | Nonzeros |
|---|---:|---:|---:|---:|
| 256 | 0 | 51,616 | 51,201 | 2,621,363 |
| 512 | 0 | 51,904 | 51,457 | 2,672,307 |
| 256 | 128 | 52,640 | 52,225 | 2,636,724 |
| 512 | 128 | 52,928 | 52,481 | 2,687,668 |
| 256 | 129 | 103,976 | 103,177 | 10,247,687 |
| 512 | 129 | 104,264 | 103,433 | 10,398,343 |

That baseline retained expanded XOR linear combinations, producing substantial
nonzero growth. The follow-up replaces each XOR product variable by its output
variable and rewrites its single row as above. It changes neither row nor variable
counts and performs no constant folding. Neither version provides evidence of
practical full-transcript proving time.

Validation on this branch: default targeted nextest 89/89 passed (run
`c53b468e-390d-469d-97e2-9a9275866a4a`); reduced features with explicit BN254
57/57 passed (run `42ad85ef-2cf2-4b6f-9828-e9343964e936`); targeted all-target
clippy, fmt check, diff check and cost diagnostic exited zero. Raw local evidence:
`/private/tmp/blake2-r1cs-nextest.log`,
`/private/tmp/blake2-r1cs-minimal-bn254-nextest.log`,
`/private/tmp/blake2-r1cs-clippy.log`, `/private/tmp/blake2-r1cs-cost.log`.
The failed feature invocation is preserved in
`/private/tmp/blake2-r1cs-minimal-nextest.log` (exit 101).

Materialized-XOR diagnostic, with identical complete row/variable counts:

| Hash | Input bytes | Nonzeros |
|---|---:|---:|
| 256 | 0 | 278,400 |
| 512 | 0 | 279,712 |
| 256 | 128 | 293,761 |
| 512 | 128 | 295,073 |
| 256 | 129 | 572,105 |
| 512 | 129 | 573,417 |

The same independent-vector, native-transcript and completed-witness tamper tests
pass after materialization: 89/89 default and 57/57 reduced-feature tests. Targeted
all-target clippy and the cost diagnostic exit zero. Follow-up raw logs are
`/private/tmp/blake2-r1cs-materialized-nextest.log`,
`/private/tmp/blake2-r1cs-materialized-minimal-nextest.log`,
`/private/tmp/blake2-r1cs-materialized-clippy.log`, and
`/private/tmp/blake2-r1cs-materialized-cost.log`. These sparse-matrix counts support
only the structural cost comparison; they are not timings or a full-wrapper estimate.
