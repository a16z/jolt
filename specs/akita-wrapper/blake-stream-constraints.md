# Blake counter-stream constraint slice

Native source: published `markosg04/akita` revision
`28fc72021c120e7bc4e0102e07844f25299051eb`, protocol epoch 6.
The Cargo manifest and lockfile pin this fork. Earlier task-local patch tests
are retained as earlier evidence; final validation uses the published revision
without `.git/validation.toml`. This is a byte relation, not a complete wrapper.

## Source and ownership

| Native source | Constrained owner |
|---|---|
| `akita-transcript/src/blake2b_stream.rs` | `jolt-transcript::r1cs::Blake2bStreamVar` |
| `akita-challenges/src/sampler/xof.rs::IndexedXofPrefix::reader` | `jolt-akita::r1cs::AkitaSparseStreamVar::new` |
| `akita-transcript/src/sponge.rs` protocol/session domain constants | imported by `AkitaTranscriptVar`; native v2 differential is rerun |

The generic stream hashes
`u32le(domain.len) || domain || u64le(context.len) || context || u64le(index)`
with full unkeyed Blake2b512. Index zero comes first. The Akita caller imports
`SPARSE_CHALLENGE_STREAM_DOMAIN` from native Akita and frames exactly the
32-byte group root followed by the public coordinate as LE-u64. Domain ownership
is not duplicated in production code. Test-oracle literals intentionally encode
the independent specification.

Every context byte is a ByteVar: constant or eight Boolean-constrained bits.
The reviewed Blake2b512/word gates constrain each output block. Partial reads
reuse existing output handles; empty reads emit nothing. Public u128 positions
track all 2^64 blocks (2^70 bytes), including the last u64 counter. A request
past capacity returns StreamExhausted before changing position, output cache,
or builder constraints. Lengths and framing conversions are checked.

All schedules, lengths, domain bytes and coordinates are fixed public circuit
shape. Witness-dependent early termination is not supported. ONE must be fixed
externally to one; every supplied handle and subsequent operation must use the
same builder. Index validation only checks bounds, not builder provenance.
Consumers must bind output bytes. A root allocated with an honest value is not
an authenticated FoldDraw root: a later caller must connect it to the constrained
transcript and descriptor. The same obligation applies to session/instance bytes.

The first Akita API has the documented external fixed-shape wrapper-synthesis
contract. It represents the actual native indexed stream, but no completed
wrapper invokes it yet. No position/sign rejection sampling, Fisher–Yates
routing, accepted-draw selection, operator norm, grinding or root derivation is
implemented by this packet. Native integer rejection remains unbounded; the
4096 outer norm-attempt cap does not bound each integer rejection loop.

## Verification and costs

Independent values are reproducible with
`python3 scripts/blake_stream_constraint_vectors.py`. Tests cover initial sparse
blocks, the final u64 counter, native published-stream parity, split reads at
0/63/64/65/127/128/129, index separation with the same root, input/output bit
mutations, identical known/unknown-witness matrices, equal shape across read
splits, empty reads, atomic exhaustion and invalid builder indices. Existing
Akita duplex tests verify v2 protocol framing and the imported session domain.
The private final-position injection is test-only; there is no production seek API.

The intentional `sparse_stream_r1cs_cost` example uses a private 32-byte root,
public coordinate zero and separately allocated/bound claimed output bytes:

| Output | Constraints | Variables including ONE | Matrix nonzeros |
|---|---:|---:|---:|
| 64 bytes / one block | 52,160 | 51,713 | 283,624 |
| 128 bytes / two blocks | 104,064 | 103,169 | 566,492 |

These are exact sizes for this framing and binding contract, not runtime or
complete-sampler costs. Each native sparse counter message fits one Blake
compression block. Root authentication and all omitted sampling work are excluded.

Focused commands (set CARGO_TARGET_DIR to an inactive task-owned target):

```sh
cargo nextest run --offline --locked -p jolt-transcript --no-default-features \
  --features r1cs --lib -E 'test(r1cs::stream)' --cargo-profile dev --cargo-quiet
cargo nextest run --offline --locked -p jolt-akita --no-default-features \
  --features r1cs --test r1cs_sparse_stream --test r1cs_transcript \
  --cargo-profile dev --cargo-quiet
cargo clippy --offline --locked -p jolt-transcript -p jolt-akita --all-targets \
  --no-default-features --features r1cs -- -D warnings
cargo run --offline --locked -p jolt-akita --no-default-features --features r1cs \
  --example sparse_stream_r1cs_cost
```
