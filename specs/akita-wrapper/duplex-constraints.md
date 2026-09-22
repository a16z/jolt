# Akita BLAKE2b duplex constraint contract

This packet constrains the byte transcript used by Akita
`252abb895046cc1d5b9955a26a2ad2318148ac26` with spongefish
`d2d190b1329d35ac9577438d05aed4f17a57b9f9`. It targets the native **64-bit**
verifier's `usize` encoding. It is not a complete Akita verifier circuit.

The generic primitive lives in `jolt-transcript::r1cs::Blake2bDuplexVar`;
`jolt-akita::r1cs::AkitaTranscriptVar` owns the actual Akita framing caller.
Both are behind each existing crate's `r1cs` feature. Dependencies point from
jolt-akita to jolt-transcript to jolt-r1cs; there is no new primitive crate or cycle.
The Akita API's external contract is fixed-shape wrapper synthesis: accept
constrained session/instance bytes, append canonical message bytes in verifier
order, and return constrained challenge bytes for later verifier constraints.
No complete wrapper calls it yet.

## Claim-to-source map

Let H be unkeyed BLAKE2b-512, M(t) be 127 zero bytes followed by t, and cv be
64 bytes, initially zero. The primitive uses the reviewed BLAKE2b and bit
constraints without changing their implementation.

| Native owner | Constrained owner and relation | Evidence |
| --- | --- | --- |
| spongefish `instantiations/hash.rs::Hash::absorb` | `Blake2bDuplexVar::absorb`: new run buffers M(0) || cv || input; subsequent absorbs concatenate, including empty calls | Native streaming/empty tests |
| `Hash::ratchet` | `ratchet`: cv = H(H(buffer)), reset to Start | Native explicit ratchet tests |
| `Hash::squeeze` | `squeeze`: finalize absorption via ratchet, then output H(M(1) || cv || index_be_u64) blocks, retaining unused suffixes | Native 1/63/65-byte streaming test |
| `Hash::squeeze_end` | `squeeze_end`: cv = H(M(2) || cv || consumed_be_u64), drop leftovers | Native partial-squeeze/reabsorb test |
| Akita `akita-transcript/src/sponge.rs::domain_separator_from_label`, `SessionBoundInstance::encode` | `AkitaTranscriptVar::new`: padded 64-byte protocol tag, padded session-domain tag, LE-u64 session length, session, LE-u64 instance length, instance | Actual `AkitaTranscript::verifier` differential |
| spongefish `domain_separator.rs::to_verifier`, `narg_verifier.rs::public_message` | constructor absorbs these fields consecutively, without extra hash or ratchet boundaries | Same differential |
| Akita `FramedBytes::encode`, `absorb_bytes` | `append_bytes`: LE-u64 length || bytes; operation labels omitted upstream | Same differential |
| Akita `squeeze_bytes`, spongefish `verifier_message::<[u8;32]>` | `challenge_bytes`: consume ceil(len/32)*32 bytes, return first len; zero length does nothing | Actual 0/1/33/32 and consecutive 1/1/31 differential |

## Resolved edge cases

Upstream `squeeze([])` changes Start to Squeeze and ratchets an Absorb run even
though it emits no bytes. In contrast, Akita `squeeze_bytes(0)` never calls the
sponge. Empty absorbs also change state. These behaviors are preserved.

An explicit upstream ratchet outside Absorb first ends squeezing, if necessary,
and then finalizes a **reset empty hasher**, producing H(H(empty)). This apparent
loss of cv is upstream behavior, not repaired here. Normal Akita byte operations
only ratchet when finishing an absorb run.

Spongefish uses native `usize::to_be_bytes()` for squeeze index and consumed
length. This packet fixes eight-byte framing rather than silently making the
relation vary with the circuit builder's host. It does not claim 32-bit native
transcript compatibility. Requests beyond `u64::MAX - 63` consumed bytes are
rejected before emission, excluding upstream overflow in count*64.

Absorb runs buffer byte handles until finalization. This preserves the exact hash
input, but is not a streaming-memory optimization. All lengths, operation types,
and control flow are synthesis-time public parameters. Witness values only
populate assignments. Known/unknown witness matrix equality is tested.

## Caller obligations and omissions

ONE must be fixed externally to one. Every handle must come from the same
builder; indices alone do not authenticate provenance. Outputs are private
handles until bound by a consumer. The previous hash review's assumptions apply.

The constructor accepts the **actual** bound instance, matching upstream's state
replacement at `bind_instance_bytes`, not the default placeholder instance.
`jolt-akita/src/adapters.rs::bridged_akita_transcript` concatenates the public
session label with the outer Jolt scalar's canonical LE bytes. A wrapper must
constrain that scalar derivation/encoding and pass the concatenated session.
It must also authenticate preprocessing/setup and constrain canonical descriptor
and message encodings; arbitrary byte witnesses do not establish those facts.

This packet does not constrain scalar challenge reduction, SHAKE256 sparse
challenge sampling or rejection, grinding predicates, descriptor/proof parsing,
any Akita field/ring acceptance relation, or the outer Jolt legacy transcript.
No end-to-end proof, outer SNARK, security proof, or performance claim follows.
Native differential tests use actual production backends, not copied old logic;
completed-witness mutations test binding but are not an exhaustive soundness proof.

## Validation commands

```
cargo nextest run -p jolt-transcript --features r1cs --cargo-quiet
cargo nextest run -p jolt-transcript --no-default-features --features r1cs --cargo-quiet
cargo nextest run -p jolt-akita --features r1cs --test r1cs_transcript --cargo-quiet
cargo clippy -p jolt-transcript -p jolt-akita --features jolt-transcript/r1cs,jolt-akita/r1cs --all-targets -- -D warnings
cargo clippy -p jolt-transcript -p jolt-akita --no-default-features --features jolt-transcript/r1cs,jolt-akita/r1cs --all-targets -- -D warnings
cargo run -p jolt-transcript --no-default-features --features r1cs --example blake2b_r1cs_cost
```

The existing cost diagnostic now includes a default duplex, one absorb, and a
32-byte squeeze, with private input allocation and private claimed-byte binding.
Each BLAKE2b compression emits 51,328 rows. For absorb lengths 0 and 1 this path
uses five compressions; length 129 uses six. Counts exclude Akita initialization
unless those bytes are included in the absorbed payload. Full-transcript costs
require the actual accepted proof schedule and are not extrapolated here.

Measured diagnostic rows (private input and 32-byte claimed digest included):

| Absorbed bytes | Rows | Variables (including ONE) | Sparse nonzeros |
| ---: | ---: | ---: | ---: |
| 0 | 256,928 | 254,977 | 1,401,329 |
| 1 | 256,936 | 254,985 | 1,401,450 |
| 129 | 309,288 | 306,953 | 1,695,035 |

The Akita protocol tag is imported from the pinned `akita-transcript` owner.
Its private session-domain tag has no public accessor and is reproduced verbatim,
with the native full-transcript differential test as the compatibility check.
The native spongefish test dependency enables `sha3` because that pinned crate
requires it to compile `StdHash`; no SHAKE constraints are implemented or called.
The existing `native_traits_tests` target now declares its required `bn254`
feature so mixed reduced-feature workspace builds do not select an invalid test.

Validation on 2026-09-22: default transcript suite 73/73 passed; minimal
`r1cs` transcript suite 8/8 passed; isolated production Akita transcript comparison
1/1 passed. Both targeted all-target clippy commands above passed. Existing
workspace manifest `default-features` warnings remain. The cost diagnostic ran
successfully with the minimal `r1cs` feature set. Workspace host/ZK suites,
complete Akita proof replay, 32-bit compatibility, and outer proving were not run.
