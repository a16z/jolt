# Spec: run jolt-transcript on spongefish's NARG state

| Field     | Value                                                           |
| --------- | --------------------------------------------------------------- |
| Author(s) | @markosg04                                                      |
| Created   | 2026-10-05                                                      |
| Status    | implemented (in review)                                         |
| Amends    | `specs/jolt-transcript-narg.md` (Grinding and previews; Alternatives considered) |

## Decision

`jolt-transcript` stops reimplementing the NARG state machine and becomes a
typed layer over spongefish's `ProverState<H, StdRng>` and
`VerifierState<'a, H>`. Atoms implement spongefish's `Encoding`,
`NargSerialize`, and `NargDeserialize` in the crates that own them. Domain
separation uses spongefish's `DomainSeparator`. The explicit typing stays
where it is now: claims structs, routes, and message groups, generated per
stage. Challenges stay exactly uniform. Grinding becomes squeeze-then-grind,
so no crate needs spongefish's `yolocrypto` feature. Jolt and Akita both
run on the result, and Akita opens inside its caller's transcript.

This reverses the narg spec's rejection of spongefish's prover and verifier
state. The four reasons given there now resolve as follows:

| Reason given | Resolution |
| --- | --- |
| `VerificationError` is a unit type | The wrapper checks each read against the remaining proof itself and reports a typed `TranscriptError` (`Truncated`, `NonCanonical`, `OutOfBounds`, ...). Sites stay in the diagnostic log. |
| Previews need `yolocrypto` | Previews are removed. Twin and replay checks compare argument strings plus one final 32-byte squeeze on each side. Grinding no longer previews. |
| Encode and serialize are separate | For byte sponges (`H::U = u8`), spongefish's blanket `NargSerialize for T: Encoding<[u8]>` makes them identical. Jolt pins byte sponges only (see Non-goals). |
| No logging hook | The wrapper logs each call with its site before delegating. |

## Invariants

1. Every prover message reaches the sponge only through `prover_message` and
   `prover_messages*`, and every public value only through `public_message*`.
   No crate reads, clones, or writes spongefish's duplex state.
2. No crate in the Jolt or Akita dependency graphs enables
   spongefish's `yolocrypto`. CI checks this with `cargo tree -e features`.
3. Field challenges are exactly uniform. The wrapper samples them by
   rejection over fixed-width `verifier_message::<[u8; N]>()` squeezes, using
   `Field::random`'s contract. The small challenge set is unchanged: 16
   squeezed bytes through `CanonicalEncoding::from_challenge_bytes`.
   On a sponge whose absorb erases the squeeze width within an output block
   (`Sponge::ABSORB_FORGETS_SQUEEZE_WIDTH`: spongefish's overwrite-mode
   Keccak duplex and the byte-level Poseidon sponge), each exact challenge
   then absorbs the number of bytes it squeezed as a public `u64`, so distinct
   rejection-retry histories cannot reconverge at the next absorb. Blake2b
   retains the width and absorbs nothing extra.
4. Message shapes stay positional. No message carries a length the public
   parameters already fix.

## Mechanism

**Crates.** `jolt-transcript` depends on `spongefish` with
`default-features = false` plus each sponge's feature. It keeps
`ProverTranscript<H>`, `VerifierTranscript<'a, H>`, `Channel`, `Sponge`,
`SiteId`, `TranscriptError`, and `ProtocolId`, and adds `Fork`. `Duplex` and
`Preview` are deleted. `CanonicalBytes` and `CanonicalDecode` stay as each
type's fixed-width codec, because the wrapper needs `NUM_BYTES` to bound a read
before it allocates and to log byte ranges. They gain spongefish's `Encoding`
and `NargDeserialize` as supertraits. Each owning crate implements those two
traits through the shared `jolt_field::narg::{encode, deserialize}`: canonical
little endian, with non-canonical encodings rejected. spongefish implements
`NargDeserialize` for `u32` and byte arrays only, so `u8`/`u64` header fields
travel as byte arrays.

**Construction.** `ProtocolId` stays the 64-byte protocol tag (name plus
sponge id). A transcript starts from
`DomainSeparator::new(protocol_id).session(B(session)).instance(B(instance))`
followed by `to_prover(H::default())` or `to_verifier(H::default(), narg)`.
`B` is a length-framed byte string. Every transcript passes an empty instance:
Jolt and Akita bind their statement data through public messages, so
the instance slot carries nothing.

**Operations.**

| `jolt-transcript` | spongefish |
| --- | --- |
| `send` / `send_all`, `receive` / `receive_n` | `prover_message`, `prover_messages`, `prover_messages_vec` |
| `send_bytes(len known)` / `receive_bytes(len)` | one `[u8; 32]` message per whole block, then one `[u8; 1]` message per remaining byte, on both sides; the verifier also checks the bytes it returns equal the bytes the sponge absorbed |
| `send_bounded_bytes` / `receive_bounded_bytes` | `u32` LE length as a prover message, then the bytes; the verifier enforces the maximum before reading |
| `public*` | `public_message*` |
| `challenge::<F>()` | rejection loop over `verifier_message::<[u8; N]>()` |
| `challenge_small`, `challenge_bytes::<N>()` | `verifier_message::<[u8; N]>()` |
| `finish` | `narg_string()` (prover); `check_eof()` (verifier) |

**Seeded forks.** `Fork<H>::new(seed, counter)` is a fresh `H` that absorbed
the tag `jolt-transcript/fork/v1`, a 32-byte seed squeezed from the live
transcript, and `LE32(counter)`. Every prover search runs on forks:

1. Squeeze a 32-byte seed `s` from the transcript.
2. The prover tries counters `c` upward from zero, each on `Fork(s, c)`, off
   the transcript.
3. The prover sends the accepted `c` as a canonical unsigned LEB128 `Nonce`
   message. The verifier receives it, rejects a non-minimal encoding or a
   value outside the search range, and rebuilds `Fork(s, c)`.

Proof of work with difficulty `g > 0` searches `c` in `[0, 2^(g+7))` and
accepts when the fork's first 32 squeezed bytes have `g` leading zero bits,
least significant bit first. The protected challenge is drawn from the
transcript after the nonce. Akita's fold-response search (Fiat-Shamir with
aborts) draws each candidate's fold challenges from the fork and accepts the
first counter whose response meets its bounds. Both roles draw the accepted
challenges from the same fork. The transcript sees one squeeze and one nonce
message of at most five bytes per search, whatever the search cost. The seed binds the full
prechallenge state, and the counter is in the argument string.

**Replay checkpoint.** The ZK prover replays its argument string through the
verifier's stage spine before BlindFold. With no previews, every ZK transcript
squeezes a 32-byte checkpoint after the stage spine. The prover compares its
own checkpoint with the replay's. **Parse-ahead.** dory-pcs needs its whole
proof struct before verifying, so `VerifierTranscript::unread()` exposes the
unread bytes for parsing. The live transcript still receives and absorbs every
one of them.

**Sites and logging.** `Channel::site` and the event log stay in the wrapper.
Byte ranges come from the known atom widths. Logging stays feature-gated.

**Cloning.** `ProverTranscript` is not `Clone`, matching `ProverState`.
Fixtures that cloned a mid-proof transcript rebuild it from a closure.

## Non-goals

- Transcript-bound prover randomness. BlindFold and the committed recorders
  draw blindings from the operating system's RNG, not from spongefish's
  `ProverState::rng()`. Binding them to the transcript is a separate change;
  the `StdRng` inside `ProverState` is unused.

- Algebraic sponges (`H::U` a field element). The encoding traits are
  implemented over `[u8]` only. The byte-level `PoseidonSponge` emits only the
  low 16 bytes of each permutation's output, whose distribution is within
  $2^{-125}$ of uniform, so challenges drawn from it stay exact up to that
  distance. Supporting a field-native Poseidon later means
  adding `Encoding<[Fr]>` impls. It does not change the typed layer.
- Changing message order or adding labels. A caller's own framing stays its
  own, absorbed as public messages.

## Alternatives considered

- **Statistical challenges via spongefish's field `Decoding`** (modulus bytes
  plus 32, then reduce). Rejected. Exact sampling works on spongefish state
  through looped fixed-width squeezes, and it keeps the rejection-sampling
  query accounting.
- **Remove `CanonicalBytes` and bound on spongefish's traits directly.**
  Rejected. The wrapper needs a fixed width per atom to check a read against
  the remaining proof before allocating.
- **Keep the preview-based grinding transition with `yolocrypto` confined to
  `jolt-transcript`.** Rejected. Cargo unifies features, so the duplex state
  would become public to every crate in the graph.
- **`spongefish-pow` (Blake3 or Keccak) for the predicate.** Rejected. It adds
  a second hash function to the security assumptions. For a Poseidon
  transcript it also makes in-circuit grind verification expensive. The
  predicate hashes with the transcript's own sponge type.

## Compatibility

Every proof changes. The domain-separator layout and the grinding transition
differ; the nonce keeps its canonical unsigned LEB128 encoding. Fixtures,
known-answer tests, the FS census, and Akita's grinding tables are
regenerated. Akita's `specs/transcript-grinding.md` is amended in its own PR.

## Execution

Two PRs, landed together. Akita pins the Jolt PR's revision.

1. **Jolt:** this spec. The spongefish codecs in `jolt-field` and
   `jolt-crypto`, the transcript core, the grinding transition, preview
   removal, then fixture and census regeneration.
2. **Akita:** port its proof-of-work and fold-response searches to forks
   (`PreviewFoldDraw` becomes a fork draw that both roles use), and amend
   `transcript-grinding.md`.

## Verification

- Clear, ZK, and Akita e2e; the verifier fixtures; tamper and FS sweeps; the
  census re-blessed and reviewed.
- Per-sponge known answers regenerated, plus a test that the exact sampler
  rejects an out-of-range candidate and resamples from the next squeeze, and
  one that pins each sponge's `ABSORB_FORGETS_SQUEEZE_WIDTH` to its actual
  absorb-after-squeeze behavior.
- Grinding: tests for accept, reject, an out-of-range nonce, and a wrong seed.
- `cargo tree -e features` contains no `yolocrypto` in either workspace.

## Unverified

- Prover cost of squeeze-then-grind versus the current transition. Both hash
  one block per candidate, so parity is expected.
