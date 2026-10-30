# Spec: One Sealed Transcript over Spongefish States

| Field       | Value                                           |
|-------------|-------------------------------------------------|
| Author(s)   | @quangvdao                                      |
| Created     | 2026-10-29                                      |
| Status      | in review                                       |
| PR          | [#2110](https://github.com/a16z/jolt/pull/2110) |

## Summary

Jolt's Fiat-Shamir transcript had two surfaces in `crates/jolt-transcript`: a
label-based `Transcript`/`AppendToTranscript` facade kept for source
compatibility, and split spongefish-native traits. Protocol crates went
through the facade. Its `Transcript::new(label)` bound no statement, every
absorb carried a constant label, and `challenge` squeezed 16 bytes for every
sponge, Poseidon included. The facade was also unsealed, so any crate could
implement it and draw challenges outside the audited codecs. PR #2110 replaces both with one
transcript: spongefish 0.8.0's `ProverState` and `VerifierState`, reached only
through sealed traits whose every method has one implementation shared by
prover and verifier. Absorbs are positional and word aligned, the statement is
bound when the transcript is built, every field challenge is full width, and
a proof-of-work mechanism is in place for the
protocol sites that #2108 adds. All callers are ported in the same PR. This
spec supersedes [`jolt-transcript-spongefish.md`](./jolt-transcript-spongefish.md)
(#1455), which describes the earlier two-surface design.

## Intent

### Goal

Make `jolt-transcript` the single Fiat-Shamir transcript for every Jolt
protocol: spongefish's prover and verifier states behind four sealed traits,
with a positional, word-aligned wire layout fixed by one protocol tag.

Key abstractions:

- **`Transcript`** (`src/transcript.rs`): operations both sides run alike.
  `append` (one whole-word encoding), `append_values` (a slice concatenated and
  zero-padded once), `append_bytes` (a length-prefixed `BytesMsg`), and
  `check_public_grind` (a proof of work whose nonce the proof carries outside
  the NARG string).
- **`FieldTranscript<F>`**: challenges in `F` through `F`'s sealed
  `ChallengeCodec`: `challenge`, `challenge_vector` and `challenge_powers`.
  It exposes `CHALLENGE_SET`, the exact
  size of one challenge's range, for proof-of-work pricing.
- **`ProverTranscript`** (`src/prover.rs`): `send` (a prover message written
  to the NARG string), `narg_string`, and, under `prover-grinding`,
  `verifier_message_oracle`, `grind` and `public_grind`.
- **`VerifierTranscript`** (`src/verifier.rs`): `prover_message`,
  `check_grind`, and `check_eof`.
- **`ProverInit` / `VerifierInit`**, together with `prover_transcript` and
  `verifier_transcript` (`src/setup.rs`): the only constructors, each taking a
  32-byte session tag and the public statement.
- **Sealing**: `Transcript` has the private supertrait `sealed::Sealed`, with
  three primitives: `absorb`, `squeeze` and `require`. `Sealed` is implemented
  only for `spongefish::ProverState`, `spongefish::VerifierState`, and the
  test-only `ScriptedTranscript` (`test-utils`). `ChallengeCodec` and
  `UniformBytes` are sealed too.

### Invariants

1. **Prover/verifier symmetry.** Every `Transcript` and `FieldTranscript`
   method is a provided method or a blanket implementation over the three
   `Sealed` primitives. `ProverState` and `VerifierState` differ only in those
   primitives: `require` returns an error on the prover and poisons the state
   on the verifier. A prover and a verifier that run the same schedule from
   the same session and statement absorb identical bytes and draw identical
   challenges. Checked by the jolt-eval invariant
   `transcript_prover_verifier_consistency`
   (`jolt-eval/src/invariant/transcript_symmetry.rs`), which uses all three
   sponges, and end to end by every clear, ZK and Akita proof that verifies.
2. **Positional, word-aligned framing.** No message carries a label. A
   message's position in the protocol schedule identifies it. Every absorb and
   every squeeze is a whole number of 8-byte words. The messages absorbed
   between two squeezes must decode uniquely from their concatenation, given
   what the verifier already knows. This means each encoding has a fixed
   width, has a length fixed by bound data, or carries its own `u64` count or
   length, as `BytesMsg` and a committed sumcheck's `CommittedOutputClaims`
   do. Enforcement:
   - `Transcript::append` has a `debug_assert` on word alignment.
   - `append_values` and `BytesMsg` pad by construction.
   - Fixed widths of field and group encodings are type properties.
   - Unique decodability is not machine-checked. It is a review obligation
     recorded per site by the FS inventories (see Testing Strategy). The
     `encoding_injectivity` fuzz target checks `BytesMsg` framing.
3. **Session binding.** `PROTOCOL_ID` is the 32-byte tag
   `a16z/jolt-transcript/v2/sha3`, or `.../v2/keccak` under
   `keccak256-digests`. It determines the spongefish session id through
   SHAKE128 (`session_id()`). Changing the layout, a codec, or the digest hash
   requires a new tag. Pinned by the known-answer tests in `src/setup.rs`.
   The sponge is not named in the tag: a different sponge is a different hash
   function.
4. **Statement binding.** The spongefish instance is the caller's 32-byte
   session tag followed by the statement as a `BytesMsg`. It is absorbed before
   any message. The constructors are the only way to build a transcript, so
   the statement cannot be skipped. For Jolt, the session tag is
   `JOLT_SESSION = protocol_tag(b"Jolt")` and the statement is `JoltStatement`
   (`crates/jolt-verifier/src/verifier.rs`): 19 configuration words, the
   advice-presence word, the 32-byte preprocessing digest, and the public
   inputs and outputs. The prover and the verifier both encode it through
   `validate_and_seed_transcript`. Digests bound into the statement or a key
   use `DigestHasher`, which is SHA3-256 by default and Keccak-256 under
   `keccak256-digests`.
5. **Proof termination.** `check_eof` consumes the verifier transcript and
   fails if NARG bytes are left unread or the state is poisoned. Both
   `jolt_verifier::verify` entry points call it on their success path
   (`VerifierError::NargNotConsumed`).
6. **Challenge width.** Field challenges come from each field's
   `ChallengeCodec`, sized so that one message is within total variation
   2^-161 of uniform.
   - BN254 `Fr` and `Fq` reduce 56-byte chunks, so challenges are full width.
   - Pseudo-Mersenne fields, including Akita's 128-bit field, use pooled
     rejection sampling.
   - Binary fields decode bijectively.
   - Field challenges require a `UniformBytes` sponge: Keccak or Blake2b512.
     `PoseidonSponge` squeezes BN254 scalars, which are not uniform bytes, so
     a Poseidon transcript cannot draw field challenges. This is a compile
     error, not a runtime check.
7. **Proof-of-work bounds.**
   - Difficulty: `grind_bits_for_loss(L, |C|, λ)` returns the least `b ≤
     MAX_POW_BITS = 57` with `|C| · 2^b ≥ L · 2^λ`, computed with exact
     256-bit integers.
   - Nonce: a `u64` below `2^(b+7)`. The sponge absorbs it as one LE word. In
     the NARG string it is canonical LEB128, and outside the NARG string
     (`check_public_grind`) it is a `u64`.
   - Check: the low `b` bits of the next 8 squeezed bytes are zero.
   - The verifier rejects difficulties above `MAX_POW_BITS`, and nonces that
     are out of range or non-canonical, before it accepts.
   - `require_no_pow(CHALLENGE_SET)` fails closed where a protocol carries no
     nonces. BlindFold prove and verify call it, as do both committed-sumcheck
     paths. On BN254 it passes. On a 128-bit field it fails.
8. **Sealed boundary.** No crate outside `jolt-transcript` can implement
   `Transcript`, `ChallengeCodec` or `UniformBytes`. So no caller can draw
   challenges except through an audited codec, and no caller can build a
   transcript without binding a session and a statement.

### Non-Goals

- **Protocol proof-of-work sites.** In this PR, no Jolt protocol step calls
  `grind`, `check_grind` or `public_grind`. #2108 adds the sumcheck-round and
  stage-site nonces; the book chapter's *Where Jolt grinds* describes them.
- **Moving Akita's opening into Jolt's sponge.** In this PR, Akita's opening
  runs its own nested Fiat-Shamir session, whose tag is extended with one Jolt
  challenge. #2111 runs the opening in Jolt's sponge through
  `GroupBatchOpening` and `check_eof`, so the Akita opening proof is carried
  in the NARG string.
- **A protocol Poseidon transcript.** `PoseidonSponge` is kept for symmetry
  tests and absorb-only use. A native-`Fr` Poseidon sponge is #2113.
- **Wire compatibility with earlier proofs.** Every challenge changes, and
  `PROTOCOL_ID` moves from v1 to v2.
- **Parallel nonce search.** The search is sequential.

## Evaluation

### Acceptance Criteria

- [ ] `jolt-transcript` exports no label-based API. `Transcript`,
      `AppendToTranscript`, `SpongeTranscript`, `DigestTranscript`, `Label`,
      `LabelWithCount` and `U64Word` are deleted.
- [ ] `Transcript`, `FieldTranscript<F>`, `ProverTranscript` and
      `VerifierTranscript` are implemented only on spongefish's `ProverState`
      and `VerifierState`, plus `ScriptedTranscript` under `test-utils`. The
      supertrait `Sealed` is crate-private.
- [ ] `PROTOCOL_ID` and `session_id()` match the known answers in
      `src/setup.rs` under both digest features.
- [ ] Every absorb in the verifier's dependency closure appears in
      `crates/jolt-verifier/tests/fs_inventory/absorb-sites.inventory`, and
      every challenge draw in `challenge-sites.inventory`. The inventories are
      regenerated and the diff is reviewed.
- [ ] Sumcheck, openings, Dory, HyperKZG, Akita, BlindFold, the verifier, the
      prover, the SDK, jolt-eval and the fuzz harnesses compile against the
      new traits, with no remaining call sites of the old facade.
- [ ] Clear, ZK and Akita proofs verify, and tampered proofs are rejected
      (e2e matrix and verifier fixtures).
- [ ] `check_eof` rejects trailing bytes and unread messages
      (`tests/narg_eof_tests.rs`).
- [ ] Proof of work:
  - `grind` and `check_grind` agree on every sponge.
  - Forged, non-canonical, truncated and out-of-range nonces are rejected.
  - Difficulties up to 25 keep their earlier nonces (`tests/pow_tests.rs`).
- [ ] `grind_bits_for_loss` matches an unbounded big-integer oracle, and
      `require_no_pow` holds exactly for wide challenge sets
      (`src/grind_bits.rs` tests).

### Testing Strategy

Crate tests (`crates/jolt-transcript`):

- `src/setup.rs`, `src/digest_hash.rs`: known answers for `PROTOCOL_ID`,
  `session_id()`, the instance encoding, and both digest hashes.
- `tests/challenge_tests.rs`, `src/challenge.rs`: codec round trips, word
  alignment of pooled messages, the byte cap, rejection exhaustion poisoning
  the verifier, the Keccak rate-boundary known answers, and `append_values`
  padding.
- `tests/pow_tests.rs`, `src/grind_bits.rs`: the manual-layout replay, forged
  and malformed nonces, oracle equivalence, the pinned low-difficulty proofs,
  and `u64` public nonces.
- `tests/narg_eof_tests.rs`, `tests/native_traits_tests.rs`: EOF rejection,
  and divergence under distinct sessions or instances.
- `tests/{keccak,blake2b,poseidon}_tests.rs`: per-sponge known vectors.
- `fuzz/`: `encoding_injectivity` and `transcript_determinism`.

CI runs these in three configurations (`.github/workflows/rust.yml`):
`--all-features`, every sponge with SHA3-256 digests, and sequential grinding.

Cross-crate gates:

- **jolt-eval.** `transcript_prover_verifier_consistency` has one Test/RedTeam
  invariant each for Blake2b512, Keccak and Poseidon, and a merged fuzz target
  in which the fuzzer picks the sponge. Each replays public messages, prover
  messages and challenges on both sides and asserts round trips and equal
  challenges. No other jolt-eval invariant changes.
- **FS obligations.** `cargo nextest run -p jolt-verifier --test
  fs_obligations --features fs-audit` checks the absorb, challenge, scope and
  source-schema inventories against the source. Regenerating them with
  `JOLT_FS_BLESS=1` is a reviewed protocol change. See
  [Testing gates](../book/src/dev/testing-gates.md#fiat-shamir-soundness).
- **FS attacks.** `scripts/ci/fs-soundness.sh` (the `fs-attacks-smoke` job)
  covers Dory clear, Dory ZK and Akita clear. Each attack must verify with
  frozen challenges and be rejected with the production transcript.
- **End to end.** The e2e matrix and verifier fixtures run in clear, ZK and
  Akita modes, as listed in `CLAUDE.md`.

### Performance

No regression is expected. Positional absorbs remove the label bytes that the
facade absorbed. Dory's `beta`, `alpha` and `d`, which the verifier
multiplies into group elements, are now full-width 254-bit scalars instead of
the previous 128-bit ones, so those multiplications cost more; #2174's Dory
verifier optimizations are expected to offset this. The `transcript_ops` Criterion bench runs
in `bench-crates.yml`, base against PR on one runner. Results are reported
with a 10% noise threshold and do not gate the PR. No jolt-eval objective is
expected to move beyond noise.

## Design

### Architecture

```
caller (sumcheck, openings, Dory, HyperKZG, Akita, BlindFold, stages)
   │  Transcript / FieldTranscript<F> / ProverTranscript / VerifierTranscript
   ▼
provided methods ──► sealed::Sealed { absorb, squeeze, require }
   ▼
spongefish::ProverState<H, Shake128>      spongefish::VerifierState<'_, H>
   H ∈ { Keccak (default), Blake2b512, PoseidonSponge }
```

- **Initialization.** `session_id()` is SHAKE128 of `PROTOCOL_ID`. The
  instance is `session_tag ‖ BytesMsg(statement)`. The prover's private coins
  come from SHAKE128 (`ProverState<H>` in `src/shake128.rs`).
- **Public and prover messages.** A public message is absorbed by both sides
  and writes nothing to the NARG string. A prover message (`send` and
  `prover_message`) is absorbed the same way and written to the NARG string.
  `JoltProof` is a serialized structure: every field the verifier reads is
  absorbed as a public message. `proof.narg` is empty for Dory proofs and,
  after #2111, holds the Akita opening proof. The verifier requires it to be
  read to the end.
- **Challenges.** `draw_challenges` squeezes `challenge_bytes(len)` bytes and
  decodes them with the field's codec. Requests over `MAX_CHALLENGE_BYTES`
  (16 MiB) are rejected before any sponge operation.
- **Proof of work.** `SpongePow` (`src/pow.rs`) owns the layout.
  `verifier_message_oracle` clones the public sponge state into an owned
  closure; it holds no private coins and no proof bytes. `grind` scans nonces
  in increasing order against that closure on the calling thread, then sends
  the least passing nonce through the ordinary path and re-checks it. The
  oracle's shape follows arkworks-rs/spongefish#306, and the layout and
  difficulty rule follow #301 and #307. Spongefish 0.8.0 exposes the sponge
  state only under `yolocrypto`, so the search is behind `prover-grinding`.
  `jolt-prover` enables it, and verifier-only builds leave it off. In 0.8.0
  the feature changes visibility only.
- **Dory adapter.** `JoltToDoryTranscript` implements dory-pcs's
  `DoryTranscript` over any `FieldTranscript<Fr>`. It drops Dory's labels,
  absorbs group elements through `append_values` (fixed compressed width), and
  draws every challenge full width with `challenge`. Dory's trait cannot
  return errors, so a refused draw yields zero and makes `finish()` return
  `OpeningsError::TranscriptRejected`.
- **SDK.** `jolt-sdk` selects `ProtocolSponge = Keccak`, or `Blake2b512` with
  its `transcript-blake2b` feature.

### Alternatives Considered

1. **Keep the label facade alongside native traits (#1455's design).**
   Rejected. The two surfaces framed messages differently: the facade used a
   marker byte and a length for `append_bytes`, and a label word before each
   value. The facade's `new(label)` bound no statement, and its 16-byte
   challenges bypassed the codecs. One sealed surface is smaller to audit, and
   the FS inventories identify each absorb by its call expression.
2. **Absorb labels as extra public messages.** Rejected. Positional order
   already separates messages in a fixed schedule. Labels cost sponge work and
   gave no binding the schedule does not already give.
3. **Unsealed traits.** Rejected. A downstream implementation could draw
   challenges from non-uniform bytes or skip statement binding, which would
   void the codec bounds.
4. **Short (128-bit) challenges everywhere.** Rejected. A 128-bit challenge
   over BN254 gives 128 bits only for loss 1. Sumcheck rounds and batching
   draws have larger losses and would need proof of work. Full-width BN254
   challenges make `require_no_pow` hold, so BlindFold and committed sumcheck
   need no nonces.
5. **Short Dory challenges.** Rejected. Dory's verifier multiplies group
   elements by `beta`, `alpha` and `d`, so drawing them at 129 to 130 bits
   (the least width that keeps a bad set of 2 to 4 values below 2^-128)
   would make those multiplications about half as expensive. It needs a
   loss table keyed by dory-pcs's labels, a test pinning dory-pcs's draw
   order, and an exact dory-pcs version pin. Verifier time on the curve is
   not a current priority, and #2174's Dory verifier optimizations more than
   offset the difference, so the surface area is not worth it.
6. **Parallel nonce search (`MessageSearch`/`Probe` with rayon).** Rejected
   for this PR. The sequential search over an owned oracle needs no `Sync`
   probe and no rayon in the transcript crate. At the difficulties #2108
   plans (about 10 bits on Akita's field), search time is small next to
   proving time.
7. **A `u32` nonce with `MAX_POW_BITS = 25`, LEB128 one byte per message.**
   Replaced by a `u64` nonce absorbed as one word. This keeps the transcript
   word aligned and raises the cap to 57 bits. Every difficulty up to 25 keeps
   its nonces.
8. **TurboSHAKE128 for the session id.** Replaced by FIPS 202 SHAKE128.
   Spongefish 0.8 builds its SHAKE128 suite only together with a non-FIPS XOF,
   so the crate supplies its own (`src/shake128.rs`).

## Documentation

New book chapter [Fiat-Shamir transcript](../book/src/how/fiat-shamir.md),
linked from `SUMMARY.md` and `how-it-works.md`. `dev/testing-gates.md` no
longer refers to absorb labels.

## Execution

Implemented in #2110. Follow-ups:

- **#2108**: proof of work at protocol sites.
  - Sumcheck rounds are priced by degree. Clear proofs carry their nonces
    outside the NARG string.
  - Stage sites use a flat `SITE_LOSS = 1023`: 10 bits over Akita's 128-bit
    field and 0 on BN254.
  - `StagePowNonces` is unconditional.
- **#2111**: Akita's opening runs inside Jolt's sponge (`GroupBatchOpening`
  and `check_eof`), replacing the bridged nested session.
- **#2113**: a width-5 native-`Fr` Poseidon sponge. The byte-based
  `PoseidonSponge` kept here resists only about 2^127 hash calls, below the
  2^128 target, so it is not a protocol sponge.

## References

- [spongefish](https://github.com/arkworks-rs/spongefish) 0.8.0. Upstream
  proposals: #301 (proof-of-work layout), #306 (`verifier_message_oracle`),
  #307 (difficulty rule).
- draft-irtf-cfrg-fiat-shamir: session-id derivation and the SHAKE128 suite.
- A. Chiesa and M. Orrù, on the duplex-sponge Fiat-Shamir reduction cited in
  `ProverTranscript::grind`'s security note.
- [`jolt-transcript-spongefish.md`](./jolt-transcript-spongefish.md): the
  superseded #1455 design.
- `crates/jolt-transcript/src/setup.rs` (`PROTOCOL_ID` documents the v2
  layout), `src/grind_bits.rs`, `src/challenge.rs`,
  `crates/jolt-dory/src/transcript.rs`.
