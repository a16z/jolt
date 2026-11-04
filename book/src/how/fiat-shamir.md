# Fiat-Shamir transcript

Jolt is an interactive protocol made non-interactive with the Fiat-Shamir
transform. Wherever the verifier would send a random challenge, the prover and
the verifier both derive it by hashing the statement and every message
before it. This chapter describes that transcript: how messages are framed,
how challenges are drawn, and where proof of work fits in.

Jolt has two proof configurations, and parts of this chapter apply to only
one of them:

| Configuration | Field | Opening | Zero knowledge |
|---------------|-------|---------|----------------|
| Dory | BN254 scalar field | [Dory](./dory.md) | optional ([BlindFold](./blindfold.md)) |
| Akita | 128-bit prime field | [Akita](./akita.md) | not supported |

In both configurations, one transcript runs from the statement through
stage 7. Dory's opening at stage 8, and BlindFold in ZK builds, continue on
that same transcript. Akita's opening at stage 8 runs its own Fiat-Shamir
session. That session's tag is extended with one challenge drawn from Jolt's
transcript, which ties the opening to everything Jolt bound before it.

The transcript lives in `crates/jolt-transcript`. It is built on the
[spongefish](https://github.com/arkworks-rs/spongefish) 0.8 duplex-sponge
framework. The normative design is in the
[transcript spec](https://github.com/a16z/jolt/blob/main/specs/2110-jolt-transcript-sealed-spongefish.md).

## The sponge

A duplex sponge has two operations. *Absorb* mixes bytes into the state.
*Squeeze* reads pseudorandom bytes out of it. Messages are absorbed, and
challenges are squeezed, so each challenge depends on every byte absorbed
before it.

The sponge is a build-time choice:

| Sponge | Selected by | Field challenges |
|--------|-------------|------------------|
| Keccak-f[1600] | default | yes |
| Blake2b-512 | `jolt-sdk` feature `transcript-blake2b` | yes |
| Poseidon (BN254, Circom parameters) | `jolt-transcript` feature `transcript-poseidon` | no |

Field challenges are decoded from squeezed bytes, so the codecs need a sponge
whose output bytes are uniform. Poseidon squeezes BN254 scalars, whose top
bits are not uniform. A Poseidon transcript therefore cannot draw field
challenges: the trait bound fails at compile time. It exists for symmetry
tests and absorb-only use, and Jolt proofs use Keccak or Blake2b.

Two other hashes appear around the sponge:

- **Session identifier.** SHAKE128 (FIPS 202) derives the spongefish session
  identifier from the crate's protocol tag (below).
- **Digests.** Digests bound into the statement or a key, such as the
  preprocessing digest, use SHA3-256. With the `jolt-transcript` feature
  `keccak256-digests`, they use Keccak-256 instead. The two hashes run the
  same permutation and differ only in padding, so a guest verifier with a
  Keccak instruction pays about the same for either. The choice is part of
  the protocol tag, so a proof made under one never verifies under the
  other.

## Framing a proof's transcript

**Protocol tag and session.** The crate-wide protocol tag,
`a16z/jolt-transcript/v2/sha3` zero-padded to 32 bytes, versions the absorb
layout, the challenge codecs, the session-identifier derivation, the
proof-of-work layout, and the digest hash. The session identifier is derived
from it, so changing any of these changes every challenge.

**Statement binding.** Before the first message, the transcript absorbs the
instance:

1. the 32-byte session tag of the protocol being proved (`Jolt` for a Jolt
   proof), then
2. the public statement, as a length-prefixed byte string.

Jolt's statement contains:

- the protocol configuration: ZK or clear, Dory or Akita, the field-inline
  settings, the memory layout, the trace length, the RAM size, the entry
  address, and the read-write, one-hot and trace-ordering parameters;
- the guest's panic flag;
- which advice commitments are present;
- the preprocessing digest, which commits to the program;
- the public inputs and outputs.

The transcript constructors take a session tag and a statement, so no
transcript exists without them. That the statement is the *right* one is up
to the caller: in Jolt, the prover and the verifier both build it in
`validate_and_seed_transcript`, so they encode it the same way.

**Positional, word-aligned messages.** Messages carry no labels. The
protocol's schedule fixes what the next message is, so a message's position
identifies it. The transcript's methods absorb whole 8-byte words:

- Jolt's field elements (32 bytes on BN254, 16 on Akita's field) and
  compressed group elements have fixed widths that are multiples of 8 bytes.
  `append` takes one such value. A narrower encoding, such as a 4-byte field
  element, trips a debug assertion.
- Integers and counts are one `u64` little-endian word.
- A byte string is its `u64` length, the bytes, and zero padding to the next
  word.
- `append_values` absorbs a slice whose length is fixed by data already
  bound: the concatenated encodings, padded once at the end.

A count or length is absorbed only where the prover chooses it. Word
alignment holds for every call through these methods. It is a convention, not
a type-level guarantee: spongefish's own `public_message` absorbs any
encoding.

The sponge absorbs bytes with no separators. So the messages between two
squeezes must decode uniquely from their concatenation, given what the
verifier already knows. Each encoding therefore has a fixed width, has a
length fixed by data already bound, or carries its own length. This rule is a
review obligation, not a type-level check. The Fiat-Shamir inventories
(see [Testing](#testing)) record every absorb site in the verifier's
dependency closure, so changing a site shows up for review. Absorbs inside
dory-pcs and Akita are outside the inventories.

**Public messages, prover messages, and the NARG string.** Spongefish
separates two kinds of message, which enter the sponge the same way:

- A *prover message* is also appended to the proof's argument string, which
  spongefish calls the NARG string. The verifier reads it back from there.
- A *public message* is absorbed by both sides and written nowhere.

A Jolt proof is a serialized structure of commitments, sumcheck messages,
claims, and opening proofs. The verifier absorbs each field it reads from
that structure as a public message, so Jolt's NARG string is empty. The
verifier still ends with `check_eof`, which rejects unread NARG bytes and any
failure the transcript recorded, such as an exhausted rejection sampler.
Without it, a valid proof with bytes appended to its NARG string would also
verify.

The transcript binds the values the verifier decoded from the proof
structure, not the bytes it decoded them from. Rejecting non-canonical or
trailing bytes in the proof structure is the proof decoder's job.

## Drawing challenges

Each challenge draw is one squeeze. A field's challenge codec decodes the
squeezed bytes so that one draw, of one or many challenges, is within
statistical distance $2^{-161}$ of uniform:

- **BN254** (`Fr`, `Fq`): each challenge reduces 56 squeezed bytes modulo
  $p$, which leaves each element within $2^{-196}$ of uniform over the whole
  field. These are *full-width* challenges.
- **Pseudo-Mersenne fields**, including Akita's: rejection sampling from a
  pool of candidates. The pool is sized so that it runs out with probability
  at most $2^{-161}$. If it does run out, the draw fails and so does the
  proof, so this costs completeness, not soundness.
- **Binary fields**, which `jolt-transcript` supports but Jolt does not prove
  over: every byte pattern of the right length is a field element, so
  decoding is a bijection.

Every challenge drawn on Jolt's transcript is full width, including the ones
Dory's verifier multiplies into group elements. The Dory adapter
(`crates/jolt-dory/src/transcript.rs`) ignores dory-pcs's challenge labels
and draws each challenge with `challenge`. Akita's opening draws its own
challenges in its own session.

## Proof of work

Under Fiat-Shamir, a cheating prover can retry: changing an earlier message
gives a fresh challenge for the cost of one hash call. Suppose a step lets a
false claim through when its challenge lands in a bad set of $L$ values, and
challenges are uniform on a set $C$. Then each attempt succeeds with
probability $L / |C|$.

- On BN254, $|C| \approx 2^{254}$. Every loss Jolt prices leaves this far
  below $2^{-128}$.
- On a 128-bit field it does not. A degree-$d$ sumcheck round gives only
  about $128 - \log_2 d$ bits per hash call.

A *proof of work* of $b$ bits before the challenge raises the expected cost
of each attempt to about $2^b$ hash calls:

1. The prover finds a nonce for which the next squeezed bytes pass a check.
2. The verifier checks the nonce with one squeeze.

The difficulty is the least $b$ with $|C| \cdot 2^b \ge L \cdot 2^{128}$,
computed with exact integers from the codec's challenge-set size. This puts
each protected challenge back at about $2^{128}$ hash calls. This is a
per-challenge (round-by-round) bound, not a bound summed over the whole
proof.

The layout of one proof-of-work step:

- The nonce is a `u64` below $2^{b+7}$, and the sponge absorbs it as one
  word. It travels either in the NARG string, encoded as canonical LEB128, or
  in the proof structure, absorbed as a public message.
- Both sides then squeeze 8 bytes. The verifier accepts if the low $b$ bits
  of their little-endian integer are zero.
- $b$ is at most 57, so the nonce space holds $2^7$ times the expected
  number of trials.
- The nonce protects the challenges drawn right after it, up to the next
  absorbed message.

The prover tries nonces in increasing order. It tests each one against a copy
of the public sponge state, so a failed attempt leaves the transcript
unchanged. The search runs only under the `prover-grinding` feature, which
`jolt-prover` enables and verifier-only builds leave off.

Protocols that carry no nonces check their challenge set when they start:
they refuse a field whose challenges would need proof of work at any loss up
to $2^{64}$. The committed sumcheck and BlindFold do this. Both exist only in
ZK builds, which run on BN254 and pass the check. A 128-bit field fails it.

### Where Jolt grinds

Jolt grinds only where a challenge set is too small on its own, which today
means Akita's 128-bit field. Every BN254 site prices to 0 bits, so a Dory
proof carries no nonces. Jolt carries its nonces in the proof structure and
absorbs them as public messages.

- **Sumcheck rounds.** Each round of a clear sumcheck grinds after its round
  message and before its challenge. The loss is the verifier's degree bound
  for the round, never a degree the prover declares, so a proof cannot lower
  its own difficulty. The nonces sit next to the round messages, one per
  round when the difficulty is positive and none otherwise
  (`crates/jolt-sumcheck/src/pow.rs`).
- **Stage sites.** After some prover messages, the verifier draws a burst of
  challenges with no message in between: stage 1's `tau`, the relation
  challenges and batching coefficients of stages 1 to 7, stage 2's
  `tau_high`, and the selector points of Akita's stage-8 opening. One nonce,
  ground before the burst's first draw, covers the whole burst. Every site
  has the same flat loss, `SITE_LOSS = 1023`, about 4.5 times the largest
  per-burst loss bound worked out in `crates/jolt-verifier/src/stages/pow.rs`.
  A per-site count that drifted below the true loss would be a silent
  soundness bug; overpricing costs only a few extra hash calls. On Akita's
  field this is 10 bits. The nonces sit in `StagePowNonces`, one slot per
  site. A slot must hold a nonce exactly when the site is visited and its
  difficulty is positive.

Challenges drawn inside Akita's opening are Akita's to protect.

## Testing

[Testing gates](../dev/testing-gates.md#fiat-shamir-soundness) lists the
Fiat-Shamir checks:

- **Inventories.** The `fs_obligations` inventories give a stable identity to
  every absorb site, challenge site, and serialized verifier input in the
  verifier's dependency closure. A schedule change then shows up as a
  reviewed diff.
- **Attack tests.** The `fs-attacks` tests, run for Dory clear, Dory ZK and
  Akita clear, make one claim false and show that the verifier accepts it
  when its challenges are frozen. They then show that the real transcript
  changes a relevant challenge and rejects.
- **Symmetry.** The `jolt-eval` invariant
  `transcript_prover_verifier_consistency` checks, for each sponge, that a
  prover and a verifier running the same operations agree on every message
  and every challenge.
- **Crate tests.** `jolt-transcript` pins the protocol tag, the session
  identifier, the challenge codecs, and the proof-of-work layout with
  known-answer tests.
