# Fiat-Shamir transcript

Jolt is an interactive protocol made non-interactive with the Fiat-Shamir
transform. The protocol has sumchecks, batching, polynomial openings, and
BlindFold. Wherever the verifier would send a random challenge, both parties
derive it from a hash of everything sent so far. A proof has one transcript,
which the prover and the verifier run in the same order: from the statement
through stage 8, and through BlindFold in ZK builds. The one exception is
[Akita](./akita.md)'s opening proof. It runs its own Fiat-Shamir session,
whose session tag is extended with one challenge drawn from Jolt's transcript.

The transcript lives in `crates/jolt-transcript`. It is built on the
[spongefish](https://github.com/arkworks-rs/spongefish) 0.8 duplex-sponge
framework. The normative design is in the
[transcript spec](https://github.com/a16z/jolt/blob/main/specs/2110-jolt-transcript-sealed-spongefish.md).

## The sponge

A duplex sponge has two operations. *Absorb* mixes bytes into the state.
*Squeeze* reads pseudorandom bytes out of it. A message is absorbed, and a
challenge is squeezed. Each squeeze depends on every byte absorbed before it.

The sponge is a build-time choice:

| Sponge | Selected by | Field challenges |
|--------|-------------|------------------|
| Keccak-f[1600] duplex sponge | default | yes |
| Blake2b-512 | `jolt-sdk` feature `transcript-blake2b` | yes |
| Poseidon (BN254, Circom parameters) | `jolt-transcript` feature `transcript-poseidon` | no |

Field challenges are decoded from raw squeezed bytes, so the sponge must squeeze
uniform bytes. Poseidon squeezes BN254 scalars, whose top bits are not
uniform. The type system therefore does not let a Poseidon transcript draw
field challenges, and Jolt proofs use Keccak or Blake2b.

Two other hashes appear around the sponge:

- **Session identifier.** SHAKE128 derives the spongefish session identifier
  from Jolt's transcript protocol tag. The prover's private randomness also
  comes from SHAKE128.
- **Digests.** Digests that the statement or a key binds, such as the
  preprocessing digest, use SHA3-256. With the `jolt-transcript` feature
  `keccak256-digests`, they use Keccak-256 (the Ethereum padding) instead.
  The two differ only in a padding byte, so a guest with a Keccak permutation
  instruction pays the same for either. The choice is part of the protocol
  tag, so a proof made under one never verifies under the other.

## How a proof's transcript is framed

**Protocol tag and session.** The crate-wide protocol tag,
`a16z/jolt-transcript/v2/sha3` padded to 32 bytes, names the transcript layout,
the challenge codecs, and the digest hash. SHAKE128 of the tag is the sponge's
session identifier, so a change to any of these changes every challenge.

**Statement binding.** Before the first message, the transcript absorbs the
instance:

- the 32-byte session tag of the protocol being proved (`Jolt` for a Jolt
  proof), then
- the public statement, as a length-prefixed byte string.

Jolt's statement contains:

- the protocol configuration: ZK or clear, Dory or Akita, field-inline
  settings, memory layout, trace length, RAM size, entry address, and the
  read-write, one-hot and trace-ordering parameters;
- the guest's panic flag;
- whether advice commitments are present;
- the preprocessing digest;
- the public inputs and outputs.

The only way to build a transcript is from a session tag and a statement, so
the statement cannot be skipped. The prover and the verifier encode the
statement through the same function.

**Positional, word-aligned messages.** Messages carry no labels. The
protocol's fixed schedule says what the next message is, so its position in
the transcript identifies it. Every absorb and every squeeze is a whole
number of 8-byte words:

- Field elements and compressed group elements have fixed widths that are
  multiples of 8 bytes.
- Integers and counts are one `u64` little-endian word.
- A byte string is its `u64` length, the bytes, and zero padding to the next
  word.
- A slice whose length is fixed by bound data is absorbed as its concatenated
  encodings, padded once at the end.

The sponge absorbs bytes with no separators. So the messages between two
squeezes must decode uniquely from their concatenation, given what the
verifier already knows. Each encoding therefore has a fixed width, has a
length fixed by data already bound, or carries its own length. This rule is a
review obligation, not a type-level check. The Fiat-Shamir inventories
described below record every absorb site, so that a change to any site
requires review.

**Public messages, prover messages, and the NARG string.** Spongefish
separates two kinds of message:

- A *prover message* is absorbed and also appended to the proof's argument
  string. Spongefish calls this the NARG string. The verifier reads prover
  messages back from it.
- A *public message* is absorbed by both sides and written nowhere.

A Jolt proof is a serialized structure: commitments, sumcheck round
polynomials, claims, and opening proofs. The verifier absorbs each field it
reads from that structure as a public message. Spongefish absorbs the same
bytes for both kinds of message, so the challenges match those of the same
protocol with every field sent through the NARG string. The NARG string of a
current Jolt proof is therefore empty. At the end of verification, the
verifier still calls `check_eof`, which rejects a proof whose NARG string has
unread bytes or whose transcript has recorded a failure. Without this check,
a valid proof followed by trailing bytes would verify, so proof bytes would
be malleable.

## Drawing challenges

Each draw is one squeeze. Each field's challenge codec decodes the squeezed
bytes so that one draw's challenges are within statistical distance $2^{-161}$
of uniform:

- **BN254** (`Fr`, `Fq`): each challenge reduces 56 squeezed bytes modulo $p$.
  Per element, the result is within $2^{-196}$ of uniform over the whole
  field. These are *full-width* challenges.
- **Pseudo-Mersenne fields**, including Akita's 128-bit field: rejection
  sampling from a pool of candidates. The pool is sized so that running out
  has probability at most $2^{-161}$.
- **Binary fields**: every byte pattern of the right length is a field
  element, so decoding is a bijection.

**Short challenges for Dory.** The Dory verifier multiplies group elements by
its challenges `beta`, `alpha` and `d`. A scalar multiplication costs time
proportional to the scalar's bit length. So these three challenges are drawn
as integers in $[0, 2^b)$ instead of full-width field elements.

The width $b$ comes from the step's *loss factor* $L$, the number of
challenge values that would let a false claim pass. $b$ is the least integer
with $2^b \ge L \cdot 2^{128}$. A bad challenge then appears with probability
at most $2^{-128}$, and no proof of work is needed:

| Challenge | Loss $L$ | Why | Width $b$ |
|-----------|----------|-----|-----------|
| `beta` | 2 | the round's checks are Laurent polynomials spanning $\beta^{-1}$ to $\beta$ | 129 |
| `alpha` | 4 | the fold spans $\alpha^{-1}$ to $\alpha$, and the last `alpha` shares a draw with `gamma` | 130 |
| `d` | 3 | the final pairing check spans $d^{-1}$ to $d^2$ | 130 |

`gamma` stays full width, because the verifier uses only $-\gamma$ and
$\gamma^{-1}$, which are full width anyway. The zero-knowledge
Sigma-protocol challenges also stay full width, because the verifier squares
them or multiplies them by other challenges. Dory names its challenges by
label, so the adapter in `crates/jolt-dory/src/transcript.rs` maps those
labels to losses. The adapter assumes Dory's draw order, so the workspace
pins dory-pcs to one exact version. Every other challenge in Jolt is full
width.

## Proof of work

Under Fiat-Shamir, a cheating prover can retry. Changing an earlier message
gives a fresh challenge at the cost of one hash call. Suppose a step loses
when its challenge falls in a bad set of $L$ values, and challenges are
uniform on a set $C$. Then each attempt succeeds with probability
$L / |C|$.

Over BN254, $|C| \approx 2^{254}$, so this is far below $2^{-128}$. Over a
128-bit field, a degree-$d$ sumcheck round gives only about
$128 - \log_2 d$ bits per hash call.

A *proof of work* of $b$ bits before the challenge makes each attempt cost
about $2^b$ hash calls:

1. The prover finds a nonce for which the next squeezed bytes pass a check.
2. The verifier checks the nonce with one squeeze.

The difficulty is the least $b$ with $|C| \cdot 2^b \ge L \cdot 2^{128}$.
This restores 128 bits per protected challenge. The rule is computed with
exact integers from the challenge codec's set size.

The layout of one proof-of-work step:

- The nonce is a `u64` below $2^{b+7}$. The sponge absorbs it as one word.
  In the NARG string it is encoded as canonical LEB128. It can instead be
  carried in the proof structure and absorbed as a public message.
- Both sides squeeze 8 bytes. The verifier accepts if the low $b$ bits of
  their little-endian integer are zero.
- $b$ is at most 57, so the nonce space holds $2^7$ times the expected
  number of trials.
- The step protects the challenges drawn right after it, up to the next
  absorbed message.

The prover searches nonces in increasing order. It tests each candidate
against a copy of the public sponge state, so a failed attempt leaves the
transcript unchanged.

Protocols that carry no nonces check their challenge set at entry. BlindFold
and the committed (ZK) sumcheck refuse a field whose challenges would need
proof of work at the largest possible loss. BN254 passes this check, and
128-bit fields do not.

### Where Jolt grinds

Jolt grinds only where a challenge set is too small on its own. Today that is
Akita's 128-bit field. Every BN254 site prices to 0 bits, and a BN254 proof
carries no nonces. Jolt carries its nonces in the proof structure and absorbs
them as public messages, not through the NARG string.

- **Sumcheck rounds.** Each round of a clear sumcheck grinds after its round
  message and before its challenge. The loss is the verifier's degree bound
  for the round, never a degree the prover declares, so a proof cannot lower
  its own difficulty. The nonces sit next to the round messages, one per
  round when the difficulty is positive and none otherwise
  (`crates/jolt-sumcheck/src/pow.rs`).
- **Stage sites.** After some prover messages, the verifier draws a burst of
  challenges: stage 1's `tau`, the relation challenges and batching
  coefficients of stages 1 to 7, stage 2's `tau_high`, and Akita's stage-8
  selector points. One nonce, ground before the burst's first draw, covers
  the whole burst. Every site has the same flat loss, `SITE_LOSS = 1023`,
  about 4.5 times the largest burst's actual loss. A per-site count that
  drifted below the true loss would be a silent soundness bug; overpricing
  costs only a few extra hash calls. Over Akita's field this is 10 bits.
  The nonces sit in `StagePowNonces`, one slot per site. A slot holds a nonce
  exactly when the site is visited and its difficulty is positive
  (`crates/jolt-verifier/src/stages/pow.rs`).

## Testing

[Testing gates](../dev/testing-gates.md#fiat-shamir-soundness) lists the
Fiat-Shamir checks.

- **Inventories.** The `fs_obligations` inventories give every absorb site,
  every challenge site, and every serialized verifier input a stable
  identity. Changing the schedule then shows up as a reviewed diff.
- **Attack tests.** The `fs-attacks` tests make a claim false and show that
  the verifier still accepts it with frozen challenges. They then show that
  the real transcript changes a relevant challenge and rejects.
- **Symmetry.** The `jolt-eval` invariant
  `transcript_prover_verifier_consistency` checks, for each sponge, that a
  prover and a verifier running the same sequence of operations agree on
  every message and every challenge.
- **Crate tests.** `jolt-transcript` pins the protocol tag, the session
  identifier, the challenge codecs, and the proof-of-work layout with
  known-answer tests.
