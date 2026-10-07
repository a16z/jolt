# Spec: One NARG transcript for Jolt and Akita

| Field     | Value              |
| --------- | ------------------ |
| Author(s) | @markosg04         |
| Created   | 2026-10-01         |
| Status    | implemented (in review); amended |
| PR        | (Jolt) / (Akita)   |
| Amended by | `specs/jolt-transcript-spongefish-state.md`: the transcript runs on spongefish's prover and verifier state, `Preview<H>` is removed, and grinding and fold-response search run on seeded forks. Sections below that describe previews are superseded. |

## Summary

Jolt and Akita each own a Fiat-Shamir transcript. Jolt's production path is a
symmetric absorb/squeeze facade over a Blake2b-256 digest chain
(`LegacyBlake2bTranscript`), with proof data carried in a typed `JoltProof`
struct that the verifier must remember to absorb field by field. Akita moved to
a spongefish NARG proof stream (#37) with exact field sampling, grinding, and
site diagnostics, in its own `akita-transcript`. Jolt wraps Akita's opening in a
nested Akita channel whose session is seeded by one squeezed "bridge" challenge
(`bridged_akita_session`, `jolt-akita/src/adapters.rs:1323`). The bridge exists only because the two
transcripts cannot compose.

This spec replaces both with one `jolt-transcript` built on the NARG model:
the proof *is* the argument string, every prover message is absorbed exactly as
transmitted, and protocols compose by sharing a channel. Jolt's proof becomes
the NARG. Akita runs on the caller's channel, so it nests inside Jolt with no
bridge and `akita-transcript` is deleted. The sponge is a type parameter;
choosing a different one is a one-line alias change.

Delivery is two co-designed PRs, one per repository, with no compatibility
layer and no follow-up migration PR.

## Intent

### Goal

`jolt-transcript` provides a role-generic proof channel, prover and verifier
ends, exact and small-set challenges, a preview handle for proof of work, and
feature-gated site diagnostics. Messages are values with a canonical codec
owned by their type. It is expressive enough that Akita
uses it unchanged, and Jolt's prover, verifier, BlindFold, Dory, and Akita
integration all run on it.

### Ownership

| Component | Owner crate | Notes |
| --- | --- | --- |
| `Sponge` trait, `ProverTranscript<H>`, `VerifierTranscript<'a, H>`, `Channel`, `Preview`, grinding, `TranscriptError`, `SiteId` | `jolt-transcript` | Depends on `jolt-field` and `spongefish` only. |
| Sponge permutations / hash duplexes | `spongefish` (Blake2b512, Keccak, ...) and `jolt-transcript::PoseidonSponge` | One workspace pin for both repos. |
| Exact uniform sampling contract | `jolt-field` (`Field::random`) | The channel feeds it a sponge-backed `RngCore`. |
| Small-set challenge decoding | `jolt-field` (`from_challenge_bytes`) | Unchanged semantics. |
| Message codec: `CanonicalBytes` (encode) and `CanonicalDecode` (checked decode) | `jolt-field` | Implemented for every field, extension field, `u8`–`u128`, and `[u8; N]`. |
| Codec impls for group elements and commitments | `jolt-crypto`, `jolt-dory`, Akita crates | Each type's owner implements the codec locally. |
| Round-message shapes, protocol order | `jolt-sumcheck`, `jolt-verifier`, Akita protocol crates | Shapes are positional, derived from public parameters. |
| Akita site coordinates, descriptor bytes | `akita-types` | Plain data; converted to `SiteId` for diagnostics. |

### Invariants

1. **Absorbed bytes are transmitted bytes.** Every prover message is absorbed
   with exactly the bytes written to (prover) or consumed from (verifier) the
   NARG. `CanonicalDecode::from_bytes_le_checked` accepts only canonical
   encodings of valid values (field elements below the modulus, on-curve and
   in-subgroup points), so a verifier cannot accept two byte strings for one
   value.
2. **No unabsorbed proof data.** The verifier obtains proof data only through
   `Verifier::receive*`. There is no hint channel and no side struct. The
   absence of a hint API is the enforcement mechanism.
3. **Exact consumption.** `Verifier::finish(self)` fails unless the NARG is
   fully consumed. Composite verifiers (Jolt calling Akita) call it once, at the
   outermost boundary.
4. **Domain separation.** Each channel starts from a 64-byte protocol id that
   binds the protocol name and `Sponge::ID`, followed by a length-framed
   session value. Protocol axes are public inputs: the Jolt verifier absorbs
   its own build's `JoltProtocolConfig` first in the public preamble. A proof
   produced under one sponge or protocol configuration cannot verify under
   another.
5. **Role symmetry.** For any operation sequence, a verifier replaying it over
   the prover's NARG reproduces every challenge and every received value.
   `jolt-eval`'s `transcript_prover_verifier_consistency` invariant checks this
   for every sponge.
6. **Previews cannot mutate the proof.** `Preview<H>` holds only a clone of the
   public sponge state. It cannot write the NARG or advance the live sponge.
7. **Sites never affect bytes.** `SiteId` is diagnostic. It is recorded under
   the `logging` feature and compiled out otherwise. Toggling the feature does
   not change a proof or a challenge.
8. **Challenge distributions are explicit per call site.** `challenge::<F>()`
   is exactly uniform over `F` (`Field::random` rejection contract).
   `challenge_small::<F>()` is the field's small-set draw
   (`from_challenge_bytes` over 16 squeezed bytes; 125 bits for BN254). No other
   draw exists.

### Non-Goals

- Changing `dory-pcs` (crates.io). Dory keeps its `DoryTranscript` trait; see
  Design.
- Unifying `jolt-sumcheck` and `akita-sumcheck`.
- Byte-compatibility with any existing proof, fixture digest, or EVM/recursion
  transcript. Every pinned proof digest is regenerated.
- Updating out-of-tree consumers that reproduce the Blake2b-256 digest chain
  (EVM verifier #1890, Blake fast recursion #1892/#1911, wrapper constraints
  #1879/#1886). They port to the new transcript in their own PRs.
- Changing what any protocol stage proves. Message order is preserved stage by
  stage. Only transport, framing, and the challenge mapping in invariant 8
  change.

## Design

### API

```rust
pub trait Sponge: DuplexSpongeInterface<U = u8> + Default + Clone + Send + Sync + 'static {
    /// Bound into every protocol id derived for this sponge.
    const ID: &'static str;
}

pub struct ProtocolId([u8; PROTOCOL_ID_LEN]);
impl ProtocolId { pub const fn new<H: Sponge>(name: &str) -> Self; }

/// Operations both roles perform identically; shared protocol code takes `&mut impl Channel`.
pub trait Channel {
    type Sponge: Sponge;
    fn site(&mut self, site: SiteId);
    fn public<A: CanonicalBytes>(&mut self, value: &A);
    fn public_all<A: CanonicalBytes>(&mut self, values: &[A]);
    fn public_bytes(&mut self, bytes: &[u8]);               // u64-length-framed
    /// Prover: send `*value`. Verifier: overwrite `*value` with the received message.
    fn exchange<A: CanonicalDecode>(&mut self, value: &mut A) -> Result<(), TranscriptError>;
    fn exchange_all<A: CanonicalDecode>(&mut self, values: &mut [A]) -> Result<(), TranscriptError>;
    fn challenge<F: Field>(&mut self) -> F;                 // exactly uniform
    fn challenge_small<F: CanonicalEncoding>(&mut self) -> F; // from_challenge_bytes(16 bytes)
    fn challenge_bytes<const N: usize>(&mut self) -> [u8; N];
    fn preview(&self) -> Preview<Self::Sponge>;
    fn challenges_small<F: Field>(&mut self, len: usize) -> Vec<F>;  // provided
    fn challenge_powers<F: Field>(&mut self, len: usize) -> Vec<F>;  // provided: 1, γ, γ², ...
}

impl<H: Sponge> ProverTranscript<H> {
    pub fn new(protocol: &ProtocolId, session: &[u8]) -> Self;
    pub fn send<A: CanonicalBytes>(&mut self, value: &A);
    pub fn send_all<A: CanonicalBytes>(&mut self, values: &[A]);
    pub fn send_bytes(&mut self, bytes: &[u8]);             // length fixed by the protocol
    pub fn send_bounded_bytes(&mut self, bytes: &[u8], max_len: usize) -> Result<(), TranscriptError>;
    pub fn send_nonce(&mut self, nonce: u32);
    pub fn grind(&mut self, bits: u8) -> Result<u32, TranscriptError>;    // returns the nonce
    pub fn narg(&self) -> &[u8];
    pub fn finish(self) -> Vec<u8>;
}

impl<'a, H: Sponge> VerifierTranscript<'a, H> {
    pub fn new(protocol: &ProtocolId, session: &[u8], narg: &'a [u8]) -> Self;
    pub fn receive<A: CanonicalDecode>(&mut self) -> Result<A, TranscriptError>;
    pub fn receive_n<A: CanonicalDecode>(&mut self, count: usize) -> Result<Vec<A>, TranscriptError>;
    pub fn receive_bytes(&mut self, len: usize) -> Result<&'a [u8], TranscriptError>;
    pub fn receive_bounded_bytes(&mut self, max_len: usize) -> Result<&'a [u8], TranscriptError>;
    pub fn receive_nonce(&mut self, nonce_bits: u8) -> Result<u32, TranscriptError>;
    pub fn check_grind(&mut self, bits: u8) -> Result<u32, TranscriptError>;
    pub fn remaining(&self) -> usize;
    pub fn finish(self) -> Result<(), TranscriptError>;     // EOF and poison check
}
```

The verifier poisons itself on its first error: every later receive returns
`TranscriptError::Poisoned`, so a caller that drops an error cannot continue
on a desynchronized transcript.

Example: one sumcheck round of degree bound `d`. The verifier knows `d` from
public parameters, so the polynomial's length is never transmitted; the prover
pads lower-degree rounds with zero coefficients.

```rust
// prover                                     // verifier
prover.send_all(&padded_coeffs);              let c = verifier.receive_n::<F>(d + 1)?;
let r = prover.challenge_small::<F>();        let r = verifier.challenge_small::<F>();
```

### Wire format

The NARG is the concatenation of message encodings. Variable-length payloads
carry a 4-byte LE length checked against the caller's `max_len` before any
read. Public byte strings are absorbed with an 8-byte LE length frame. The NARG
carries no labels or tags. `JoltProof` is `{ protocol: JoltProtocolConfig,
narg: Vec<u8> }`. `protocol` exists only so a configuration mismatch surfaces as
a typed error before verification; it is never trusted, since the verifier
absorbs its own configuration.

The first prover message is the proof header: `trace_length` and `ram_K` as
u64, the six rw/one-hot round counts as `[u8; 6]`, the trace order as u64, and
the untrusted-advice flag as u8. The verifier validates the header against its
preprocessing before using it to size any later receive. The commitments follow
in `ProofCommitments` order at counts fixed by the RA layout.

Byte order is little-endian canonical (`CanonicalBytes`). The legacy
big-endian reversal disappears.

### Composition: Jolt calls Akita

Akita's prove and verify entry points take `&mut ProverTranscript<H>` / `&mut VerifierTranscript<H>`
(or `&mut impl Channel` for shared sites) and absorb their instance descriptor
with `public_bytes` as their first operation. Akita's standalone API constructs
its own channel with an Akita `ProtocolId` and calls the same inner function, so
there is one code path. Inside Jolt, stage 8 passes its live channel. Akita's
messages land in Jolt's NARG, and Jolt's transcript state binds everything
Akita squeezes. `bridged_akita_session`, the nested session label, and
`AkitaBatchProof`'s opaque proof bytes are deleted.

Akita's current instance binding (`DomainSeparator.instance(descriptor)`)
becomes a public message. Soundness is unchanged: the descriptor is absorbed
before any Akita challenge in both cases.

### Dory

`dory-pcs` exposes a symmetric `DoryTranscript` and verifies from a proof struct
it receives up front. `jolt-dory` runs it inside the NARG with no double
transmission:

- The prover adapter sends every value dory-pcs absorbs through `append_serde`,
  so the NARG holds Dory's messages in Dory's Fiat-Shamir order.
- The verifier rebuilds the proof struct by reading those messages from a
  scratch clone of the transcript (`read_proof`), then runs `dory::verify` with
  an adapter whose every `append_serde` receives the next message from the live
  transcript and checks it equals the value dory-pcs absorbs.
- The Σ₁ responses of a ZK proof, which dory-pcs never absorbs, are sent after
  the Dory messages.

Every NARG byte is therefore absorbed exactly once on each side, and invariant
2 holds without trusting `dory-pcs` to absorb what it reads.

### Grinding and previews

> Superseded: previews are gone. Grinding squeezes a seed and searches nonces
> on seeded forks; see `specs/jolt-transcript-spongefish-state.md`.

`Preview<H>` clones the sponge and exposes `absorb`, `absorb_bytes`,
`absorb_nonce`, and `squeeze::<N>()`. `grind(bits)` searches LEB128 nonces
against a preview, sends the winning nonce, then squeezes and checks the
predicate on the live channel. `check_grind` receives the nonce and checks the
same predicate. The
predicate (leading zero bits of 8 squeezed bytes, little-endian bit order) and
the nonce bound move from `akita-transcript/src/grinding.rs`. Akita's
fold-response nonce search builds on `Preview` inside Akita.

### Diagnostics

Under `logging`, each channel records `(SiteId, op kind, NARG byte range)`. Akita
converts `ProtocolSiteId` to `SiteId([u8; 32])`. Jolt's verifier tags each stage
with a `SiteId` and retires `fs_audit`'s thread-local scope. Tests use the
recorded ranges for two purposes:

- **Equivalence:** the prover's and verifier's site streams are identical.
  This replaces Akita's `transcript_hardening`. Jolt keeps its static source
  census (`fs_obligations`), retargeted to the message, absorb, challenge, and
  site calls, as a review gate.
- **Tamper sweep:** flipping one byte in every recorded prover-message range
  makes verification fail. This is the NARG form of the existing per-field
  tamper sweeps.

### BlindFold

The ZK prover finishes stage 8 on its `ProverTranscript`, then runs the
verifier's `verify_stages` over a `VerifierTranscript` built on its NARG prefix.
That is the same function `verify` runs: the seeding messages, stages 1–8, and
the lowering into the BlindFold protocol. The replay must consume the whole
prefix and reach the prover's sponge state (a 32-byte squeeze from a preview).
Otherwise proving fails with `ProverError::InvariantViolation`. BlindFold then
proves onto the original transcript. ZK recorders send round commitments
instead of round polynomials; the choice is made by recorder type, not by a
second transcript.

### Stage-level wire decisions

- **Output claims are received by shape.** Each stage receives its claims in
  the shape of its derived output points, on its typed claim routes
  (`specs/jolt-verifier-typed-messages.md`). Aliased openings are not sent;
  they are filled from their canonical source. Stage 6b's booleanity
  bytecode-RA alias is decided by opening-point equality over the derived
  points, so its ZK commitment count follows the verified rounds.
- **Stage 4 staged openings.** The RAM value-check input claim consumes the
  advice and program-image openings, so a clear proof sends them after the
  stage's gamma draws and before the batch. A ZK proof commits them in its
  output-claim rows in the claims struct's declaration order.
- **Uni-skip** output claims are sent right after the round's challenge, before
  any later draw.
- **Stage 8** runs `HomomorphicBatch`'s sequence on both sides: absorb the
  scaled claim values, draw the batching powers, open, and absorb the joint
  evaluation claim.

### Sponge selection

Protocol code is generic over `H: Sponge`. That replaces today's
`T: Transcript` one for one, so generic arity is unchanged. Jolt fixes its
default in one place (`pub type JoltSponge = jolt_transcript::Blake2b512` in
`jolt-verifier`). The SDK's `transcript-*` features choose `ProtocolSponge`
(Blake2b512, Keccak, or Poseidon). Akita drops the
`transcript-blake2b`/`transcript-keccak` features and the `compile_error!` that
enforces exactly one of them. The sponge reaches Akita through its caller's
`H`, and Akita's own default is `AkitaSponge = jolt_transcript::Blake2b512`.
`PoseidonSponge` stays behind the `transcript-poseidon` feature because it
pulls in BN254.

### Alternatives considered

- **Keep spongefish's `ProverState`/`VerifierState` and its codec traits.**
  Rejected for four reasons:
  - Typed errors: spongefish's `VerificationError` is a unit type, which
    conflicts with the verifier-closure lints.
  - Previews need the sponge state, which spongefish exposes only through its
    `yolocrypto` feature.
  - Its encode-then-serialize split permits absorbed bytes and transmitted
    bytes to differ, which invariant 1 forbids.
  - There is no logging hook.

  `jolt-transcript` uses spongefish for sponges only.
- **Keep labels and absorb them per message.** Rejected. Order is fixed by
  public parameters, so labels add no binding, and they cost one absorption per
  message. The diagnostic need is met by `SiteId` at zero release cost.
- **Ship the Dory proof as a hint, absorbed only through `DoryTranscript`.**
  Rejected. It reintroduces unabsorbed proof data, so soundness would depend on
  `dory-pcs` absorbing every element it reads.
- **One exactly uniform challenge everywhere.** Rejected for this change. The
  125-bit draw is a measured prover optimization (cheaper binding
  multiplications in BN254 Montgomery form). Changing it is a performance
  decision, not a transport one. The mapping is: today's `challenge()` becomes
  `challenge_small`, and today's `challenge_scalar()` and Akita's field
  challenges become exact `challenge`. Each draw moves to an equal or larger
  sample set.

## Evaluation

### Acceptance criteria

- [ ] `jolt-transcript` exposes only the API above. `Transcript`,
      `AppendToTranscript`, `Label`, `LabelWithCount`, `U64Word`,
      `DigestTranscript`, `LegacyBlake2bTranscript`, `SpongeTranscript`, and
      the split `ProverTranscript`/`VerifierTranscript` traits are deleted.
- [ ] `akita-transcript` is deleted. No Akita crate depends on `spongefish`
      directly. Akita depends on `jolt-transcript` at the Jolt PR's revision.
- [ ] `jolt-akita` contains no nested transcript. Akita's messages are in
      Jolt's NARG, and the only `finish()` call is Jolt's.
- [ ] `JoltProof` is `{ protocol, narg }`. No verifier stage reads proof data
      except through `VerifierTranscript`.
- [ ] Changing `JoltSponge` to `Keccak` compiles, and the prover acceptance
      suites pass (exercised once in CI on one suite).
- [ ] Tamper sweep: in the clear, ZK, and Akita fixture proofs, every recorded
      prover-message range rejects under a one-byte flip.
- [ ] Every pinned proof digest and fixture is regenerated. The fixture test
      decodes and verifies them.

### Testing strategy

- `jolt-transcript`:
  - Frozen known-answer vectors per sponge for protocol-id derivation, each
    atom kind, framed bytes, `challenge`, `challenge_small`, and grinding.
  - Round-trip symmetry through the `jolt-eval` invariant (updated to the new
    API).
  - Rejection of truncated, trailing, non-canonical, and over-length input,
    leaving the cursor unmoved on error.
  - Preview non-mutation.
- `jolt-field`: checked decoding for extension fields rejects any
  non-canonical coefficient.
- Workspace: the prover acceptance suites in CLAUDE.md (clear, ZK, Akita), the
  verifier fixtures, and `cargo clippy` in both modes.
- Akita: the existing PCS suites, with `transcript_hardening` rewritten
  against `logging` site streams.

### Performance

- No regression above noise on `transcript_ops`.
- No regression in end-to-end prover time on `fibonacci` at 2^16 and 2^20
  (`telemetry:fibonacci:prover_time_s`), Dory and Akita.
- Proof size is reported before and after. It should shrink: labels and
  32-byte framing words disappear, and the double-absorbed Dory bytes are not
  transmitted twice.

## Execution

### Implementation slices (both PRs developed together)

1. Pin spongefish `ef97413` in Jolt. Add the new `jolt-transcript`, and add
   `CanonicalEncoding` for extension fields.
2. Port the crates bottom-up: `jolt-crypto`, `jolt-poly`, `jolt-sumcheck`,
   `jolt-openings`, `jolt-dory`, `jolt-blindfold`.
3. Port `jolt-verifier` (stages read from `Verifier`, `JoltProof` becomes the
   NARG) and `jolt-prover` (stages write to `Prover`, BlindFold replays the
   NARG).
4. Akita PR: rebase on Akita `main`, delete `akita-transcript`, and make the
   entry points channel-taking. Port `akita-sumcheck`, `-types`, `-prover`,
   `-verifier`, `-pcs`, `-challenges`, `-config`, and `-cpu-backend`.
5. Bump the Akita pin in the Jolt PR to the Akita PR head. Port `jolt-akita`
   (Jolt already pins a post-#37 Akita, `1cc7a2a`, 12 commits behind `main`, and compiles against Akita `main` unchanged) and delete the bridge.
   Extend `[patch."https://github.com/a16z/jolt"]` to `jolt-transcript` and
   `jolt-poly`.
6. Regenerate fixtures and digests. Update `jolt-sdk`, `jolt-eval`, and
   `examples/recursion`.

### Merge order

1. The Akita PR merges first, pinning the Jolt PR's head revision (retained
   through its PR ref).
2. The Jolt PR repins Akita to the merge commit and merges.

After the Akita PR merges, `jolt-transcript`, `jolt-field`, and `jolt-poly` must
not change in the Jolt PR without a repin. Akita's next routine Jolt bump moves
it onto Jolt `main`.

## Unverified / risks

- **Akita API churn.** 17 open Akita
  PRs will conflict with the channel-taking signatures.
- **Recursion cycle count.** `examples/recursion` verifies in-guest with
  software Blake2b512 duplexing. The cycle-count change is not yet measured.
- **Dory parsing.** Every Dory message is decoded with checked canonical
  decoding and compared against what dory-pcs absorbs; the byte-flip sweep in
  `jolt-dory`'s tests covers one byte per message, not every byte.
- **API gaps found by the Akita port.** `Preview` cannot replay a
  `public_bytes` call, so Akita absorbs its fold payload unframed. A semantic
  failure after a successful receive does not poison the verifier; callers rely
  on returning `Err`.
- **Small-challenge wrapper constraints.** Whether any wrapper constraint
  depends on 125-bit challenges is out of scope here. It matters for the
  out-of-tree ports.

## References

- `specs/jolt-transcript-spongefish.md` (#1455): the prior staged port this supersedes.
- Akita #37 (spongefish proof stream), `akita-transcript` on Akita `main`.
- spongefish `ef97413` (`DuplexSpongeInterface`, `instantiations`).
