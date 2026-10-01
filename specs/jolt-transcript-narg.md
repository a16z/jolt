# Spec: One NARG transcript for Jolt and Akita

| Field     | Value              |
| --------- | ------------------ |
| Author(s) | @markosg04         |
| Created   | 2026-10-01         |
| Status    | proposed           |
| PR        | (Jolt) / (Akita)   |

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
ends, typed atoms, exact and small-set challenges, a preview handle for proof
of work, and feature-gated site diagnostics. It is expressive enough that Akita
uses it unchanged, and Jolt's prover, verifier, BlindFold, Dory, and Akita
integration all run on it.

### Ownership

| Component | Owner crate | Notes |
| --- | --- | --- |
| `Sponge` trait, `Prover<H>`, `Verifier<'a, H>`, `Channel`, `Atom`, `Preview`, grinding, `TranscriptError`, `SiteId` | `jolt-transcript` | Depends on `jolt-field` and `spongefish` only. |
| Sponge permutations / hash duplexes | `spongefish` (Blake2b512, Keccak, ...) and `jolt-transcript::PoseidonSponge` | One workspace pin for both repos. |
| Exact uniform sampling contract | `jolt-field` (`Field::random`) | The channel feeds it a sponge-backed `RngCore`. |
| Small-set challenge decoding | `jolt-field` (`from_challenge_bytes`) | Unchanged semantics. |
| Canonical encode/checked decode of fields | `jolt-field` (`CanonicalEncoding`) | Extension fields gain `CanonicalEncoding` (checked per coefficient). |
| Atom impls for group elements, commitments | `jolt-crypto`, `jolt-dory`, Akita crates | Each type's owner implements `Atom` locally. |
| Round-message shapes, protocol order | `jolt-sumcheck`, `jolt-verifier`, Akita protocol crates | Shapes are positional, derived from public parameters. |
| Akita site coordinates, descriptor bytes | `akita-types` | Plain data; converted to `SiteId` for diagnostics. |

### Invariants

1. **Absorbed bytes are transmitted bytes.** Every prover message is absorbed
   with exactly the bytes written to (prover) or consumed from (verifier) the
   NARG. `Atom::read` accepts only canonical encodings of valid values (field
   elements below the modulus, on-curve and in-subgroup points), so a verifier
   cannot accept two byte strings for one value.
2. **No unabsorbed proof data.** The verifier obtains proof data only through
   `Verifier::receive*`. There is no hint channel and no side struct. The
   absence of a hint API is the enforcement mechanism.
3. **Exact consumption.** `Verifier::finish(self)` fails unless the NARG is
   fully consumed. Composite verifiers (Jolt calling Akita) call it once, at the
   outermost boundary.
4. **Domain separation.** Each channel starts from a 64-byte protocol id that
   binds the protocol name, version, proof mode (clear or ZK), and
   `Sponge::ID`, followed by a length-framed session value. A proof produced
   under one sponge or mode cannot verify under another.
5. **Role symmetry.** For any operation sequence, a verifier replaying it over
   the prover's NARG reproduces every challenge and every received value.
   `jolt-eval`'s `transcript_prover_verifier_consistency` invariant checks this
   over `Prover`/`Verifier` for every sponge.
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

/// Fixed-width canonical value. Shapes are positional: callers send and receive
/// exactly the counts their public parameters imply.
pub trait Atom: Sized {
    const SIZE: usize;
    fn write(&self, out: &mut [u8]);                       // exactly SIZE bytes
    fn read(bytes: &[u8]) -> Result<Self, TranscriptError>; // canonical-only
}
impl<F: CanonicalEncoding> Atom for F { /* LE canonical bytes */ }
// Local atoms: U32, U64, Bytes<N>. Group elements and commitments implement
// Atom in their owner crates.

pub struct ProtocolId([u8; 64]);
impl ProtocolId {
    pub fn new<H: Sponge>(name: &str, version: u32, mode: ProofMode) -> Self;
}

pub struct Prover<H: Sponge> { /* sponge, narg, logging */ }
impl<H: Sponge> Prover<H> {
    pub fn new(protocol: &ProtocolId, session: &[u8]) -> Self;
    pub fn send<A: Atom>(&mut self, value: &A);
    pub fn send_all<A: Atom>(&mut self, values: &[A]);
    pub fn send_bytes(&mut self, bytes: &[u8], max_len: usize) -> Result<(), TranscriptError>;
    pub fn grind(&mut self, bits: u8) -> Result<(), TranscriptError>;
    pub fn finish(self) -> Vec<u8>;
}

pub struct Verifier<'a, H: Sponge> { /* sponge, remaining narg, logging */ }
impl<'a, H: Sponge> Verifier<'a, H> {
    pub fn new(protocol: &ProtocolId, session: &[u8], narg: &'a [u8]) -> Self;
    pub fn receive<A: Atom>(&mut self) -> Result<A, TranscriptError>;
    pub fn receive_n<A: Atom>(&mut self, count: usize) -> Result<Vec<A>, TranscriptError>;
    pub fn receive_bytes(&mut self, max_len: usize) -> Result<Vec<u8>, TranscriptError>;
    pub fn check_grind(&mut self, bits: u8) -> Result<(), TranscriptError>;
    pub fn finish(self) -> Result<(), TranscriptError>;
}

/// Operations both roles perform identically; shared protocol code takes `&mut impl Channel`.
pub trait Channel {
    type Sponge: Sponge;
    fn public<A: Atom>(&mut self, value: &A);
    fn public_all<A: Atom>(&mut self, values: &[A]);
    fn public_bytes(&mut self, bytes: &[u8]);               // length-framed
    /// Prover: send `*value`. Verifier: overwrite `*value` with the received atom.
    fn exchange<A: Atom>(&mut self, value: &mut A) -> Result<(), TranscriptError>;
    fn challenge<F: Field>(&mut self) -> F;
    fn challenge_small<F: Field>(&mut self) -> F;
    fn challenge_bytes<const N: usize>(&mut self) -> [u8; N];
    fn preview(&self) -> Preview<Self::Sponge>;
    fn site(&mut self, site: SiteId);
}
```

`Prover` and `Verifier` implement `Channel`. `public_*` and `challenge*` are
inherent on both as well, so role-specific code needs no trait import.

Example: one sumcheck round of degree `d`. The verifier knows `d` from public
parameters, so the poly's length is never transmitted.

```rust
// prover                                     // verifier
prover.send_all(round_poly.coeffs());         let c = verifier.receive_n::<F>(d + 1)?;
let r = prover.challenge_small::<F>();        let r = verifier.challenge_small::<F>();
```

### Wire format

The NARG is the concatenation of atom encodings and length-framed byte payloads
(8-byte LE length, then bytes, with `len <= max_len` checked before any
allocation). It carries no labels or tags. `JoltProof` becomes a
`Vec<u8>` newtype with serde. Prover-chosen parameters the verifier needs before
reading shapes (`trace_length`, `ram_K`, configs, advice presence) are the first
prover messages. The verifier receives and validates each one before using it
to size a later receive. The proof mode is in the protocol id, and the verifier
also checks it against its build before reading, so a mismatch surfaces as a
typed error rather than a challenge mismatch.

Byte order is little-endian canonical (`CanonicalBytes`). The legacy
big-endian reversal disappears.

### Composition: Jolt calls Akita

Akita's prove and verify entry points take `&mut Prover<H>` / `&mut Verifier<H>`
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

`dory-pcs` exposes a symmetric `DoryTranscript` and its own proof struct.
`jolt-dory` implements `DoryTranscript` for `&mut impl Channel`: `append_*` maps
to `public_*` and `challenge_scalar` to `challenge`. The serialized Dory proof is
sent with `send_bytes` and decoded with `DorySerialize`, which rejects trailing
and non-canonical data. Dory therefore absorbs its group elements twice, once
as NARG bytes and once through its own transcript calls. That costs a few
hundred group-element absorptions per proof, and it keeps invariant 2: there is
no hint path. A Dory-native NARG integration belongs to a `dory-pcs` release
and is out of scope.

### Grinding and previews

`Preview<H>` clones the sponge and exposes `absorb<A: Atom>` and
`squeeze(&mut [u8])`. `grind(bits)` searches `U32` nonces against a preview,
sends the winning nonce, then squeezes and checks the predicate on the live
channel. `check_grind` receives the nonce and checks the same predicate. The
predicate (leading zero bits of 8 squeezed bytes, little-endian bit order) and
the nonce bound move from `akita-transcript/src/grinding.rs`. Akita's
fold-response nonce search builds on `Preview` inside Akita.

### Diagnostics

Under `logging`, each channel records `(SiteId, op kind, NARG byte range)`. Akita
converts `ProtocolSiteId` to `SiteId([u8; 32])`. Jolt's verifier tags each stage
with a `SiteId` and retires `fs_audit`'s thread-local scope. Tests use the
recorded ranges for two purposes:

- **Equivalence:** the prover's and verifier's site streams are identical.
  This replaces Akita's `transcript_hardening` and Jolt's `fs_inventory`.
- **Tamper sweep:** flipping one byte in every recorded prover-message range
  makes verification fail. This is the NARG form of the existing per-field
  tamper sweeps.

### BlindFold

The ZK prover finishes stage 8 on its `Prover`, then runs the verifier's own
stage functions over a `Verifier` built on the NARG prefix to obtain the stage
outputs that BlindFold lowers. The replay has to end with the prefix fully
consumed and a 32-byte sponge fingerprint (squeezed from a clone) equal to the
prover's. Otherwise proving fails with the existing
`ProverError::InvariantViolation`. BlindFold then continues on the original `Prover`. ZK recorders send
round commitments instead of round polynomials. The choice is made by message
type, not by a second transcript.

### Sponge selection

Protocol code is generic over `H: Sponge`. That replaces today's
`T: Transcript` one for one, so generic arity is unchanged. Jolt fixes its
default in one place
(`pub type JoltSponge = spongefish::instantiations::Blake2b512` in
`jolt-verifier`). Akita drops the `transcript-blake2b`/`transcript-keccak`
features and the `compile_error!` that enforces exactly one of them. The sponge
reaches Akita through its caller's `H`. `PoseidonSponge` stays behind the
`transcript-poseidon` feature because it pulls in BN254.

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
- [ ] `JoltProof` is a NARG newtype. No verifier stage reads proof data except
      through `Verifier`.
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
- **Dory parsing.** That `DorySerialize` rejects every non-canonical proof
  encoding is assumed from `jolt-dory`'s existing serde tests, not re-audited.
- **Small-challenge wrapper constraints.** Whether any wrapper constraint
  depends on 125-bit challenges is out of scope here. It matters for the
  out-of-tree ports.

## References

- `specs/jolt-transcript-spongefish.md` (#1455): the prior staged port this supersedes.
- Akita #37 (spongefish proof stream), `akita-transcript` on Akita `main`.
- spongefish `ef97413` (`DuplexSpongeInterface`, `instantiations`).
