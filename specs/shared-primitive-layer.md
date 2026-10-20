# Spec: One shared primitive layer for Jolt and Akita

| Field     | Value                                   |
| --------- | --------------------------------------- |
| Author(s) | @markosg04                              |
| Created   | 2026-10-06                              |
| Status    | proposed                                |
| PR        | (Jolt) / (Akita), per migration           |

## Summary

Jolt and Akita already share a field, polynomial, and transcript layer: after
the transcript-unification PRs (jolt #2000, akita #185), Akita depends on
`jolt-field`, `jolt-poly`, and `jolt-transcript` directly, and `akita-field` and
`akita-transcript` are gone. Most other primitives are still implemented once per
repository, or bridged:

| Concept | Copies today |
|---|---|
| Sumcheck driver | `jolt-sumcheck`, `akita-sumcheck/src/proof_stream.rs` |
| eq / bind / Gruen | `jolt-poly`, `akita-algebra/src/{eq_poly,split_eq,poly}.rs` |
| Keccak / SHAKE | spongefish (via `jolt-transcript`), `akita-challenges/src/sampler/xof.rs` |
| Signed small accumulator | `akita-sumcheck/accum.rs`, `akita-cpu-backend` range image |
| SIMD dispatch policy | compile-time (`jolt-field`), runtime `OnceLock` + env (`akita-algebra`) |
| Serialization | serde, `ark-serialize`, `akita-serialization`, spongefish codecs |
| PCS boundary | `jolt-akita` adapter (~3.7k LOC of bridge code) |

Shared crates pay off only if no consumer keeps its own copy and no consumer
needs an adapter. Whiteout makes this pressing. It masks *every* witness-dependent
field message of the Jolt PIOP and of Akita's folds 1..k with pads committed in
advance, and its Σ-protocol needs ring arithmetic, Ajtai commitments with hiding
randomness, sparse ring challenges, and Gaussian rejection sampling. With two
sumcheck drivers, the masking hook gets written twice, and the pad schedules can
drift apart, which is a soundness hazard.

This spec makes the jolt repository the single owner of the shared primitive
layer. It defines the ownership rules, the target crate layout, and the
interfaces that must serve every consumer. It also sets the minimal migration
path to Whiteout, delivered in review points that each need only one team.

## Intent

### Goal

Every proof-system or cryptographic primitive used by more than one repository
has exactly one implementation, in a jolt-repo crate, and each consumer uses it
directly, with no adapter.

### Ownership rules

These rules decide where code goes. They are the part reviewers should check most
carefully.

1. **Location.** The shared layer lives in the jolt repository. Akita pins it by
   git hash. Publishing to crates.io comes later and does not change
   the layout.
2. **Algebra vs. crypto.** `jolt-algebra` holds math that needs no hardness
   assumption. `jolt-crypto` holds anything that carries a hardness assumption or
   a security parameter: groups, Pedersen, Ajtai/SIS commitments, hiding
   randomness, samplers, ring challenge sets, and hashing.
3. **Protocol vs. primitive.** A PCS protocol (Akita's prover, verifier,
   schedules, planner, setup, and config; Dory's protocol) stays with its owner.
   So do generated parameter tables keyed by protocol types, such as Akita's SIS
   tables and the estimator that generates them. Moving those would create a
   dependency cycle.
4. **Kernels stay with their owner for now.** A consumer keeps its
   performance-tier kernels and implements the shared traits. Moving Akita's
   kernels to the `jolt-prover`/`jolt-kernels` architecture is deferred (M11).
5. **Two serialization layers only.** Proof and transcript bytes use spongefish
   codecs with shape-bounded decoding. Everything at rest (setup, preprocessing,
   keys, configs, fixtures) uses serde.

### Target layout

| Crate | Owns | Absorbs |
|---|---|---|
| `jolt-algebra` (renamed from `jolt-field`) | prime and extension fields, packed SIMD, delayed reduction, the cyclotomic ring R_q, NTT/CRT, smooth FFT, balanced digit decomposition, norms, checked and narrowing arithmetic, the one runtime SIMD dispatcher | `akita-algebra` (ring/, ntt/, fft.rs, decomposition, `cfg_try_fold_reduce!`); `akita-error::{checked,narrowing}`; `akita-sumcheck/accum.rs` |
| `jolt-poly` | multilinear, eq / split-eq / Gruen, univariate, compressed round forms, `Point<E, F>` | `akita-algebra/src/{eq_poly,split_eq,poly}.rs`, the generic parts of `offset_eq` (`eq_eval_at_index`, `OffsetEqWindow`, `EqPairTensor*`) |
| `jolt-transcript` | spongefish channel, codecs, sites, grinding, `Fork`; re-exports spongefish's `Encoding`/`NargDeserialize`; the private-message seam (Design §4) | — |
| `jolt-crypto` | groups, Pedersen, vector commitments; `lattice` (Ajtai matrix derivation, inner/outer commitment, outer hiding randomness); `challenge` (sparse ring challenges, operator-norm rejection); `sample` (uniform, discrete Gaussian, rejection rule); `hash` (SHAKE256) | `akita-challenges` (minus `fold_draw.rs`); `akita-types/proof/setup.rs` matrix derivation; `akita-cpu-backend` commitment stages (the generic Ajtai operation) |
| `jolt-sumcheck` | the one sumcheck driver, round formats, challenge policies, the recorder trait, batching | `akita-sumcheck/{proof_stream,traits,single}.rs` |
| `jolt-openings` | PCS traits, with the hiding commitment split from the ZK opening | — |
| `jolt-zk` (renamed from `jolt-blindfold`) | the masking front end (pad schedule, masking recorder, residual relation, back-end trait), the `blindfold` back end (curve), the `whiteout` back end (constant-term lift, lattice Σ-protocol) | `jolt-sumcheck`'s `committed` and `r1cs` features; the building blocks in Akita `dev`'s `akita-zk-prover` and `akita-zk-verifier` |

`akita-algebra`, `akita-challenges`, `akita-sumcheck`, and `akita-serialization`
are deleted. `akita-error` keeps only its protocol error type. `akita-config`
gains a parameterized external-family preset, which replaces
`jolt-akita/src/configs.rs`.

```mermaid
graph BT
  subgraph shared["jolt repo: shared layer"]
    algebra[jolt-algebra]
    poly[jolt-poly] --> algebra
    transcript[jolt-transcript] --> algebra
    crypto[jolt-crypto] --> poly
    sumcheck[jolt-sumcheck] --> transcript
    sumcheck --> poly
    openings[jolt-openings] --> crypto
    openings --> transcript
    zk[jolt-zk] --> sumcheck
    zk --> openings
  end
  subgraph akita["akita repo"]
    akpcs[akita-pcs / prover / verifier] --> zk
    akbe[akita-cpu-backend] --> sumcheck
    akbe --> crypto
  end
  subgraph joltapp["jolt repo: Jolt"]
    joltakita[jolt-akita: one-hot packing only] --> akpcs
    joltakita --> akbe
    prover[jolt-prover / jolt-verifier] --> zk
    prover --> joltakita
  end
```

The only edges that leave the shared layer go from consumers into it. The
remaining cycle between repositories (jolt-akita → akita → shared layer) involves
only Jolt-specific code and is accepted until M11.

### Invariants

- **I1. Dependency direction.** No shared-layer crate depends on an `akita-*`
  crate. `jolt-akita` is the only jolt crate that depends on Akita. A CI
  check over `cargo metadata` enforces both. M1b and M5b move the ring and
  challenge code into jolt before `jolt-zk::whiteout` needs it, so no exception
  is required.
- **I2. One owner per concept.** Each concept in the Summary table has exactly
  one implementation once its migration lands. A move (M1b, M5b, M6d) leaves a
  duplicate for exactly one review point, until Akita deletes its copy (M1c, M5c,
  M6e). Reviewers reject a new local copy.
  There's no reliable automated check for this, so it is enforced in review.
- **I3. Refactors preserve proofs.** Every M ticket except M6c is a refactor. On
  fixed-seed inputs, Jolt's verifier fixtures and Akita's proof bytes are
  byte-identical before and after. This is checked during the PR (old vs. new
  prover on the same inputs) and is not kept as a permanent test. M6c changes
  the Jolt ZK proof format on purpose and regenerates the ZK fixtures. W tickets
  change proofs by design.
- **I4. Bit order is a type.** Every eq, multilinear-evaluation, and bind API that
  takes a point takes `Point<E, F>`, where `E` is `HIGH_TO_LOW` or `LOW_TO_HIGH`.
  Converting between orders requires an explicit `match_endianness`. No reversal
  happens silently inside a bridge.
- **I5. Verifier input is parsed once.** Proof bytes are decoded by the transcript
  through spongefish codecs, with lengths bounded by the protocol's shape. serde
  never parses verifier input. No verifier parses the proof into a struct and then
  re-serializes it for comparison.
- **I6. The verifier closure keeps its lints.** Lattice code stays out of the
  default (Dory) build: `jolt-algebra`'s ring, NTT, and FFT modules and
  `jolt-crypto`'s `challenge` and `lattice` modules sit behind features that only
  Akita-mode crates enable, so the default verifier closure keeps its unsafe
  budget. `jolt-crypto`'s prover-only `sample` module has its own `sample`
  feature, so the Akita verifier's `lattice` build does not compile it. Beyond that: `jolt-algebra`, `jolt-crypto`,
  `jolt-sumcheck`, and `jolt-transcript` stay in the verifier closure
  (`specs/verifier-closure-lints.md`). Code moved in must meet that crate's lints,
  or carry a narrowly scoped `#[expect]` with a reason. SIMD modules keep
  `deny(unsafe_op_in_unsafe_fn)` and a SAFETY comment at every site.
- **I7. One pad schedule.** Every witness-dependent field message, whether from
  a sumcheck round, a claimed evaluation, or an Akita partial evaluation, goes
  through the private-message seam, in schedule order, in every repository.
  Masking is defined once, in `jolt-zk`.

### Non-Goals

- Whiteout's protocol design and parameters, which the paper owns. This spec
  plans how Whiteout is delivered (W1–W6) and which migrations come first.
- Moving Akita's PCS protocol, its SIS tables and estimator, its planner, or its
  CPU-backend kernels. M11 sketches the kernel move.
- Publishing to crates.io.

## Design

### 1. `jolt-algebra`

`jolt-field` is renamed and absorbs `akita-algebra`'s ring, NTT/CRT, FFT, and
decomposition modules unchanged. Moving the code does not rewrite it. The crate
takes over `akita-error`'s checked and narrowing arithmetic, together with a typed
`AlgebraError`, so moved code no longer returns `AkitaError`.

Kernels (NTT butterflies, signed dot products, Keccak in `jolt-crypto`) use one
runtime dispatcher: the ISA ceiling is detected once into a `OnceLock`, and a
single `JOLT_SIMD_BACKEND` environment variable can override it. Packed-field
widths stay compile-time, because they are types and cannot be chosen at runtime.

Cost of I6: the NTT and SIMD modules contain about 300 `unsafe` sites, and the
Akita verifier reaches SIMD code through `mat_vec_i16` and `ifma52_enabled`. That
is consistent with `jolt-field`'s current carve-out (`deny(unsafe_op_in_unsafe_fn)`,
no `forbid`). Unlike the SIMD code, `fft.rs` has no unsafe but has about 19
panic or assert sites, and the verifier reaches it through setup contribution. M1b
must convert those to typed errors.

### 2. Bit order

Jolt uses `HIGH_TO_LOW`; Akita uses `LOW_TO_HIGH`. Both orders are
supported. `Point<const E: Endianness, F>` already exists in `jolt-poly/src/point.rs`.
The `eq` and multilinear-evaluation APIs change from `&[F]` to `&Point<E, F>`
(I4), and they are written once, generic over `E`. Akita keeps its order and
deletes its own eq code. `jolt_to_akita_evals` and `reverse_point` in `jolt-akita`
become typed `match_endianness` calls at the single place Jolt hands a point to
Akita.

### 3. The sumcheck driver

`jolt-sumcheck` becomes the only driver. It has to support the behavior of both
current drivers. The table below is the requirement; the API shape is the
implementer's choice, as long as every row holds.

| Requirement | Jolt today | Akita today |
|---|---|---|
| Base field / extension split | single `F` | `F: CanonicalEncoding`, `E: ExtField<F>` |
| Round format (which coefficient is omitted) | standard (omit c1); uni-skip (all coefficients sent, check weighted by domain power sums) | standard; eq-factored (omit q0, `OmittedConstantPoly`, normalized running claim) |
| Challenge policy per round | `challenge_small` | full extension challenge, optional grind (`grinded_ext_challenge`) |
| Site tagging | none | `sumcheck_site(invocation, round, role)` |
| Fallible prover | `ProveRounds::prove_round` / `finish_rounds` return `Result<_, SumcheckError<F>>` | two traits: fallible `SumcheckKernel` (`AkitaError`), and infallible `SumcheckInstanceProver` / `EqFactoredSumcheckInstanceProver` adapted by wrappers |
| Prover self-check `s(0)+s(1) = claim` before send | yes (`prove_batch`, uni-skip) | yes |

Design decisions:

- The batched engine keeps its API and bytes. A single-instance driver
  (`prove_single` / `verify_single`) reproduces Akita's unbatched drivers byte
  for byte, and Akita moves onto it.
- `SumcheckDomain` stays the one owner of round-check weights. The eq-factored
  round becomes a third implementation next to `BooleanHypercube` and
  `CenteredIntegerDomain`, consumed by the verifier and the masking layer (§7).
  Eq-factored running claims are normalized (q(r), not eq(τ, r) · q(r)), as in
  Akita today.
- Challenges and sites come from a round context that owns the transcript,
  matching Akita's grinding context. Akita's grinding plan stays in Akita, and
  its context must reproduce `grinded_ext_challenge`'s bytes exactly (I3).
- Exact signatures are in `specs/whiteout-interfaces.md` §1.
- **The prover is fallible end to end.** There is one prover trait, and every
  method that does work returns `Result`: computing the round polynomial,
  binding a challenge, and finishing. Akita's infallible traits and their
  wrappers are deleted, and no infallible variant exists. The kernel error type
  appears only on the single-instance driver; the batched engine keeps
  `K = Infallible`. The shape (round
  count, degree bound, input claim) is a value fixed when the instance is
  built, so its getters cannot fail. The motivation is liveness: a prover
  serving requests must report a bad input or a kernel bug as an error, not
  abort the process.
- The driver is generic over the kernel's error type: `SumcheckError<E, K>` with
  a `Kernel(K)` variant. It is not `Box<dyn Error>`. A sumcheck error is always
  an invariant violation or a kernel failure: retrying with the same input fails
  again. Retryable aborts (a one-shot fold response that overflows its digit
  range, W2, and a Σ-protocol rejection, W4) happen outside any sumcheck, so they
  live in the prove-level errors of `jolt-zk` and Akita, and the caller retries
  the whole proof with fresh randomness.
- The driver checks the shape of every round message, including its degree
  against the bound. The prover self-check `s(0)+s(1) = claim` runs in release
  builds and returns a typed error, so a kernel bug surfaces at the point of
  fault rather than as a verifier rejection.
- Parallel loops inside kernels propagate errors with a fallible reduction
  (`cfg_try_fold_reduce!`, moved to `jolt-algebra` in M1b), not with panics.
- A fallible signature does not remove panics inside a kernel body. Kernel
  crates deny `unwrap`, `expect`, and `panic` in non-test code. Indexing in hot
  loops stays at warn level and relies on structural guarantees, because a
  checked `get` adds a branch per access and the performance gate (Evaluation)
  decides case by case. The top-level prove entry point converts any remaining
  panic into a typed error with `catch_unwind`. This works only under
  `panic = "unwind"`, and the release profile must keep that setting.

Akita's kernels in `akita-cpu-backend/src/opaque/sumcheck` implement the driver's
prover trait. `akita-sumcheck/types.rs` is a test oracle and is deleted, and the
tests that use it switch to the reference tier.

### 4. Private-message seam

Masking covers more than sumcheck rounds: Jolt's claimed evaluations and Akita's
partial-evaluation messages are witness-dependent too. `jolt-transcript` gains
one role-generic seam, `PrivateChannel`, for private field messages, which
depend on the witness:

- **Prover:** sends values, or value + pad in masked mode.
- **Verifier:** receives them and returns the pad slots the prover used, so both
  sides index pads identically.
- **Derived claims:** a claim the verifier can recompute is bound publicly in
  clear mode and not at all in masked mode, since earlier masked slots already
  bind it.

Public messages (commitments, nonces, setup-offloading claims) go straight to
the channel. `SumcheckRecorder` is built on the seam. The seam must sit in a
crate Akita already depends on, which is why it goes in `jolt-transcript` and not
`jolt-zk`. `specs/whiteout-interfaces.md` §2 lists which Jolt and Akita messages
go through it.

### 5. Commitment traits

`jolt-openings` splits the current `ZkOpeningScheme` into two capabilities. The
split lands in W6, when Akita first commits with randomness, so both halves have
a caller. M4 only deletes `jolt-akita`'s placeholder `ZkOpeningScheme` and
`ZkBatchOpeningScheme` impls, which return a transparent opening or an error.

- `HidingCommitment`: commit with a blind. The opening proof need not be zero
  knowledge.
- `ZkOpening`: the opening proof reveals nothing beyond the evaluation.

Dory implements both. Akita implements only `HidingCommitment`, because under
Whiteout the opening is covered by later folds and the Σ-protocol. `akita-pcs`
implements `CommitmentScheme`, `BatchOpeningScheme`, and `HidingCommitment`
natively. This deletes `jolt-akita/src/{scheme.rs (adapter parts), adapters.rs
(adapter parts), native_batching.rs, shape_guard.rs}`.

### 6. `jolt-crypto`

- `lattice`: derivation of Ajtai public matrices (Akita's SHAKE256 paged sampler,
  moved in M5b), then in W4 the Ajtai operation A·G⁻¹(x), its matrix-layout type,
  and outer hiding randomness. `jolt-crypto` owns the definition and layout;
  Akita's planner fills the layout, and Akita's kernels are the optimized tier,
  tested in Akita against the `jolt-crypto` operation. The
  randomness is drawn from a box with unit differences and fresh per slice, as
  in Whiteout §4.1. Akita's planner keeps choosing ranks and moduli, and its SIS
  tables stay in Akita (rule 3).
- `challenge`: `SparseChallenge`, the challenge-set configuration, and
  operator-norm rejection. Akita's `fold_draw.rs` (grinding preview over its fold
  schedule) stays in Akita.
- `sample`: uniform sampling mod q (one rejection sampler, replacing Akita's xof
  cursor and the transcript's exact challenge), a discrete Gaussian, and the joint rejection rule. The Gaussian
  and the rejection rule are new code, and both are **constant time** with
  respect to secret data. The Gaussian sampler's running time, branches, and
  memory addresses do not depend on the sampled value or the witness. For
  example, a CDT sampler scans its whole table with constant-time comparisons;
  it never uses an early-exit binary search. The rejection rule evaluates its
  acceptance probability and compares against uniform randomness without
  data-dependent branches. The only value that leaves the rejection step is the
  accept/reject bit, and rejection sampling makes that bit independent of the
  secret. Public-coordinate handling (y = 0 where s is public) branches only on
  public layout.
- `hash`: SHAKE256 for Akita's challenge sampler and public-matrix derivation,
  replacing the hand-rolled sponge in `akita-challenges/src/sampler/xof.rs`.
  `jolt-transcript` keeps spongefish's own Keccak sponge. Where spongefish
  accepts an external permutation, the transcript may use `jolt-crypto`'s
  instead, and both are supported. The single-state permutation inside
  spongefish is a third-party dependency, so I2 does not count it as a copy.
- `commitment`: `VectorCommitment` keeps its scalar blind. The lattice pad
  commitment is not a `VectorCommitment`: its committed vector is mixed-radix
  and its opening is proved by the Σ-protocol, never by a row opening. It is
  a `jolt-zk::whiteout` type that produces `ZkBackend`'s `PadCommitment` and
  `PadOpening`, so the masking front end is generic over the back ends
  through `ZkBackend`, not through `VectorCommitment`
  (`specs/whiteout-sigma.md` §1.6).

### 7. `jolt-zk`

`jolt-blindfold` is renamed to `jolt-zk` and split into three modules.

- `masking` (shared front end): the pad schedule (pads indexed in schedule order,
  committed before the first challenge), the masking `PrivateChannel` and
  `SumcheckRecorder` (send `m + p` for every coefficient the clear proof
  sends), and the residual relation: field-level affine rows and product rows
  over the pads and the residual witness, plus the back-end trait that
  `blindfold` and `whiteout` implement. The PCS supplies a cut point and its
  private messages up to that cut. Dory's cut is at the start of its opening,
  since its own recursion handles ZK. Akita's cut is fold k ∈ {2, 3, 4}.
- `blindfold` (curve back end): the existing Nova random-instance fold and the
  outer/inner sumcheck, now fed the residual relation over committed pads instead
  of per-round Pedersen commitments. It takes over `jolt-sumcheck`'s `committed`
  and `r1cs` features, so `jolt-sumcheck` drops its optional dependencies on
  `jolt-crypto` and `jolt-r1cs`.
- `whiteout` (lattice back end): the constant-term lift (moved from Akita
  `dev`'s `akita-zk-*` crates), the Σ-protocol with a Gaussian mask
  (`jolt-crypto::sample`) and joint rejection, and the response statement. It
  uses the ring from `jolt-algebra` and the sparse challenges from
  `jolt-crypto`. Proving that
  statement is PCS-specific, so `whiteout` defines a `ResponseProver` trait and
  Akita implements it. Akita's fold verifier likewise implements a
  `ResidualRows` trait defined in `masking`. Both traits are defined in
  `jolt-zk` and implemented in Akita, so the dependency runs from Akita into the
  shared layer (I1).

The two back ends follow the same recipe: fold the residual relation with a random
satisfying instance, then prove the folded instance with an argument that is not
zero knowledge. They share no fold code. The random instance (uniform field vs.
Gaussian ring element), the rejection step, and the ring lift differ, so no
generic fold trait is defined. The only planned code sharing is evaluating the
quadratic form and its polar form over residual rows.

### 8. Serialization and errors

- `akita-serialization` is deleted. Its implementing types fall into three groups:
  - **Transcript-facing:** `SetupPrefixSlotId`, the instance-descriptor and
    selection digests, and `CommittedGroup`/`Commitment` inside Jolt's proof.
    These get spongefish codecs with shape bounds.
  - **At rest:** setup, expanded setup, registries, `FlatMatrix`, seeds. These
    move to serde.
  - **Test oracles only:** `SumcheckProof`, `EqFactoredSumcheckProof`,
    `PhysicalL2NormProof`, `TerminalResponse`'s serialize path. These are deleted.
- `ark-serialize` leaves `jolt-algebra`, `jolt-crypto`, `jolt-riscv`,
  `jolt-program`, and `jolt-verifier`. `jolt-crypto` keeps a serde wrapper for
  arkworks curve points, which is the one place it remains.
- Each layer has its own typed error. `AkitaError` stays inside Akita's protocol
  crates.

### Alternatives considered

- **A separate primitives repository.** Rejected by the owner. Shared code stays in
  the jolt repo, pinned by hash, until it is published.
- **A separate `jolt-ring` crate.** Rejected in favor of one algebra crate. The
  case for a split was keeping SIMD-heavy NTT code out of a lint-strict crate,
  but `jolt-field` already carries an unsafe carve-out, so the split buys
  nothing (I6).
- **Naming the crate `jolt-whiteout`.** Rejected. Whiteout names the lattice
  transformation; the crate also contains the curve back end.
- **A generic fold trait across back ends.** Rejected. No code is shared beyond
  the residual rows; see §7.
- **Moving the SIS tables and estimator.** Rejected. The tables are keyed by Akita
  types, and `akita-sis-estimator` depends on `akita-types`, so moving them
  creates a cycle.
- **The lattice back end in Akita's `akita-zk-*` crates.** LZ had started the
  building blocks there, and keeping them in Akita would have taken M1b, M1c, M5b, and M5c
  off the path to Whiteout. Rejected by the owner: ZK and crypto primitives
  belong to the shared layer, and one `jolt-zk` serves both back ends. The cost
  is that those moves and M6d are on the path.
- **One canonical bit order.** Rejected. Supporting both through the type costs
  nothing at runtime and avoids changing Akita's transcript.

## Migrations and review points

### Why review points

Jolt and Akita are reviewed by different teams: jolt by its a16z maintainers,
Akita by the LayerZero (LZ) Akita maintainers. Jolt's `Cargo.toml` redirects
Akita's dependencies on jolt crates to the jolt workspace
(`[patch."https://github.com/a16z/jolt"]`), so the pinned Akita always compiles
against current jolt. A breaking change to a jolt crate that Akita uses therefore
breaks jolt's build until Akita updates and jolt re-pins. That forces a pair of
PRs, one per repo, that must merge together. M0 is such a pair, and pairs like
it are slow to land.

The plan avoids that with three rules:

- **Add first, remove later.** A jolt change to a crate Akita uses adds the new
  API and keeps the old one. A move into jolt copies the code in and leaves
  Akita's copy alone. Akita switches to the jolt version in its own review
  point, deleting its copy. Jolt removes the old API in a later one. No two PRs
  ever have to merge together, and the temporary duplicates live for one review
  point. Once Akita depends on `jolt-sumcheck`, `jolt-crypto`, and `jolt-zk`,
  they join the redirect list and fall under the same rule.
- **One repo per review point.** Every review point after RP0 touches only jolt
  or only Akita, so its gate needs one team. The next review point starts only
  after every PR in the current one has merged. Development does not wait on
  review points; see Development workflow.
- **Shallow stacks.** No PR depends on an unmerged PR in the other repo. Within
  one repo, PRs stack at most one level deep.

### Akita branches

Akita has two long-lived branches (`docs/branches.md` in Akita). `main` carries
the current protocol and needs LZ maintainer approval. `dev` is `main` plus
opt-in extensions; it needs passing CI and a review from another `dev`
contributor, and with no extension selected it must produce exactly `main`'s
bytes. Changes flow from `main` into `dev` by sync.

- Seams that keep Akita's behavior (M1c, M2b, M3b, M5c) go to `main`, as Akita's
  policy requires, and reach `dev` through a sync.
- Whiteout's Akita work (W1, W2, W3, W5, M6e) goes to `dev`. Akita's ZK roadmap
  forbids a selectable partial ZK mode, so none of it is selectable until the
  last PR of RP4, which turns the mode on.
- Jolt pins Akita `dev` for Whiteout. Bringing ZK into Akita `main` later is a
  separate step the LZ maintainers choose.

LZ's tracking issue for ZK (LayerZero-Labs/akita#120) already landed building
blocks on `dev` in `akita-zk-prover` and `akita-zk-verifier`: the constant-term
lift (`LiftedFunctional`) and the zero-constant-term mask sampler. Under this
spec they move into `jolt-zk` (M6d), and the `akita-zk-*` crates keep only
Akita-specific code.

### Tickets

Priority tags: **P0** is on the path to Whiteout; **P1** is worth doing before
Whiteout but not required; **P2** is independent cleanup; **P3** is deferred.
Where a migration spans both repos, its parts are lettered by step. W tickets
are Whiteout work.

| ID | Ticket | Repo | Priority |
|---|---|---|---|
| M0 | Transcript unification (jolt #2000, akita #185 on `main`), then sync `main` → `dev` | both | P0 |
| M1a | Rename `jolt-field` → `jolt-algebra`; keep `jolt-field` as a re-export crate | jolt | P0 |
| M1b | Copy Akita's ring, NTT/CRT, FFT, digit decomposition, checked arithmetic, and signed accumulators into `jolt-algebra`, unchanged except where jolt's lints require, behind a feature | jolt | P0 |
| M1c | Akita switches to `jolt-algebra` and deletes those modules from `akita-algebra` | Akita `main` | P0 |
| M1d | Delete the `jolt-field` re-export crate | jolt | P0 |
| M2a | `jolt-sumcheck`: fallible prover with a kernel error type, release self-check, shape checks; round formats, challenge policies, sites, base/extension split. Additive. | jolt | P0 |
| M2b | Port Akita's driver onto `jolt-sumcheck`; Akita's kernels implement the prover trait; delete `akita-sumcheck`'s driver and test-only proof types | Akita `main` | P0 |
| M2c | Delete Jolt's superseded sumcheck call paths | jolt | P0 |
| M3a | `jolt-transcript`: role-generic private-message seam (`PrivateChannel`); `SumcheckRecorder` built on it; Jolt's claimed evaluations routed through it | jolt | P0 |
| M3b | Route Akita's partial-evaluation messages through the seam | Akita `main` | P0 |
| M4 | Delete `jolt-akita`'s placeholder `ZkOpeningScheme` / `ZkBatchOpeningScheme` impls | jolt | P0 |
| M5a | `jolt-crypto::sample`: constant-time discrete Gaussian and rejection rule | jolt | P0 |
| M5b | Copy Akita's sparse challenges, operator-norm rejection, and public-matrix derivation into `jolt-crypto`, behind a feature. Akita's commitment operation is not separable from its plans and kernels, so the Ajtai operation lands with W4 (below). | jolt | P0 |
| M5c | Akita switches to `jolt-crypto`'s challenges and matrix derivation; delete `akita-challenges` and Akita's copy of the derivation; keep Akita's matrix-length check and its truncated-prefix test | Akita `main` | P0 |
| M6a | Rename `jolt-blindfold` → `jolt-zk` (pure rename) | jolt | P0 |
| M6b | `jolt-zk::masking`: pad schedule, masking recorders, residual-relation rows, the back-end trait | jolt | P0 |
| M6c | BlindFold onto committed pads (Dory); remove per-round Pedersen commitments | jolt | P0 |
| M6d | Move `akita-zk-*`'s building blocks (constant-term lift, zero-constant-term sampler) into `jolt-zk::whiteout` | jolt | P0 |
| M6e | Akita deletes the building blocks M6d moved | Akita `dev` | P0 |
| W1 | Akita: outer hiding randomness per slice, carried into the next fold's witness | Akita `dev` | P0 |
| W2 | Akita: one-shot fold responses in ZK mode; overflow is an abort, not a retry | Akita `dev` | P0 |
| W3 | Akita fold verifier implements `ResidualRows`; masked cut point at fold k | Akita `dev` | P0 |
| W4 | `jolt-zk::whiteout`: Σ-protocol with Gaussian mask and joint rejection, response statement, `ResponseProver` trait; `jolt-crypto::lattice`'s Ajtai operation and layout type; `VectorCommitment::Blind` and the lattice pad commitment | jolt | P0 |
| W5 | Akita implements `ResponseProver` (digit ranges, ring-quadratic reduction, setup offloading); the last PR turns ZK mode on and adds hidden-vs-transparent tests | Akita `dev` | P0 |
| W6 | Jolt + Akita ZK integration: pin Akita `dev`; split `HidingCommitment` / `ZkOpening`, and `jolt-akita` implements `HidingCommitment`; the `akita` and `zk` features become compatible; end-to-end prover and verifier | jolt | P0 |
| M7 | eq / multilinear evaluation into `jolt-poly` with typed bit order | both | P1 |
| M8 | `akita-pcs` implements `jolt-openings` natively; delete the `jolt-akita` adapter; `jolt-akita`'s configs become an `akita-config` preset | both | P1 |
| M9 | Delete `akita-serialization` and `ark-serialize` | both | P2 |
| M10 | `jolt-crypto::hash`; delete Akita's hand-rolled XOF | both | P2 |
| M11 | `akita-cpu-backend` → `jolt-prover` / `jolt-kernels` architecture | both | P3 |
| M12 | One runtime SIMD dispatcher in `jolt-algebra`, replacing the per-module detection and Akita's `AKITA_*` overrides (an operational change for Akita) | both | P2 |

### Review points

| RP | Repo · branch · reviewers | PRs | Gate |
|---|---|---|---|
| RP0 | both | M0; this spec | M0 merged and synced into Akita `dev`; this spec approved by both teams, including LZ's agreement in akita#120 that Whiteout's building blocks live in `jolt-crypto` and `jolt-zk` |
| RP1 | jolt · a16z | M1a, M1b, M5b (moves); M2a, M3a, M4, M5a (new API) | Merged. Jolt proofs byte-identical; the pinned Akita still builds |
| RP2 | Akita · `main` · LZ maintainers | M1c, M5c, M2b, M3b | Merged and synced into `dev`. Akita proofs byte-identical |
| RP3 | jolt · a16z | M1d, M2c, M6a, M6b, M6c, M6d, W4 | Merged. BlindFold runs on pads with regenerated ZK fixtures; masking and Σ-protocol tests pass |
| RP4 | Akita · `dev` · `dev` contributors | M6e, W2, W1, W3, W5 | Merged. ZK mode selectable only after the last PR; hidden-vs-transparent tests pass |
| RP5 | jolt · a16z | W6 | End-to-end ZK Jolt + Akita proof verifies; fixtures, tamper sweep and benchmark recorded |

RP1 and RP3 are large for one review round. Each splits into two batches that the
same reviewers take in turn: RP1 into the moves (M1a, M1b, M5b) and the new APIs
(M2a, M3a, M4, M5a); RP3 into the cleanup and BlindFold work (M1d, M2c, M6a,
M6b, M6c) and the lattice work (M6d, W4). The moves are pure copies, which
reviewers can check with a rename-aware diff, except where moved code must meet
`jolt-algebra`'s lints (I6), for example `fft.rs`'s panic sites.

Within a review point, PRs are independent except for one-level stacks in the
same repo: M3a builds on M2a, M6c and M6d build on M6b, and W4 builds on M6d.
Because `dev` needs no maintainer gate, RP4 lands PR by PR without building a
stack.

Akita pins jolt at the start of RP2 (post-RP1 jolt) and again at the start of
RP4 (post-RP3 jolt, for `jolt-zk`). The RP2 pin is the first to name
`jolt-algebra` and `jolt-crypto` by git, so the jolt-side pin-bump commit adds
both to jolt's `[patch]` redirect list; otherwise Cargo resolves two copies and
the ring and field types stop matching. Jolt pins Akita `dev` at the start of RP5.

The add-first rule also runs the other way. Jolt's `jolt-akita` names Akita
paths (`akita_algebra::ring`, `akita_challenges`), and Akita's recursion
profile workspace compiles it against local Akita. M1c and M5c therefore keep
thin public re-exports at those paths. RP3 ports `jolt-akita` to the jolt
paths, and RP4 deletes the re-exports.

```mermaid
graph LR
  RP0 --> RP1 --> RP2 --> RP3 --> RP4 --> RP5
  RP5 --> After["After Whiteout: M7, M8, M9, M10, M11"]
```

### Development workflow

Review points govern what lands upstream, not how fast the work proceeds. All
development happens on one integration branch per repo, which runs as far ahead
of upstream as needed. At each review point, the tickets due are extracted into
PRs by cherry-picking.

**Branches.**

| Repo | Integration branch | Started from | Extraction targets |
|---|---|---|---|
| jolt (a16z/jolt) | `whiteout/dev` | the transcript-unification branch (#2000) | `main` |
| Akita (LayerZero-Labs/akita) | `whiteout/dev` | the transcript-unification branch (#185) | `main` for seams (M1c, M2b, M3b, M5c); `dev` for the rest |

The integration branches stay local; only extracted PRs are pushed. Until M0
lands, both build on the transcript-unification branches and merge them as they
move. Akita's `dev`-only code, including the `akita-zk-*` building blocks, enters
the Akita integration branch through a merge of Akita `dev` before RP4, when the
first `dev`-targeted tickets are extracted.

The integration branches never merge into upstream `main` or `dev`; only
extracted PRs do. They are never force-pushed, because several agents share
them. The jolt integration branch pins Akita's `whiteout/dev` by revision
through ordinary pin commits, and the Akita integration branch pins jolt's. For
local work across both repos, an untracked `.cargo/config.toml` `[patch]`
pointing at sibling checkouts replaces the pins without committing them.

**Commits.** Extraction works only if the history is already shaped like the
PRs:

- Every commit belongs to exactly one ticket and one repo, and its subject
  starts with the ticket ID: `[M2a] jolt-sumcheck: add the eq-factored round domain`. Pin bumps
  are tagged `[pin]` and upstream syncs are merge commits; neither is ever
  extracted.
- A change to a ticket that already has commits is a new commit tagged with that
  ticket: `fixup! [M2a] ...`. Extraction squashes it in.
- Work follows ticket order. A commit for a ticket depends only on commits for
  the same ticket or for tickets earlier in the review-point order.
- The add-first rule holds on the integration branch too. Making a breaking
  change in both repos at once would be faster to write, but it would have to be
  rewritten at extraction.
- Akita seam tickets touch only code that also exists on Akita `main`, so they
  cherry-pick onto `main` cleanly.

**Extraction.** For each ticket due at a review point:

1. Branch `whiteout/pr/<ticket>` from the upstream target.
2. Cherry-pick the ticket's commits in order, squashing its `fixup!` commits.
3. Set pins to upstream revisions only, never an integration revision.
4. Run that repo's suites, check the I3 byte-identical claim, and open the PR.

A ticket that does not cherry-pick cleanly signals a commit-discipline problem.
Fix that on the integration branch with a `fixup!`, not only in the PR.

**Convergence.** Once a PR merges upstream (possibly changed in review), merge
the upstream branch into the integration branch. Conflicts in that ticket's code
resolve to the upstream version, because landed code is canonical. Then run the
combined suites. Merge upstream into the integration branches at every
review-point gate and whenever upstream moves under unlanded work.

**Testing the integration branches.** The integration branches have no CI, so
nothing is committed to them unless the full jolt suites, the Akita suites, and
the combined `akita` + `zk` build pass locally.

**Cost.** The further the integration branch runs ahead of what has landed, the
more convergence work each review produces. When review changes a ticket
heavily, later tickets that build on it are re-resolved against the landed
version during convergence. That cost is accepted in exchange for never waiting
on a review to keep working.

### Deferred tickets

M7–M11 run after RP5, each as its own add-then-remove sequence. While they wait,
Whiteout follows three guards, so that deferring them leaves only mechanical
work:

- **Codecs.** New proof messages use spongefish codecs, never `AkitaSerialize`.
- **Kernels.** New Akita sumcheck kernels (W5's ring-quadratic reduction)
  implement the `jolt-sumcheck` prover trait.
- **No logic in the adapter.** Akita-side Whiteout logic lives in Akita's
  protocol crates; `jolt-akita` holds only the thin `HidingCommitment` impl
  until M8.

## Evaluation

### Acceptance Criteria

- [ ] I1 holds: a CI step fails if a shared-layer crate depends on `akita-*`, or
      if any jolt crate other than `jolt-akita` depends on Akita.
- [ ] After each migration, every row of the Summary table that it covers has one
      implementation. The deleted crates listed under Target layout no longer
      exist.
- [ ] I3 holds for every M ticket except M6c: on fixed-seed inputs, Jolt's
      verifier fixtures pass unchanged, and Akita's proof bytes are
      byte-identical to the pre-change prover. Shown in each PR.
- [ ] While the pinned Akita uses an API, no jolt PR removes it: the pinned
      Akita builds in jolt's workspace after every jolt PR. Removals (M1d, M2c)
      land only after Akita has switched in RP2.
- [ ] M2b: Akita's sumcheck kernels run through `jolt-sumcheck`, and its test
      suites pass, including the eq-factored and grinded-challenge
      paths. No infallible prover trait remains. A kernel that returns an error
      and a kernel that panics both surface as a typed `SumcheckError` from the
      prove entry point, without aborting the process.
- [ ] M6b and W4: a masked transcript and its pads satisfy the residual rows,
      the Σ-protocol is complete with the abort rate its parameters predict, and
      tampered masked messages are rejected. Tested on synthetic relations.
- [ ] W6: an end-to-end ZK Jolt + Akita proof verifies with the `akita` and
      `zk` features together. Verifier fixtures and a tamper sweep over masked
      messages and the pad commitment are recorded. Proof size and prover time
      are reported against the paper's estimate.
- [ ] M6c: clear, ZK, and verifier-fixture suites pass, and the ZK fixtures are
      regenerated.
- [ ] M5a: in the Gaussian sampler and the rejection rule, every branch, loop
      bound, and memory address is independent of the output and the witness,
      confirmed by review of the generated code. (The sampler's trial loop
      depends on fresh coins but not on the output, which rejection sampling
      guarantees.) A documented dudect-style timing tool
      (fixed-vs.-random secret inputs) shows no detectable timing difference.
      The tool is kept as an intentional diagnostic, not a CI test.
- [ ] I4: no shared eq or multilinear-evaluation API takes an untyped `&[F]`
      point.
- [ ] I5: no verifier re-serializes decoded proof data.
- [ ] I6: workspace clippy passes in both modes
      (`--features host` and `--features host,zk`) after each migration.

### Testing Strategy

The Jolt CI suites (standard, ZK, akita, verifier fixtures) and the Akita
workspace suites must pass at every migration. Moved tests move with their code. Under the repo's testing
guidelines, an old-vs.-new equivalence harness is part of the PR process and is
not added to the permanent suite. The one new permanent test is the I1
dependency check.

### Performance

Each migration that touches a hot path (M1b, M1c, M2b, M7, M10) must show no regression
beyond run-to-run noise. Noise is measured on the same machine with five runs
before and after, on:

- `jolt-prover profile` for `fibonacci` and `sha2-chain`, both `--features akita`
  and the default build;
- Akita's commit and opening benchmarks.

M2b is the riskiest. Akita's kernels currently call `send_all` directly, and the
recorder indirection must stay monomorphized.

## Ambiguity register

- **Spongefish permutation hook.** It is unverified whether spongefish accepts an
  external Keccak permutation. Either outcome is acceptable (§6): if it doesn't,
  the transcript uses only spongefish's Keccak.
- **Upstream `GruenSplitEq` equivalence.** It is unverified whether Akita's
  `GruenSplitEq` and `jolt-poly::GruenSplitEqPolynomial` compute the same thing
  with different conventions, or differ in semantics.
- **`offset_eq` boundary.** `eval_affine_digit_intervals` is generic math, but
  its API is shaped around Akita's digit layout. Proposed: it stays in Akita
  until a second consumer exists.

## Documentation

The book's architecture page gets the target layout and the ownership rules. The
`jolt-zk` module docs replace `specs/jolt-prover-blindfold.md`'s description of
per-round Pedersen commitments once M6c lands.

## References

- Whiteout draft: `akita-paper/whiteout.tex`, §3 (residual relation) and §4
  (protocol).
- `specs/jolt-transcript-narg.md` (M0), `specs/jolt-prover-blindfold.md`,
  `specs/verifier-closure-lints.md`, `specs/consolidate-field-traits.md`.
- Surveys behind the Summary table: the transcript-unification branches at jolt
  27267cdae and akita c521f8f1.
