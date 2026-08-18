# Spec: Fiat-Shamir Soundness Analysis for `jolt-verifier`

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | @markosg04                     |
| Created     | 2026-07-26                     |
| Status      | proposed                       |
| PR          |                                |

## Summary

This campaign adds blocking CI checks and adversarial test generation for
Fiat-Shamir under-constraints in `jolt-verifier` and its production dependency
closure. It runs independently over every supported verifier instance: currently
Dory clear, Dory ZK, and Akita clear. It does not pin transcript bytes or require
compatibility with an earlier implementation. A protocol change may alter every
challenge and still pass; it fails only when a prover-controlled degree of freedom is
no longer constrained algebraically or bound into fresh randomness before the check
that depends on it.

The core oracle constructs a semantically invalid proof that preserves a verifier
equation under frozen challenges. The production transcript must make a relevant
later challenge change and reject the proof. Static obligation analysis prevents new
anonymous transcript operations and mines binding requirements from Jolt's symbolic
claim metadata; nightly attack synthesis searches the current verifier for missing
requirements.

## Intent

### Goal and scope

Protect and audit the Fiat-Shamir soundness of `jolt-verifier`, including transcript,
sumcheck, opening, BlindFold, commitment-scheme, claim-model, and derive code reached
by a concrete Dory or Akita verifier build. `jolt-prover-legacy` is outside the
analysis and mutation closure. It may generate honest fixtures or construct
adversarial fixtures that require witness-side recomputation, but prover/verifier
agreement is never a soundness oracle.

A **Fiat-Shamir degree of freedom** is a prover-controlled value, or a coordinated
change to several values, that can make an underlying claim false while preserving
the verifier's algebraic checks for already-fixed challenges. Each such freedom must
either be removed by a deterministic check or affect a fresh challenge used by a
later randomized check.

The analysis uses four stable identities:

- `FsSourceId`: a proof leaf, public statement component, commitment, claim vector,
  or round message controlled by the prover or statement.
- `FsChallengeId`: a named squeeze, using `JoltChallengeId` where one exists and
  stage/batch/round IDs for protocol-engine challenges.
- `FsCheckId`: the randomized or deterministic verifier check that discharges an
  obligation.
- `FsObligation`: either `BindBefore(source, challenge)` or
  `CheckBy(source, check)`.

### Security properties

- **FS-1, complete inventory.** Every security-relevant proof/public source and every
  challenge in the verifier closure has a stable ID. Adding a `JoltProof` leaf or an
  anonymous `challenge*` call fails CI until it is classified.
- **FS-2, binding order.** For every `BindBefore(s, c)`, all paths exercised by the
  supported protocol configurations absorb `s` before squeezing `c`.
- **FS-3, live adversarial oracle.** Every registered attack accepts under its
  frozen-challenge control, proving that the mutation preserves the targeted
  algebraic checks when Fiat-Shamir binding is removed. The production transcript
  rejects at or after the targeted challenge.
- **FS-4, semantic invalidity.** An attack records the violated underlying claim
  (for example, an individual opening no longer matches its commitment). Merely
  producing a different valid proof is not a finding.
- **FS-5, instance coverage.** Obligations and attacks run separately over every
  supported `(PCS, ZK)` verifier instance, including advice and committed-program
  branches. The current matrix is Dory clear, Dory ZK, and Akita clear. Coverage on
  Dory does not cover Akita. If a ZK Akita verifier becomes buildable, it enters the
  required matrix before it can be declared supported.
- **FS-6, no production oracle.** Fixture generation, prover behavior, prior
  transcript bytes, and prover/verifier equality cannot cause a soundness test to
  pass.

### Non-goals

- Transcript compatibility, stable challenge values, and byte-identical schedules.
- Prover analysis or instrumentation beyond fixture generation.
- A formal proof of Fiat-Shamir soundness or random-oracle security.
- Hash-function cryptanalysis or statistical testing of challenge distributions.
- Raw-byte proof fuzzing as the primary strategy; it spends most executions on
  decoding and shape rejection rather than algebraic degrees of freedom.
- Proof uniqueness or non-malleability when all underlying claims remain true.

## Evaluation

### The adversarial oracle

Every dynamic attack has one fixed evaluator:

1. Generate or load an accepted synthetic fixture and record its typed challenge
   trace.
2. Synthesize a structurally valid proof/public-input mutation with an explicit
   violated claim. Clear attacks normally mutate the fixture in place. A ZK or
   PCS-native attack may invoke a fixture constructor to recompute commitments and
   subproofs against the frozen challenge tape.
3. Verify the mutation with `FrozenTranscript`, which replays the recorded challenge
   kinds and values. It must accept; rejection before completion means the attack
   does not isolate the claimed Fiat-Shamir defense.
4. Verify the same mutation with the production transcript. It must reject, and the
   typed trace must show divergence at a challenge named by the obligation.

Production acceptance is a soundness candidate and a hard failure. Frozen rejection
is a coverage failure, not evidence that the production verifier is safe. A
production rejection before the targeted challenge is also a coverage failure
because an unrelated check masked the intended test.

`jolt-fs matrix` enumerates the declared `pcs = {dory, akita}` and
`zk = {off, on}` axes, compile-probes each corresponding verifier feature set, and
requires a fixture adapter for every buildable combination. Its current expected
matrix is:

| Instance | Features |
|---|---|
| Dory clear | `prover-fixtures` |
| Dory ZK | `prover-fixtures,zk` |
| Akita clear | `akita,prover-fixtures` |

`akita,prover-fixtures,zk` is currently unsupported. If that combination starts
compiling but has no obligation trace and attack fixture, `fs-obligations` fails.

### CI jobs

| Job | Cadence | Blocking | Work |
|---|---|---|---|
| `fs-obligations` | PR | Yes | Source/schema census, anonymous-operation lint, symbolic obligation mining, and binding-order checks |
| `fs-attacks-smoke` | PR | Yes | Every attack family on each applicable supported verifier instance; at least one live attack on every instance |
| `fs-attack-sweep` | Nightly | Yes | Every applicable attack site across the full fixture/configuration corpus |
| `fs-coordinated-fuzz` | Nightly | Fails on finding | Grammar-aware combinations of valid attack mutations, guided by verifier progress |
| `fs-verifier-mutants` | Weekly | Report, then floor | Delete/delay absorbs, reuse challenges, change framing/kind, and omit vector elements in the verifier closure |

PR jobs are path-filtered from the production dependency graph, not a handwritten
crate list. Fixture-only prover changes do not trigger them. Fixture generation is a
separate cached setup step; attack workers execute only the verifier.

### Acceptance criteria

- [ ] A derived `FsVisit` inventory covers every `JoltProof` field, nested clear/ZK
      claim leaf, public input, verifier configuration value, and preprocessing value
      read by verification. New unclassified leaves fail a structural test.
- [ ] `jolt-fs matrix` discovers Dory clear, Dory ZK, and Akita clear, and requires
      every newly buildable PCS/ZK combination to provide fixtures and traces.
- [ ] Every challenge draw in the verifier closure is named directly or occurs
      inside an audited combinator that assigns deterministic indexed IDs.
- [ ] Direct `append_to_transcript` calls are allowed only inside codecs or beneath a
      named source absorption; the source analyzer rejects boundary-level bypasses.
- [ ] The obligation miner consumes `SymbolicSumcheck` required openings/challenges,
      `SumcheckBatch` ordering, opening-batch metadata, and explicitly annotated
      stage-level challenges.
- [ ] `FrozenTranscript` records and replays `challenge`, `challenge_scalar`,
      `challenge_vector`, and `challenge_scalar_powers` without conflating their
      kinds or squeeze counts.
- [ ] Opening-batch cancellation, relation-batch cancellation, sumcheck
      equivocation, statement replay, and cross-proof splicing each have a live PR
      canary on every applicable verifier instance. The opening-batch family has
      Dory homomorphic-RLC and Akita native/prefix-packed constructors.
- [ ] Every family/instance pair has either a live canary or an inapplicability
      certificate identifying the deterministic check or trivial-kernel rank
      calculation that removes the degree of freedom. A committed or ZK
      representation is not by itself a reason to mark an attack inapplicable.
- [ ] The full nightly sweep reports zero production-accepted invalid proofs and zero
      applicable sites without a live control.
- [ ] A failing job uploads the mode, fixture recipe, attack ID, violated claim,
      source/challenge/check IDs, production and frozen traces, and serialized
      mutated proof.
- [ ] Instrumentation is test-only and causes no proof-format or public verifier API
      change.

### Metrics and budgets

Report counts per verifier instance, not a single blended score: inventoried sources,
named challenges, mined obligations, applicable attack sites, live controls,
uncovered sites, production-accepted invalid proofs, and mutant kill rate by
operator class.

`fs-obligations` should complete within two minutes. `fs-attacks-smoke` should
complete within ten minutes on a fixture-cache hit. The nightly sweep has a
60-minute budget; coordinated fuzzing uses a fixed per-mode time budget and retains
its corpus and findings as artifacts. Mutation testing starts report-only until one
month of runs establishes a stable per-operator baseline.

## Design

### Typed audit surface

An audit transcript decorates the production transcript but does not infer semantics
from byte payloads. Security-relevant boundaries call typed helpers that emit
`Absorb(FsSourceId)`, `Challenge(FsChallengeId, ChallengeKind)`, and
`Check(FsCheckId)` events. Serialization beneath an already-named absorption remains
unchanged, including direct `AppendToTranscript` calls in polynomial commitment and
sumcheck codecs.

Generic engines receive an `FsScope` from the verifier. A sumcheck scope derives
round IDs from `(stage, batch, round)`; an opening scope derives reduction IDs from
the opening scheme and batch index. This supplies attribution without parsing labels
or relying on Rust type names. With auditing disabled, helpers delegate directly to
the underlying transcript and the observer is a zero-sized no-op.

The top-level API continues to construct the production transcript. Internally, the
stage spine is factored behind a crate-private seeded-transcript entry point so the
attack harness can pass an `AuditedTranscript<T>` or `FrozenTranscript` instance and
recover its events. This hook is compiled only for the audit feature and does not
change the public verifier API.

`FsVisit` is derived on verifier-owned proof and claim structs. Vectors and options
produce stable wildcard paths plus concrete indices in traces. External commitment
and proof types get manual leaf implementations at the verifier boundary; their
internal serialization is not treated as separate protocol sources.

### Program analysis and obligation mining

The static analyzer resolves the production dependency closure for each matrix
instance and performs three checks:

1. A pinned-nightly `rustc_driver` wrapper inspects expanded HIR and resolves calls
   to the `Transcript` and `AppendToTranscript` trait method definitions. Anonymous
   challenge draws and source-boundary absorbs outside audited helpers are errors;
   codec-internal calls require a narrow reviewed annotation.
2. Proof-schema comparison. The `FsVisit` source set must cover the verifier-owned
   serialized schema and every public/preprocessing field read by verification.
3. Semantic graph construction. Existing `JoltOpeningId`, `JoltChallengeId`,
   `JoltRelationId`, `SymbolicSumcheck` expressions, and generated batch ordering
   produce candidate source/challenge/check edges. Handwritten obligations are
   limited to stage-level composition, PCS reductions, and values whose constraint
   is intentionally deferred.

The analyzer compares this independent obligation graph with typed traces from every
fixture configuration. A trace demonstrates path coverage; it does not define the
obligation. Unobserved conditional obligations remain uncovered and fail the nightly
sweep.

### Attack synthesis

| Attack | Preconditions | Constructed invalidity |
|---|---|---|
| Opening-batch kernel | At least one nonzero degree of freedom after collecting all frozen affine checks on final claims | Solve for a nonzero claim delta in the frozen verifier's kernel. The Dory constructor preserves the homomorphic joint claim/commitment equation; the Akita constructors preserve native-batch or prefix-packed `alpha`/`beta` reductions. At least one individual opening is false. |
| Relation-batch kernel | Independently meaningful claims folded by random coefficients | Build the full frozen coefficient matrix for every check touching the candidate claims, then perturb along a nonzero nullspace vector so all folded claims remain unchanged. |
| Sumcheck equivocation | A round of sufficient degree and a mutable claimed sum/round message | Perturb the claimed sum and interpolate a round correction whose Boolean-endpoint sum accounts for the perturbation and whose value at the frozen round challenge is zero. |
| Statement replay | A public or preprocessing source with a same-shape alternative | Reuse the proof under a different statement/configuration value. |
| Cross-proof splicing | Two accepted same-shape fixtures | Replace a stage, claim group, commitment group, or opening payload so the resulting cross-stage claims are inconsistent. |

The two kernel attacks use symbolic batch metadata to include every earlier frozen
linear check, rather than assuming that preserving one weighted sum is sufficient.
If the matrix has trivial kernel, the site has no attackable degree of freedom and
the artifact records the rank calculation. Sumcheck attacks use the explicit
interpolation template. For ZK or PCS-native sites where serialized mutation cannot
preserve commitments, the constructor may regenerate only the necessary fixture
subproofs against the frozen tape; that generator is not part of the security oracle.
The same rule applies to attack-family coverage: representation differences select a
different constructor, while only a deterministic constraint or a trivial kernel
justifies an inapplicability certificate.

Delta, selected coordinates, round, splice boundary, and constructor choice form the
input grammar for `fs-coordinated-fuzz`. Fuzzing maximizes verifier progress and
unexplored source/challenge pairs; acceptance is always the failure condition.

### Mutation testing

The source mutator operates only on `jolt-verifier` and its production dependency
closure. Operators delete or move a named absorption, replace a challenge with an
earlier value, swap `challenge` and `challenge_scalar`, alter vector length framing,
drop one repeated absorption, and change one `cfg(zk)`/`cfg(akita)` arm. A mutant is
killed only by the obligation or adversarial jobs; schedule differences do not
count. Each survivor is uploaded as a patch with the attack sites exercised against
it.

### Alternatives considered

Committed schedule snapshots are useful diagnostics but rejected as the oracle:
regenerating a snapshot can bless the missing absorption it was meant to detect.
Prover/verifier transcript equality is rejected because a shared omission preserves
equality. Untyped recording at `append_bytes` is rejected because it cannot
distinguish labels from 32-byte values or recover proof-field provenance through
direct `AppendToTranscript` calls.

## Execution

1. Add `jolt-fs matrix`, typed source/challenge/check IDs, `FsVisit`, audit traces,
   and `FrozenTranscript`. Bring up statement replay and one opening-batch kernel
   canary on Dory and Akita; require the four-step oracle before proceeding.
2. Add the source/schema analyzer and symbolic obligation miner. Make anonymous
   challenges and newly unclassified proof leaves blocking.
3. Add relation-batch and sumcheck attack synthesis, then run the PR smoke matrix for
   Dory clear, Dory ZK, and Akita clear.
4. Add cross-proof splicing and the full nightly site/configuration sweep. Publish
   machine-readable coverage and finding artifacts.
5. Add coordinated fuzzing and verifier-closure mutation testing. Establish
   per-operator mutation floors only after the report-only baseline is stable.

The initial implementation may reuse the existing tamper manifest and fixture cache,
but its `checked_at` classifications are inputs to obligations, not substitutes for
the frozen-challenge control.

## Open questions

- Which transcript backends are production-supported for each verifier instance?
  This axis must be named before `fs-attack-sweep` becomes blocking.
- Can every stage expose a useful rejection/check ID without changing public error
  variants? If not, test-only check events are preferred over string-matching errors.
- Which sumcheck degrees admit the generic equivocation template after compressed
  round encoding? Unsupported degrees must be marked not-applicable with a derivation,
  not silently skipped.

## Documentation

Add a Fiat-Shamir soundness section to `book/src/dev/testing-gates.md`: the
four-step oracle, how to reproduce an attack artifact, the difference between a
soundness finding and a coverage failure, and the review process for a new
obligation or codec annotation. The audit feature is internal test infrastructure
and is not part of the user-facing verifier API.

## References

- #1696 - verifier dependency fuzz infrastructure and fixture execution support.
- #1697 - verifier fixture cache, tamper manifest, and soundness CI conventions.
- [`symbolic-sumcheck.md`](symbolic-sumcheck.md) - symbolic opening/challenge metadata
  used by obligation mining.
- Dao, Miller, Wright, Grubbs,
  [*Weak Fiat-Shamir Attacks on Modern Proof Systems*](https://eprint.iacr.org/2023/691)
  (IEEE S&P 2023).
- Trail of Bits,
  [*Coordinated disclosure of vulnerabilities affecting Girault, Bulletproofs, and PlonK*](https://blog.trailofbits.com/2022/04/13/part-1-coordinated-disclosure-of-vulnerabilities-affecting-girault-bulletproofs-and-plonk/).
