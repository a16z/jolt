# Standalone use of jolt-sumcheck

## Status and scope

Implemented on branch `omid/refactor-sumcheck-crate`, based on
`7a052796e37311205cfc0add809bc012c44baa02`. This document records the final
dependency contract, the cross-crate changes, and the regression gates.

An external application must be able to prove and verify clear sumchecks with its
own scalar and transcript without compiling a curve backend, polynomial commitment
system, or Rayon. Explicit, additive features enable stock transcripts,
parallel polynomial operations, committed rounds, and R1CS lowering. Existing Jolt
clear and ZK protocols retain their proof encoding and transcript behavior.

“General” initially means fields implementing Jolt's existing algebra/encoding
traits, with the stronger `JoltField` bundle required by stock polynomial/prover
types. It does not mean every third-party field implements these traits already,
or every field characteristic is supported by every algorithm. This work does
not promise `no_std`, remove all allocations, introduce a generic dense-polynomial
prover, replace the transcript protocol, or prepare a crates.io release.

## Baseline findings

The reviewer's dependency diagnosis is correct, with several additional blockers.

| Boundary | Current evidence | Consequence |
| --- | --- | --- |
| Sumcheck manifest | [Cargo.toml](../crates/jolt-sumcheck/Cargo.toml) unconditionally enables crypto, field, openings, poly, transcript, and `rand_core/getrandom`; only R1CS is optional | Disabling sumcheck defaults does not remove these dependencies |
| Workspace inheritance | [Root manifest](../Cargo.toml) enables defaults on the relevant internal dependency entries | A leaf manifest change alone cannot neutralize the inherited defaults |
| Field | [Field manifest](../crates/jolt-field/Cargo.toml) defaults to `bn254`; poly and crypto also request field defaults | BN254 arrives even without a direct sumcheck-to-crypto edge |
| Crypto | [Crypto manifest](../crates/jolt-crypto/Cargo.toml) defaults to `bn254`, which in turn enables `parallel` | Backend and threading are coupled; the reviewer described the effective defaults accurately |
| Crypto source | [Pedersen](../crates/jolt-crypto/src/ec/pedersen.rs) imports `Fr` unconditionally and fixes `VectorCommitment::Field = Fr` | Disabling manifest defaults alone does not make crypto backend-free |
| Polynomial | [Poly manifest](../crates/jolt-poly/Cargo.toml) defaults to `parallel` | Clear univariate proof types bring Rayon even though the round scheduler is sequential |
| Openings | [claim.rs](../crates/jolt-sumcheck/src/claim.rs) re-exports `jolt_openings::EvaluationClaim` | This dependency is used, but only to share a small polynomial evaluation claim |
| Transcript | [Manifest](../crates/jolt-transcript/Cargo.toml) enables all three sponges by default and unconditionally requests Arkworks, Spongefish Arkworks codecs, and field BN254 | Even a custom transcript implementation pays for stock backends |
| Transcript source | [legacy.rs](../crates/jolt-transcript/src/legacy.rs), [lib.rs](../crates/jolt-transcript/src/lib.rs), [prover.rs](../crates/jolt-transcript/src/prover.rs), [setup.rs](../crates/jolt-transcript/src/setup.rs) mix traits with sponge code, default `Fr` parameters, BN254 challenge helpers, and `StdRng` factories | Requires source separation, including generic defaults and re-exports |
| Proof/prover API | [proof.rs](../crates/jolt-sumcheck/src/proof.rs), [recorder.rs](../crates/jolt-sumcheck/src/recorder.rs), [prover.rs](../crates/jolt-sumcheck/src/prover.rs) make clear recording return `SumcheckProof<F, C>` and an optional committed witness | A clear caller must name a meaningless commitment parameter; gating one module is insufficient |
| R1CS | [R1CS manifest](../crates/jolt-r1cs/Cargo.toml) unconditionally depends on [jolt-claims](../crates/jolt-claims/Cargo.toml), which depends on openings and RISC-V types | Enabling R1CS would reintroduce a large Jolt graph even after fixing clear mode |
| Tests | Sumcheck unit, integration, and fuzz tests predominantly use `Fr` and stock Blake2b | Workspace tests cannot establish that a minimal external consumer works |

Inspection included a successful offline resolution:

`cargo tree -p jolt-sumcheck --no-default-features --edges normal,build --offline --prefix none`

It contains BN254, crypto, openings, Rayon, Light Poseidon, all stock sponge
dependencies, and `getrandom`. It also contains the fork's Arkworks 0.5 and
Spongefish's registry Arkworks 0.6 simultaneously. This is evidence of dependency
cost and potentially incompatible codec types, not a claim that every combination
fails. The implementation replaces that graph with the feature contract below
and checks it through an isolated external-consumer fixture.

## Dependency and feature contract

### Sumcheck features

Set `default = []`. Basic claims, domains, clear verification, stock clear proof
types, the sequential batch driver, clear recorder, and clear uni-skip remain
available with no features. Both omitted features and `default-features = false`
must yield this behavior in an isolated consumer.

| Feature | Enables | Must not implicitly enable |
| --- | --- | --- |
| `parallel` | `jolt-poly/parallel`, `jolt-field/parallel`, and weak forwarding to optional crypto/R1CS parallel features | Committed mode, any curve, or a transcript backend |
| `transcript-blake2b` | `jolt-transcript/transcript-blake2b` | BN254, Poseidon, Keccak, Rayon |
| `transcript-keccak` | `jolt-transcript/transcript-keccak` | BN254, Poseidon, Blake2b, Rayon |
| `transcript-poseidon` | The current BN254 Poseidon implementation and its required field/backend dependencies | Committed mode or Rayon |
| `committed` | Optional `jolt-crypto` with defaults disabled and optional `rand_core` without `getrandom` | A concrete curve, stock transcript, R1CS, or Rayon |
| `r1cs` | Optional minimal `jolt-r1cs`, defaults disabled | Committed mode, Jolt claim lowering, curves, or Rayon |

Do not add a sumcheck `zk` feature: committed transcript consistency is not a
complete zero-knowledge proof. Jolt's existing `zk` feature continues to select
BlindFold and protocol validation at the application layer.

Consumers select a concrete scalar through `jolt-field` and a concrete commitment
through `jolt-crypto` (same pinned Jolt revision), or implement the existing traits.
Do not add a sumcheck backend selector solely to forward those choices. Poseidon
is an explicit exception to backend independence because its current sponge is
intrinsically BN254; document that fact even when its challenge type is generic.

### Dependency propagation

1. Disable defaults on each dependency edge in the standalone sumcheck closure.
   Do not globally change the root workspace declarations merely to make this
   leaf minimal: that would silently change every workspace consumer and create
   a much larger migration. A member using `workspace = true` must add
   `default-features = false` on its own edge where needed. Audit direct `path`
   declarations too; they bypass workspace dependency policy.
2. On every edge in the supported standalone closure, disable defaults. Keep
   defaults in other packages only if their default surface is intentional;
   standalone sumcheck must never inherit them accidentally.
3. Use weak optional forwarding such as `jolt-crypto?/parallel` and
   `jolt-r1cs?/parallel`. Enabling parallelism must not activate optional packages.
4. Remove `parallel` from crypto's `bn254` feature. Explicitly preserve parallelism
   for existing Jolt applications at their owning manifest. Check BN254's
   sequential implementations rather than assuming this is a manifest-only edit.
5. Remove Spongefish's workspace-wide `sha3` feature floor unless a production
   consumer requires it; move that request to its actual caller. Audit third-party
   defaults and features as well as internal edges.
6. Remove sumcheck's direct entropy request. Committed proving already takes a
   caller-supplied RNG. `rand_core` can remain transitively present through field
   traits; that does not require OS entropy or a thread pool.

Cargo combines features requested by consumers of the same package. Thus these
guarantees apply to the isolated sumcheck closure; another dependency can enable
BN254 or Rayon in an application's combined graph. There is no negative
“sequential” feature that overrides another consumer's request. See the
[Cargo feature reference](https://doc.rust-lang.org/cargo/reference/features.html).

## Source and API changes

### 1. Give polynomial evaluation claims a lower-level owner

Move the existing `EvaluationClaim<F>` definition and constructor into
`jolt-poly`, alongside `Point`. Preserve `Point<HIGH_TO_LOW, F>`, derives, field
order, and constructor behavior. Re-export the same type from sumcheck and
openings; do not create two equivalent claim types or reverse the dependency so
openings depends on sumcheck. `jolt-claims` is not an appropriate destination:
it already depends on openings and carries protocol-specific machinery.

Move the `AppendToTranscript` implementation with the type, behind a new
`jolt-poly/transcript` feature using a defaults-disabled optional transcript
dependency. Openings enables that feature to preserve its public contract.
Sumcheck needs only the claim data, so need not enable it. Transcript must depend
only on field at its trait boundary, never on poly; this keeps the graph acyclic.
Retain the exact `opening_point`/count/coordinate/`opening_eval` byte sequence.
Remove sumcheck's openings dependency entirely, including committed mode.

### 2. Make jolt-transcript trait-only without defaults

Use the existing package rather than adding a second public transcript trait
crate. Move `Transcript`, `AppendToTranscript`, canonical field absorption, labels,
count framing, and `append_length_prefixed` into an always-available internal
module. Keep their public paths and single trait identity. This module depends
on defaults-disabled `jolt-field`, with no sponge, digest, curve, or RNG backend.

Separate the remaining capabilities:

- A `spongefish` feature owns the optional Spongefish dependency, codecs, generic
  split prover/verifier traits, sponge adapter, and setup factories. Stock sponge
  features activate it. Gate the setup RNG dependency and explicitly request
  `StdRng` requirements instead of relying on unrelated feature unification.
- A `digest` feature owns `DigestTranscript` and the digest dependency; Blake2b
  activates it to retain `LegacyBlake2bTranscript`.
- An `arkworks` feature owns Spongefish's Arkworks codec integration. Stock byte
  sponge facades do not need it; their canonical encoding uses Jolt field traits.
- A `bn254` feature owns `jolt-field/bn254` and the `Fr`-specific
  `OptimizedChallenge` API. Poseidon additionally owns its actual Arkworks and
  Light Poseidon dependencies. Gate optimized impls by both BN254 and their sponge.
- Remove the implicit `Fr` default from the generic `SpongeTranscript` type.
  Stock aliases accept an explicit field in every configuration. When `bn254` is
  enabled they retain the existing `Fr` default so current Jolt source remains
  compatible; without `bn254`, callers must name the field and no BN254 type is
  referenced.

Make `jolt-transcript` default to its trait surface too; migrate stock consumers
in the same implementation. The resulting smaller boundary serves sumcheck,
openings, and crypto immediately, so a new package is unnecessary. Keep the
existing facade as a supported external contract for this work; its retirement
note must not promise removal without a separately reviewed migration. Do not
combine dependency separation with switching sumcheck to the split NARG API.

The two Arkworks versions require an explicit adapter compatibility check. Keep
byte transcript use independent of either codec. Test any advertised native
codec using the actual field type and source identity; do not infer compatibility
from similar names or a successful transcript-crate build.

### 3. Isolate committed recording from clear use

Keep `ClearProof<F>`, full/compressed proof types, `RoundMessage`, `ClearRound`,
generic committed proof data, and clear verifier methods unconditional. Generic
committed data carries no crypto dependency and preserves the stable serialized
`SumcheckProof<F, C>` envelope. Gate the crypto-backed builder, committed recorder,
round commitment operation, RNG imports, and committed uni-skip prover under
`committed`.

Retain `SumcheckRecorder` and `RecordedSumcheck` so existing generated stage
drivers and the Jolt proof envelope do not require a source or wire migration.
`ClearSumcheckRecorder<F, C = ()>` and `ProvedUniskip<F, C = ()>` default the
otherwise phantom commitment parameter to `()`, giving standalone clear callers
the simple one-field-parameter surface. `prove_batch` remains the single
recorder-generic engine. This compatibility choice preserves scheduler ownership,
round order, and fused bind/evaluate behavior.

### 4. Make optional crypto and R1CS genuinely optional capabilities

For crypto, keep the generic `VectorCommitment` and group interfaces usable
without a field backend. Gate the current `Fr`-specific Pedersen implementation
and exports under `bn254`, and explicitly enable `jolt-field/bn254` there. This
honestly reflects the current implementation. Generalizing Pedersen/group scalar
typing is separate work; external committed users can implement `VectorCommitment`.
Audit all generic exports for unconditional backend imports.

For R1CS, keep builder, variables, linear combinations, and matrix algebra usable
without `jolt-claims`. Make the latter optional behind an explicit
`claim-lowering` integration feature, gate `lowering.rs` and its exports, and make
`field-inline` request the dependency it uses. Audit ISA-specific modules for
edges back into that integration. Jolt/BlindFold callers opt into claim lowering;
sumcheck's `r1cs` feature opts into only the algebraic layer. Disable field/poly
defaults on that path. Existing default R1CS behavior may be preserved by its
default features, since sumcheck disables them explicitly.

The implementation of `SumcheckR1csRound` for `VerifiedCommittedRound` requires
both `r1cs` and `committed`; the lowering trait and generic lowering functions
require only `r1cs`. Do not make `r1cs` imply committed proofs merely for this impl.

### 5. Document and enforce the actual external contract

The generic verifier already accepts `SumcheckScalar` and custom `ClearRound`
types. Preserve that lighter boundary. Stock polynomial/prover APIs currently
require `JoltField`, including accumulator and serde capabilities; document how
to implement its component traits with a local field wrapper. Do not claim an
arbitrary Arkworks field is accepted directly. Remove redundant bounds in
`SumcheckScalar` without inventing another algebra hierarchy.

Audit public dimensions before arithmetic/allocation. In particular,
`BatchPrelude::new` subtracts a member's rounds before validating it;
`prove_batch` adds offset and rounds unchecked, computes `degree + 1`, and unwraps
the inverse of two. Provide checked construction with typed errors, retain
validation at the driver boundary while fields remain publicly constructible,
and reject unsupported characteristic-two batching before transcript mutation.
Validate the power-of-two padding bound imposed by the current `mul_pow_2` helper.
Use checked dimension arithmetic for externally supplied round and domain sizes.
Document the existing nonzero degree requirement and centered-integer domain
limits; do not silently widen their supported range during this refactor.

Document that the caller binds the protocol, public statement, degree/round
bounds, polynomial identity, and batch ordering before drawing relevant
challenges. `SumcheckVerifier::verify` does not bind all that automatically.
Preserve Jolt's existing absorb schedule rather than adding new absorbs inside
the engine. After a reduction, the caller must discharge `EvaluationClaim`,
including the zero-variable case. Batching coefficients must be sampled after
binding the batch statement, not supplied adversarially. Explain how to form a
`BatchPrelude` using the canonical constructor without copying its padding law.

Stock transcript challenge decoding and its entropy limits must be stated for
the chosen field, including extension fields and the existing truncated challenge
paths. A small field example demonstrates API use, not a production security level.
No custom hash or field wrapper is secure merely because it implements the traits.

## Implemented cross-crate inventory

The final implementation changes the following packages outside
`jolt-sumcheck`. Manifest-only rows deliberately restore capabilities that those
packages previously received from dependency defaults.

| Package | Implemented change | Regression risk controlled |
| --- | --- | --- |
| Workspace root / `Cargo.lock` | Stop enabling Spongefish `sha3` globally; callers select it through transcript features. Refresh the lockfile without changing dependency sources. | Existing transcript backends explicitly recover `sha3`; lockfile source audit. |
| `jolt-transcript` | Empty default set; optional digest, Spongefish, stock sponge, Arkworks, RNG, and BN254 dependencies; trait-only core; cfg-gated backend modules and exports; conditional compatibility defaults for stock aliases. | Backend-specific nextest suites and minimal/per-backend compilation preserve transcript behavior. |
| `jolt-poly` | Disable field defaults on this edge; add an optional transcript feature; move `EvaluationClaim` here with the same layout and transcript schedule. | Poly users retain one type identity; openings tests cover the moved absorb implementation. |
| `jolt-openings` | Enable poly's transcript integration and re-export `EvaluationClaim` from poly; update internal imports and explicit test transcript features. | Opening and Dory-opening suites exercise unchanged public paths and equations. |
| `jolt-crypto` | Disable defaults on internal dependency edges; separate BN254 from `parallel`; retain both in the crate's own default set for compatibility; gate the BN254 Pedersen implementation; provide sequential iteration in BN254 batch-addition/GLV paths when Rayon is absent. | BN254 compiles in sequential and parallel modes; the full 133-test crypto suite covers group, pairing, MSM, Pedersen, and serialization behavior. |
| `jolt-r1cs` | Disable field/poly defaults; make `jolt-claims` optional under `claim-lowering`; gate lowering exports; keep existing defaults for ordinary Jolt consumers. | Minimal `r1cs` graph excludes claims/RISC-V, while the all-feature R1CS suite covers lowering. |
| `jolt-verifier` | Forward each selected transcript backend explicitly instead of receiving all transcript defaults. | Modular clear and ZK builds use their configured backend. |
| `jolt-prover` | Forward `parallel` to crypto and sumcheck; enable sumcheck `committed` only for the `zk` integration. | Clear and ZK acceptance suites cover both feature paths. |
| `jolt-prover-legacy` | Restore crypto/sumcheck parallel features explicitly in its prover bundle. | Required host and host+zk `muldiv` runs cover both legacy modes. |
| `jolt-dory` | Add an explicit default `parallel` owner feature and explicit stock transcript features for tests. | Dory tests exercise commitment/opening paths; sequential crypto remains independently compilable. |
| `jolt-blindfold` | Select the concrete transcript features used by its tests. | BlindFold and modular ZK tests preserve committed verification. |
| `jolt-kernels` | Select the BN254 Blake2b transcript used by its tests explicitly. | Modular prover checks cover reference and optimized kernel integration. |
| `jolt-akita` | Select only Blake2b for tests; disable BN254-bearing field, crypto, and openings defaults. Move the mixed Akita/Dory comparison benchmark to `jolt-dory`. | `scripts/check-akita-dependencies.sh` inspects normal, build, and dev dependencies with all Akita features; none may introduce Arkworks or Dory. |

No source changes were needed in `jolt-verifier-derive`, stage drivers, BlindFold,
Dory, Akita, kernels, or either prover. Keeping the recorder envelope compatible
avoided those migrations. The only supporting source changes are in transcript,
poly, openings, crypto, and R1CS.

## Original migration audit

The table below records the pre-implementation impact analysis. Rows describe
the expected work and regression evidence; the implemented inventory above is
authoritative where the compatibility design avoided a predicted source change.

### Crates that require implementation changes

| Crate | Required manifest changes | Required source changes | Regression evidence |
| --- | --- | --- | --- |
| Workspace root and `Cargo.lock` | Keep existing workspace dependency defaults unless a narrower edge cannot express the required policy. Add or revise shared feature declarations only when every consumer still selects its previous effective features. Lockfile changes must be limited to dependencies made optional/removed or sources intentionally pinned by this work. | No root application source change. Add isolated-consumer CI jobs and scripts without placing fixtures inside the workspace. | Compare resolved feature graphs for every existing mode; review the lockfile source/version diff; run the full required workspace lint and acceptance suites. |
| `jolt-field` | No default-set change is required. Keep backend, parallel, allocative, and assembly features independent. All new minimal inbound edges disable defaults. | None expected unless transcript-specific bounds can be made smaller without changing `JoltField`. Do not alter arithmetic, encodings, accumulators, or backend representations. | All field backend, serialization, accumulator, and fuzz tests; compile each backend alone and in existing Jolt combinations. |
| `jolt-poly` | Its sumcheck/transcript-facing dependencies disable defaults. Add optional defaults-disabled `jolt-transcript` plus a `transcript` feature for evaluation-claim absorption. Keep `parallel` additive and preserve the crate's own existing default unless separately approved. | Own `EvaluationClaim<F>` next to `Point`; preserve layout, derives, constructor, point order, and transcript implementation. Re-export it. No polynomial algorithm changes. | Poly tests/benches compile sequential and parallel; serialization and transcript bytes for the moved claim match golden/live pre-change output. |
| `jolt-transcript` | Default becomes trait-only. Make digest, Spongefish, RNG, Arkworks codecs, concrete sponges, BN254, and Poseidon optional with the feature relationships specified above. Disable field defaults. | Split the always-built traits/labels from stock implementations; remove `Fr` from the generic sponge and from aliases when BN254 is absent; cfg-gate backend imports and exports. Keep one public trait identity and preserve existing wire behavior when a stock feature is enabled. | Per-backend tests in isolation; facade parity fixtures; split prover/verifier symmetry; EOF, malformed codec, label, challenge-width, fuzz, and benchmark coverage. |
| `jolt-crypto` | The edge from sumcheck is optional and defaults-disabled. Separate `bn254` and `parallel`; make field/poly/transcript edges defaults-disabled and explicitly forward needed features. | Gate the `Fr`-fixed Pedersen implementation and BN254 exports. Keep backend-agnostic commitment/group traits available. No group law, MSM, or commitment equation changes. | Generic no-backend compile fixture plus BN254 group/pairing/Pedersen/serialization suites in sequential and parallel modes. |
| `jolt-openings` | Select field, poly, transcript, crypto, and required claim-transcript features explicitly. Preserve current default Jolt behavior. | Replace the local `EvaluationClaim` definition with a public re-export from poly; update internal imports only. Do not change PCS traits, batching, opening equations, or transcript schedule. | Existing mock and Dory opening suites, prefix packed-layout tests, serialization, transcript parity, and homomorphic opening tests. |
| `jolt-r1cs` | Disable defaults on its field/poly edges. Make `jolt-claims` optional behind `claim-lowering`; make `field-inline` request that integration; keep parallel independently selectable. | Gate `lowering` and claim-expression exports. If ISA constraint modules pull claim machinery into the algebraic core, gate them under an accurately named Jolt integration feature. Do not change matrix construction or evaluation. | Builder/matrix tests with minimal features; claim lowering and field-inline tests with integration enabled; sequential/parallel matrix parity through existing reference paths. |
| `jolt-sumcheck` | Set `default = []`; remove openings; make crypto, R1CS, and RNG optional/defaults-disabled; disable field/poly/transcript defaults; add the feature table above. Dev dependencies select their concrete field/transcript features explicitly. | Gate crypto-backed committed operations while retaining generic proof data; default clear phantom parameters to `()`; add checked public construction; update exports/docs/examples. Keep protocol arithmetic and wire encoding unchanged. | Minimal external fixture, feature matrix, existing soundness/roundtrip/committed/Mersenne tests, and Jolt proof parity. |

### Direct sumcheck consumers that require migration

| Crate | Required change | Regression evidence |
| --- | --- | --- |
| `jolt-verifier-derive` | Generate the new recorder associated-result API. Construct checked batch preludes and propagate typed errors. Remove the phantom clear commitment parameter. Continue generating committed consistency code only where the owning build enables it. | Macro expansion/compile tests for clear and ZK; inspect generated code through existing stage builds. The generated Fiat-Shamir order must be byte-identical. |
| `jolt-kernels` | Depend on the clear sumcheck core with defaults disabled. Forward `parallel` to the exact poly/sumcheck capabilities it uses. No committed feature is needed. Adjust only constructor/error signatures introduced by checked batch metadata. | Reference and optimized kernel suites, engine twins, both sequential and parallel builds; no extra trace passes or allocations in hot paths. |
| `jolt-prover` | Explicitly enable sumcheck's committed envelope because proof/session types include it in both protocol builds; enable the committed recorder only on `zk` if the envelope and recorder are separately gated. Forward parallelism deliberately. Adapt stage drivers/recorder to typed clear and committed results and wrap them into the stable proof envelope. | Modular clear byte-diff ratchet, engine twins, full clear acceptance, ZK accept/tamper/advice/committed-program suites, profiling smoke tests, and relevant benchmark baselines. |
| `jolt-verifier` | Explicitly select the sumcheck proof envelope and the configured transcript backend. Its `transcript-*` features must forward to `jolt-transcript` rather than rely on transcript defaults. Enable R1CS/committed consistency on the ZK/BlindFold integration path as required. Adapt proof and stage verifier signatures without changing validation order. | All verifier soundness, Fiat-Shamir attack, proof-shape, completeness, statistical-independence, and transcript-audit suites for each supported transcript; reject protocol/config mismatches as before. |
| `jolt-blindfold` | Enable defaults-disabled sumcheck `r1cs,committed`, minimal R1CS plus its required Jolt integration, and the exact transcript/crypto capabilities used. Update imports for moved/gated consistency and witness types. | Assignment, proof, and sumcheck pipeline tests; malformed/tampered proof rejection; R1CS satisfaction; ZK end-to-end tests. This is the primary guard against weakening committed mode. |
| `jolt-prover-legacy` | Its optional modular dependency bundle must explicitly enable the same field, transcript, sumcheck envelope, committed/R1CS, crypto, and parallel capabilities previously received through defaults for each `minimal`, `host`, `zk`, and transcript feature. Adapt only imports/construction affected by the moved claim and recorder/envelope API. | Legacy `muldiv` in host and host+zk, advice tests in clear mode, all configured transcript modes, proof serialization, Dory bridge, and existing e2e/profile smoke tests. |

`jolt-verifier-derive` is not a direct Cargo dependency of sumcheck, but it is a
direct source-level producer of sumcheck calls and therefore belongs in the API
migration. Conversely, the kernels use sumcheck heavily but should remain on the
clear algebraic surface; adding `committed` there would be a dependency regression.

### Consumers of the supporting crates

These crates need explicit features or import adjustments because transcript,
crypto, openings, poly, or R1CS boundaries change. They must not receive source
changes unrelated to those boundaries.

| Crate | Required change | Regression evidence |
| --- | --- | --- |
| `jolt-dory` | Explicitly select BN254, the transcript facade/backend it uses, poly parallelism according to its owner feature, and openings claim absorption. Update the moved evaluation-claim import only if it bypasses the openings re-export. | Commit/open/verify, tamper fuzz target, streaming, transcript bridge, serialization, and benchmark compilation. |
| `jolt-akita` | Explicitly select Solinas and its actual transcript requirements, with no accidental BN254. Preserve its crypto/openings adapters and use the stable openings re-export. | Akita e2e, native batching, pathology, prefix packing, and both reference/optimized path tests; verify its graph contains no newly introduced BN254. |
| `jolt-claims` | Select poly/openings features explicitly. Keep importing `EvaluationClaim` through the compatibility re-export or migrate once to poly; do not create a dependency cycle. No R1CS dependency or reverse feature forwarding is added here; `jolt-r1cs` owns the optional claim-lowering edge. | Claim derive/expression tests and every generated verifier/prover build. |
| `jolt-eval` | Explicitly enable every stock transcript backend exercised by `transcript_symmetry` and the backend used by telemetry/invariants. Keep these as evaluation dependencies, not sumcheck defaults. | Transcript invariants for all sponges, generated fuzz targets, telemetry objective tests, and guest compilation. |
| `jolt-sdk` | Its host/guest-verifier bundles explicitly enable the field, crypto, Dory, verifier, and transcript features they require. Continue forwarding user-facing transcript selection to the correct legacy and modular owners; avoid enabling all stock sponges through the optional transcript edge. | Host SDK examples, guest verifier build, each transcript selection, host+zk, and the required RISC-V guest builds. |

### Manifest audit and validation-only crates

`jolt-lookup-tables`, `jolt-witness`, `jolt-utils`, and `tracer` depend on
`jolt-field`; `jolt-kernels` and the crates above also depend directly on poly or
R1CS. Because this specification no longer proposes a global change to the root
workspace dependency defaults, the first four should need no source change and
usually no manifest change. Audit their resolved features anyway. If an internal
dependency declaration is changed to defaults-disabled during implementation,
that same crate must explicitly request the backend/parallel feature it actually
uses and join the implementation table above.

The dedicated fuzz packages for field, poly, crypto, transcript, sumcheck, and
Dory use direct path dependencies. Update each manifest to name the backend,
transcript, committed, parallel, and serialization features its target needs.
Do not use broad defaults merely to make fuzz targets compile. Preserve the
existing fuzz targets and seed corpora; add dependency-boundary fuzzing only when
it gives a distinct malformed-input signal.

`jolt-profiling`, `jolt-program`, `jolt-riscv`, `common`, `jolt-platform`, inline
guest crates, examples, `zklean-extractor`, and `z3-verifier` have no direct edge
requiring a planned source migration in the inspected revision. They remain
transitive integration coverage. If compilation reveals an implicit-default
dependency, record the concrete edge, add the smallest explicit feature, and
update this inventory before implementation handoff. Do not preemptively edit
them.

### Files intentionally unchanged

Do not change proof-version constants, serialized enum/struct field order,
Fiat-Shamir labels, transcript domain identifiers, challenge conversion,
sumcheck formulae, Dory equations, BlindFold constraints, generated guest ABI,
or RISC-V execution semantics. No golden proof, verifier artifact, or benchmark
baseline may be regenerated merely to make this refactor pass. A changed golden
artifact means the compatibility invariant failed and the implementation must be
fixed or separately reviewed as a protocol change.

## Compatibility and security invariants

- Existing Jolt proof bytes, discriminants, coefficient compression, point order,
  domain labels, challenge mapping, and absorb/squeeze order remain unchanged.
- Clear round count, degree, and round-sum checks stay in release builds;
  malformed compressed proofs continue to fail with typed errors.
- Committed consistency checks do not prove scalar claim relations. Preserve
  statement-only verification, committed output shape/capacity checks, retained
  witness blindings, and mandatory BlindFold/R1CS/opening verification in Jolt.
- Committed recording never absorbs private coefficients or claims in cleartext.
  Keep claim formulas and BlindFold constraints synchronized, including advice
  reconstruction in standard mode. Backend isolation cannot disable any check.
- API signature changes are intentional and require an in-repo migration and
  external migration notes; this is not a source-compatible patch. No wire
  version change is expected. If parity fails, resolve it before handoff.

## External examples and pinned Git validation

Deliver a crate README and standalone consumer fixtures during implementation.
The README must give a real tested full Git revision, toolchain requirement,
feature table, field/transcript integration contract, and commands that build
and run a consumer. A Git dependency selects package `jolt-sumcheck` from the Jolt
repository; its sibling path dependencies come from that checkout. Examples
using sibling crates must pin them to the same Git source and revision to avoid
distinct trait/type identities. See
[Cargo Git dependencies](https://doc.rust-lang.org/cargo/reference/specifying-dependencies.html).

Required consumer scenarios:

1. **Minimal clear, custom transcript:** no sumcheck features; an explicit non-BN254
   field and a local adapter to an existing cryptographic transcript backend.
   Prove a small, explicitly defined polynomial via `ProveRounds`, replay the
   verifier, and check the resulting evaluation against that polynomial directly.
   The fixture must exercise the lighter custom-round verifier contract too.
2. **Stock clear:** separate Blake2b and Keccak configurations over the non-BN254
   field, full and compressed proof verification, and a sequential batch example.
   Add a parallel configuration of the same workload.
3. **Optional integrations:** R1CS without committed mode; committed mode with a
   caller-provided backend (a test-only mock establishes compilation/shape only);
   and a real BN254 commitment/transcript example with full consistency discharge
   through Jolt's existing BlindFold pipeline. Poseidon is tested separately and
   explicitly permits BN254 in its graph.

Use existing soundness tests where they already cover the behavior; do not add
copies of old production logic as permanent oracles. The small example's polynomial
evaluation is independent ground truth, not a second implementation of batching.

### Installation procedure

CI copies each fixture outside the Jolt workspace into a fresh temporary project
with its own manifest, lockfile, and Cargo configuration. Run from that directory,
with no inherited repository `.cargo/config.toml`, workspace lockfile, local
path overrides, or root patches. Resolve a reachable full commit SHA from the
implementation PR, record it in the generated dependency declarations, then run
consumer Clippy, `cargo nextest run --cargo-quiet`, and the example binary. Keep
the generated lockfile and graph as CI artifacts; repeat using `--locked`.
A fresh resolution is essential: pinning Jolt does not freeze branch-based
transitive dependencies. Library “installation” here means depending on and
building the library, not `cargo install`, which installs binaries.

Run both normal/build dependency inspection and feature inspection from that
consumer, using `cargo tree --edges normal,build` and `cargo tree -e features`.
Check reachable selected packages/features, not just all entries in a lockfile
or raw metadata package list; resolution can retain unused optional packages.
Isolate test-only backend dependencies from the fixture whose graph is measured.

### Patches and source identities

Cargo only honors patches at the consuming workspace root; the Jolt repository's
patch tables do not transfer with a Git dependency. See
[Cargo dependency overrides](https://doc.rust-lang.org/cargo/reference/overriding-dependencies.html).

The baseline root patches Arkworks 0.5 packages to the a16z fork. Direct workspace
Arkworks dependencies already use that Git source, while Light Poseidon and
other third-party packages can request registry copies. The baseline lock pins
the fork to `76bb3a4518928f1ff7f15875f940d614bb9845e6`; its manifest uses the moving
`dev/twist-shout` branch. Spongefish is separately pinned to
`d2d190b1329d35ac9577438d05aed4f17a57b9f9` and resolves registry Arkworks 0.6 codecs.
A 0.5 patch cannot establish 0.6 trait compatibility.

The minimal and byte-transcript clear fixtures must build without consumer
patches and without Arkworks in their selected build graph. For BN254/Poseidon
and native codec configurations, first test without patches; if still required,
deliver an exact, tested consumer-root patch recipe naming every required package
and matching source identity, with its own locked external test. Do not blindly
copy a `rev` patch alongside branch dependencies and assume Cargo unifies them.
Prefer pinning direct fork dependencies and the tested recipe coherently during
implementation if practical. Document any remaining moving-source requirement.

Do not copy Jolt's `[patch."https://github.com/a16z/jolt"]` local field override
into standalone examples. It serves the Akita workspace integration, which is
not in this sumcheck closure. If an optional integration actually needs a field
source override, test and document it separately. Publication, registry version
requirements on path dependencies, and publish ordering remain follow-up work.

## Acceptance matrix

Each standalone row is independently resolved and compiled, with actual
prove/verify coverage where applicable.

| Configuration | Required graph property |
| --- | --- |
| Default and defaults disabled, custom transcript | No crypto, openings, R1CS, claims, curves, stock sponges, Rayon, or sumcheck-requested OS entropy |
| Blake2b only; Keccak only | No Arkworks/BN254, unwanted stock sponge, commitments, or Rayon |
| `parallel` only and with each byte transcript | Parallel dependencies allowed; no curves or commitment machinery |
| `committed` with no concrete backend selected | Generic committed APIs compile without BN254, stock sponges, openings, or Rayon |
| `r1cs` only | No committed machinery, claims, RISC-V dependencies, curves, or Rayon |
| `r1cs,committed` | Both integrations compile without an implicit concrete backend |
| BN254 committed, sequential and parallel | Real backend works; Rayon appears only in the latter |
| Poseidon clear | BN254 allowed; no implicit committed mode or Rayon |
| All sumcheck features | Additive features compose; Jolt mode selection remains separately controlled |

Preserve existing negative tests for wrong claims, degrees, round counts,
compressed encodings, and final oracle evaluation. Add distinct boundary tests
for newly checked dimensions/unsupported fields. Gate existing backend-specific
tests and declare their needed dev features explicitly; do not hide broken
minimal compilation behind a blanket test-module gate.

Implementation validation also includes:

- Clippy for the changed crates in each supported minimal feature combination,
  plus `cargo clippy --all --features host -q --all-targets -- -D warnings` and
  the same command with `host,zk`; `cargo fmt -q`.
- Existing sumcheck, transcript, crypto, openings, R1CS, and BlindFold suites via
  `cargo nextest`, selecting their explicit feature requirements.
- `cargo nextest run -p jolt-prover-legacy muldiv --cargo-quiet --features host`
  and the same command with `host,zk`.
- Both modular acceptance suites: `cargo nextest run -p jolt-prover --features
  prover-fixtures --cargo-quiet` and the same with `prover-fixtures,zk`.
  Preserve clear byte parity, ZK acceptance/tamper rejection, advice, and committed
  program coverage. Use existing live-reference/golden checks for transcript
  continuity, not a permanent copy of the pre-refactor implementation.
- Rustdoc under minimal and documented feature configurations, with working
  links/examples and no backend-only names exposed in minimal docs. Record the
  tested Rust version (baseline toolchain is 1.95); establish a lower MSRV only
  with a separate successful check.

No arithmetic optimization is proposed. If implementation changes polynomial
algorithms or hot paths rather than only boundaries/types, profile and benchmark
those changes under the repository's performance policy.

## Implementation validation record

The branch was validated with the repository's Rust 1.95 toolchain:

- The isolated Solinas/custom-transcript consumer builds and runs with its own
  workspace and lockfile. Its normal/build graph contains no Arkworks, crypto,
  openings, R1CS, Spongefish, Rayon, or `getrandom` packages.
- A fresh project under `/tmp`, with no Jolt workspace or root patches, builds
  and runs from the local Git URL pinned to implementation commit
  `5a01f83ce8da90afc942be8af7256fcd5e627fd0`; its dependency graph satisfies
  the same absence checks.
- Sumcheck passes 25 minimal tests and 84 `committed,r1cs` tests. All documented
  feature combinations compile, and minimal/all-feature rustdoc builds succeed.
- Transcript passes 62 tests, crypto 133, openings 18, R1CS 74, BlindFold 76,
  Dory 47, and Akita 47. Strict Clippy passes for the changed crates across the
  minimal, all-feature, sequential BN254, and parallel configurations.
- The modular clear acceptance suite passes 21 tests, including every legacy
  byte-diff ratchet. The modular ZK suite's ordinary/committed muldiv, advice,
  tamper-rejection, and engine-twin cases pass. Two other ZK cases exhausted a
  test thread's stack only when the suite ran concurrently; each passed when
  rerun alone.
- Legacy `muldiv` passes all three selected tests in both `host` and `host,zk`.
- Both required workspace-wide Clippy commands reach the recursion guest and
  stop at its pre-existing `#[jolt::provable]` “invalid attribute” expansion;
  the resulting unused-import errors are consequences of that failed expansion.
  Package-level strict Clippy covers every crate changed by this refactor.

`cargo nextest run --cargo-quiet` selects zero tests under the workspace's bare
default configuration and exits with nextest's no-tests status. The targeted
suites above provide the applicable regression coverage.

## Zero-regression release gate

No finite test suite can prove the absence of every possible regression. For this
refactor, “no regression” is an implementation acceptance rule: there may be no
known functional, soundness, wire-compatibility, feature-composition, build,
performance, or dependency regression when the branch is handed off. A failure
is fixed in the branch; tests, fixtures, proof goldens, or thresholds are not
weakened to accept it.

Before implementation, record the exact base revision and capture the following
artifacts from a clean checkout. Compare the final branch against the same base,
toolchain, target, features, workload inputs, deterministic RNG seeds, and
preprocessing parameters:

1. Serialized clear and committed proofs for the existing deterministic fixtures,
   transcript states/challenges at every protocol boundary, verification results,
   and expected rejection classes for tampered proofs. Proof bytes and transcript
   checkpoints must match exactly. Where randomness is intentionally external,
   pin the seed; do not compare nondeterministic runs.
2. `cargo metadata`, `cargo tree --edges normal,build`, and `cargo tree -e features`
   for all rows in the acceptance matrix and the Jolt host, host+zk, modular clear,
   modular ZK, and Akita builds. Existing modes must retain every required feature;
   minimal rows must satisfy the stated absence rules.
3. Public API and rustdoc snapshots for sumcheck, transcript, poly, openings,
   crypto, and R1CS. Intentional source breaks listed in this spec receive migration
   notes; any unlisted removal, bound strengthening, or public path change blocks
   handoff.
4. Prover/verifier benchmark summaries and peak memory for representative
   Fibonacci and hash-chain workloads in standard, ZK, and relevant optimized
   modes. Boundary-only changes should produce identical proof work. Treat a
   statistically credible slowdown or memory increase as a regression; investigate
   noise with repeated paired runs rather than relaxing a threshold.

Run CI in separate jobs for feature combinations so Cargo feature unification in
an all-workspace invocation cannot make a minimal build pass accidentally. Include
`cargo hack` or an equivalent scripted powerset for the documented additive
features, excluding no combinations unless an explicit compile error and reason
are part of the public contract. Test Linux and macOS plus the repository's guest
targets; run a 32-bit compile check for checked size conversions where supported.

Add compile-fail coverage for backend-only names under minimal features and
compile-pass coverage for the generic custom field, transcript, round, commitment,
and R1CS surfaces. These tests assert public capability boundaries; they should
not assert Cargo implementation details that users cannot rely on.

Use the following disposition for any difference:

- A dependency removal in a minimal graph is expected only when required by this
  spec. A new dependency or enabled feature must have an identified owner and
  documented reason.
- Source changes listed in the migration inventory are accepted only with the
  stated behavior and wire invariants. Unlisted public API changes are regressions.
- Any proof-byte, transcript, verification, rejection, constraint, guest ABI, or
  arithmetic difference is a protocol regression and blocks the refactor.
- Any failure in clear byte parity, ZK acceptance/tamper rejection, advice,
  committed-program, Dory, Akita, or legacy compatibility blocks the refactor.
- Test flakiness is diagnosed and fixed; rerunning until green is not evidence.

The implementation PR must report the base and final revisions, commands, tested
feature matrix, external fixture lockfiles, graph diffs, proof/transcript parity
result, performance comparison, and any platform not exercised. Merge readiness
requires all required jobs to pass from clean state with no local patches or
uncommitted files.

## Implementation sequence and completion bar

1. Establish isolated consumer graph fixtures and capture baseline Jolt parity.
2. Split transcript capabilities and move the shared evaluation claim, preserving
   their encodings and public type identity.
3. Separate recorder results and committed exports; migrate all Jolt callers.
4. Repair crypto/R1CS optional boundaries and dependency defaults, restoring
   explicit feature selection on every edge whose inherited behavior changes.
5. Add the checked external boundaries, examples, README, and Git installation CI.
6. Run the matrix and Jolt regressions; record any remaining patch requirements
   and the exact tested revision before handoff.

Completion requires a real standalone clear prover/verifier, the asserted minimal
dependency graph, explicit working optional integrations, and unchanged Jolt
committed verification. A successful workspace build or a manifest with empty
defaults alone does not satisfy this specification.
