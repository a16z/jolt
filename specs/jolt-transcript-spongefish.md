# Spec: Port jolt-transcript to spongefish

| Field     | Value           |
| --------- | --------------- |
| Author(s) | @shreyas-londhe |
| Created   | 2026-04-17      |
| Status    | implemented     |
| PR        | 1455            |

## Summary

Port the internal implementation of the `crates/jolt-transcript` crate to use [spongefish](https://github.com/arkworks-rs/spongefish)'s duplex-sponge Fiat-Shamir construction. The crate currently holds a hand-rolled digest-based implementation; this PR adds spongefish-backed `ProverTranscript` / `VerifierTranscript` traits, keeps the existing `Transcript` / `AppendToTranscript` facade source-compatible for current modular-crate consumers, and adds a new `PoseidonSponge` adapter over `light-poseidon`. Per maintainer guidance on #1455, the staged rollout still leaves `jolt-core`, dory, the transpiler, `zklean-extractor`, and `JoltProof` untouched. However, `jolt-transcript` is already used by `jolt-sumcheck`, `jolt-openings`, and `jolt-crypto`; this PR must either preserve their existing API surface or migrate those direct consumers in the same PR. Default decision: preserve compatibility in this PR and validate those consumers explicitly. `jolt-eval` also gains a dependency on `jolt-transcript` so a new `transcript_prover_verifier_consistency` invariant can mechanically verify sponge symmetry across all three sponges.

## Intent

### Goal

Replace `crates/jolt-transcript`'s hand-rolled digest-based internals with spongefish-backed implementations. Add spongefish-native `ProverTranscript` and `VerifierTranscript` traits implemented directly on `spongefish::ProverState<Sponge, R>` / `spongefish::VerifierState<'_, Sponge>`, while retaining the current symmetric `Transcript` / `AppendToTranscript` facade for `jolt-sumcheck`, `jolt-openings`, and `jolt-crypto`. Methods are positional, matching spongefish-native shape. Three sponges are feature-selectable within the crate: `spongefish::instantiations::Blake2b512`, `spongefish::instantiations::Keccak`, and a new `PoseidonSponge` adapter over `light-poseidon`. `jolt-eval` gains a new invariant that exercises the new trait surface across all three sponges. `jolt-core`, dory, the transpiler, `zklean-extractor`, and `JoltProof` are not modified in this PR.

Key abstractions:

- **`ProverTranscript`** (new trait, additive to the current symmetric `Transcript` facade) — positional methods: `public_message<T: spongefish::Encoding<[H::U]> + ?Sized>(&mut self, &T)`, `prover_message<T: spongefish::Encoding<[H::U]> + spongefish::NargSerialize + ?Sized>(&mut self, &T)`, `verifier_message<T: spongefish::Decoding<[H::U]>>(&mut self) -> T`, and `narg_string(&self) -> &[u8]`.
- **`VerifierTranscript`** (new trait) — positional methods: `public_message<T: spongefish::Encoding<[H::U]> + ?Sized>(&mut self, &T)`, `prover_message<T: spongefish::Encoding<[H::U]> + spongefish::NargDeserialize>(&mut self) -> spongefish::VerificationResult<T>`, `verifier_message<T: spongefish::Decoding<[H::U]>>(&mut self) -> T`, and `check_eof(self) -> spongefish::VerificationResult<()>`.
- Per-sponge challenge-width contract: Blake2b and Keccak expose both a 128-bit-truncating optimized decoder and a full-field decoder (decoding to `ark_bn254::Fr`). Poseidon exposes only a full-field decoder (decoding to `ark_bn254::Fr`; 254-bit — the sponge's native field-element width). The 128-bit decoder is simply not defined on `PoseidonSponge`, so calling a `_optimized` method against it is a compile error.
- **`PoseidonSponge`** (new type) — adapter making `light-poseidon` usable as a spongefish sponge via `spongefish::DuplexSpongeInterface`. Circom-compatible BN254 parameters, matching Jolt's intended Poseidon transcript configuration.
- Trait impls live directly on `spongefish::ProverState<Sponge, R>` / `spongefish::VerifierState<'_, Sponge>` (orphan rule allows it); no wrapper structs.
- **Compatibility facade** — the existing `Transcript`, `AppendToTranscript`, `Blake2bTranscript`, `KeccakTranscript`, `PoseidonTranscript`, `Label`, `LabelWithCount`, and `U64Word` public names remain available for current direct consumers. Their internals may wrap spongefish state, but external source compatibility is preserved in this PR. The compat types must continue to satisfy `Default + Clone + Sync + Send + 'static`, since those bounds are baked into the `Transcript` supertrait that `jolt-sumcheck`, `jolt-openings`, and `jolt-crypto` use in their generic bounds.

### Invariants

1. **Prover/verifier sponge symmetry (within the new crate).** A `spongefish::ProverState<Sponge, R>` / `spongefish::VerifierState<'_, Sponge>` pair constructed with identical Spongefish domain setup — protocol identifier, session choice, and prefix-free instance encoding — and replaying the same ordered sequence of public messages, prover messages, and challenges produces identical challenge streams via `verifier_message` and identical verifier-side `prover_message` round trips. Holds for each of the three sponges. This invariant is encoded mechanically in `jolt-eval` (see Acceptance Criteria and Testing Strategy).

2. **No external behavior change for the staged-out systems.** `jolt-core/src/transcripts/`, the `JoltToDoryTranscript` bridge (`jolt-core/src/poly/commitment/dory/wrappers.rs:336`), the `transpiler/` and `zklean-extractor/` crates, and `JoltProof` are not modified. All their existing tests continue to pass unchanged. No downstream artifact (Solidity verifier, gnark verifier, Lean extraction) needs updating as a consequence of this PR. Direct modular consumers of `jolt-transcript` (`jolt-sumcheck`, `jolt-openings`, `jolt-crypto`) continue to compile and pass against the compatibility facade.

### Non-Goals

1. **Integrating the new crate into `jolt-core`, dory, or the transpiler.** jolt-core's ~148 transcript callsites keep using `jolt-core/src/transcripts/`; the `JoltToDoryTranscript` bridge is untouched; `transpiler/` and `zklean-extractor/` are untouched. Three separate follow-up PRs will land these: (a) a16z/dory PR replacing dory's own `DoryTranscript` with a `crates/jolt-transcript` dependency; (b) a jolt-core migration PR replacing `jolt-core/src/transcripts/` with `jolt-transcript` imports and updating the 148 callsites; (c) a transpiler update PR regenerating the gnark verifier against spongefish's Poseidon absorb layout.

2. **Cryptographic redesign of what gets absorbed, at any layer.** Per-sponge challenge-width APIs are explicit and staged: Blake2b/Keccak's new split traits expose both a 128-bit-optimized decoder and a full-field decoder; Poseidon's new split traits expose only a full-field decoder; the compatibility facade preserves the source-level modular-crate API. jolt-core's `challenge-254-bit` feature (gating `JoltField::Challenge` at `jolt-core/src/field/ark.rs:2`) is not modified and continues to behave exactly as today — since this PR does not connect jolt-core to the new crate, there is no cross-crate coupling on that feature.

3. **Guaranteeing stable spongefish NARG byte format across future spongefish releases.** `crates/jolt-transcript` targets the current published `spongefish` `0.7.x` API at implementation time; future spongefish version bumps must revalidate API signatures, NARG semantics, and compatibility-facade byte encodings.

4. **Performance improvement beyond no-regression.** Bar: `transcript_ops` micro-benchmark within 5% of the `main_run` CI baseline per `BenchmarkId`.

## Evaluation

### Acceptance Criteria

- [ ] `crates/jolt-transcript` replaces its current hand-rolled internals with spongefish-backed `ProverTranscript` / `VerifierTranscript` traits, implemented directly on `spongefish::ProverState<Sponge, R>` / `spongefish::VerifierState<'_, Sponge>` (no wrapper structs for the new split traits), while preserving the existing `Transcript` / `AppendToTranscript` facade for direct modular consumers.
- [ ] Three new Cargo features added to `crates/jolt-transcript/Cargo.toml` — `transcript-blake2b`, `transcript-keccak`, `transcript-poseidon` — selecting `spongefish::instantiations::Blake2b512`, `spongefish::instantiations::Keccak`, and the new `PoseidonSponge` adapter respectively. These features are new to the crate; the identically-named features in `jolt-core/Cargo.toml:45-48`, `jolt-sdk/Cargo.toml:44-46`, and `transpiler/Cargo.toml:26-28` are not modified and continue to drive `jolt-core/src/transcripts/` unchanged.
- [ ] Per-sponge challenge-width contract: Blake2b/Keccak expose both 128-bit-optimized and full-field decoders; Poseidon exposes full-field only (the 128-bit API is not defined on `PoseidonSponge`, so using it is a compile error).
- [ ] `jolt-core/src/transcripts/` is **not** modified; `jolt-core` does **not** depend on `jolt-transcript`; `JoltToDoryTranscript` is **not** modified; `transpiler/` and `zklean-extractor/` are **not** modified; `JoltProof` is **not** modified. `jolt-sumcheck`, `jolt-openings`, and `jolt-crypto` are either left source-compatible via the facade or migrated in this PR; the default path is source-compatible preservation.
- [ ] `crates/jolt-transcript/tests/` and `crates/jolt-transcript/fuzz/` are not blindly deleted. Generic tests that only duplicate spongefish's own determinism/order-sensitivity coverage may be removed, but local coverage remains or is added for the compatibility facade, local codecs, Poseidon adapter, NARG EOF rejection, and challenge-width behavior.
- [ ] `crates/jolt-transcript/benches/transcript_ops.rs` is adapted to the new positional API and retained.
- [ ] A new jolt-eval invariant `transcript_prover_verifier_consistency` is added at `jolt-eval/src/invariant/transcript_symmetry.rs`, with one instantiation per sponge and `type Setup = ()`. The invariant covers the full trait surface — `public_message`, prover-side and verifier-side `prover_message`, and `verifier_message` — via an `Input` enum with variants `PublicBytes`, `PublicScalar`, `ProverBytes`, `ProverScalar`, `Challenge`. The `check` method derives a valid Spongefish `DomainSeparator` from the exact operation sequence, runs the sequence on a `ProverTranscript` to build the NARG, constructs a `VerifierTranscript` against it, replays the sequence, and asserts both verifier-side `prover_message` round-trip equality and `verifier_message` challenge equality. A sequence may begin with `Challenge`; validity is determined by the generated `DomainSeparator`, not by a hard-coded "must absorb first" rule. The `#[invariant(Test, Fuzz, RedTeam)]` macro generates `#[test] fn seed_corpus()` and `#[test] fn random_inputs()` per instantiation; the generated fuzz targets compile; the seed corpus covers the empty sequence, single-message (each variant), 10-message mixed, and 1000-message mixed cases.
- [ ] `jolt-eval/Cargo.toml` gains a dependency on `jolt-transcript`. `jolt-eval/src/invariant/mod.rs` registers the new module. `jolt-eval/sync_targets.sh` is run so the fuzz target Cargo.toml entries are synchronized (per `jolt-eval/README.md` lines 229-239).
- [ ] `cargo nextest run -p jolt-eval --features jolt-transcript/transcript-blake2b,jolt-transcript/transcript-keccak,jolt-transcript/transcript-poseidon` passes, exercising the new invariant's generated tests for all three sponge instantiations.
- [ ] `cargo nextest run -p jolt-sumcheck -p jolt-openings -p jolt-crypto` passes, proving all current direct `jolt-transcript` consumers remain compatible.
- [ ] `cargo nextest run -p jolt-core muldiv --cargo-quiet --features host` and `cargo nextest run -p jolt-core muldiv --cargo-quiet --features host,zk` continue to pass unchanged (workspace-build sanity check; jolt-core does not exercise the new code in this PR, so a per-sponge matrix is not meaningful here).
- [ ] `jolt-transcript` depends on the current published `spongefish` `0.7.x` release on crates.io at implementation time; version captured in `Cargo.lock`. Do not enable Spongefish's arkworks codec features unless compatibility with Jolt's patched `ark-ff`/`ark-serialize` versions is verified.

### Testing Strategy

**Existing tests that must keep passing, unchanged:**

- All `jolt-core` unit, integration, and e2e tests — no modifications, since jolt-core's transcript path is untouched. `muldiv` under `--features host` and `--features host,zk` included.
- All `transpiler/` and `zklean-extractor/` tests — no modifications, same reason.
- All dory-related tests via the existing `JoltToDoryTranscript` bridge — no modifications.
- All current direct modular consumers of `jolt-transcript`: `cargo nextest run -p jolt-sumcheck -p jolt-openings -p jolt-crypto`.
- Primary jolt-core sanity checks: `cargo nextest run -p jolt-core muldiv --cargo-quiet --features host` and `cargo nextest run -p jolt-core muldiv --cargo-quiet --features host,zk`.

**Tests removed or rewritten alongside the code they test:**

- Generic property tests under `crates/jolt-transcript/tests/` may be removed when they only duplicate spongefish's own properties (determinism, order sensitivity, clone behavior). Tests that protect Jolt-local behavior — compatibility facade byte encodings, scalar endianness, label/count packing, Poseidon adapter semantics, and NARG EOF rejection — must be retained or replaced.
- `crates/jolt-transcript/fuzz/` may be reduced if it only fuzzes spongefish's no-panic behavior. Keep or add fuzz/invariant coverage for local codecs and the compatibility facade.

**New tests added in this PR:**

- `jolt-eval`'s `transcript_prover_verifier_consistency` invariant, instantiated per sponge. The `#[invariant(Test, Fuzz, RedTeam)]` macro auto-generates two `#[test]` entries per instantiation (`seed_corpus`, `random_inputs`), plus a `libfuzzer_sys` fuzz target. Test entries run under `cargo nextest run -p jolt-eval`. The fuzz target builds in CI but is invoked ad-hoc via `cargo fuzz run -p jolt-eval <target>`.
- This invariant is the only in-tree correctness gauge for the new crate's spongefish wiring. Its differential-comparison shape (a prover/verifier pair replaying the same operation sequence must round-trip prover messages and produce the same challenges) directly mechanizes Invariant #1. If an adapter-layer bug corrupts state but deterministically (e.g., wrong rate on `PoseidonSponge`), the invariant still passes — ultimate validation of absorb correctness happens in the follow-up jolt-core migration PR's `muldiv` e2e, which exercises the full protocol flow.

### Performance

**Regression budget: ≤5% wall-clock regression per `BenchmarkId` on `transcript_ops`**, measured against the `main_run` baseline artifact `criterion-baseline` stored by `.github/workflows/bench-crates.yml` on main pushes. The PR CI job at the same workflow downloads the baseline, runs the PR benchmarks with `--save-baseline pr_run`, and invokes `critcmp main_run pr_run --threshold 5` in the "Compare benchmarks" step to enforce the budget per individual `BenchmarkId` (e.g., `transcript_ops/absorb_scalar/Blake2b`), not against an aggregate.

Rationale: transcript operations are fundamentally hash calls (absorb, squeeze). The selected cryptographic primitives remain Blake2b, Keccak, and Circom-compatible BN254 Poseidon; spongefish adds a domain-separation and NARG framework around those primitives. The 5% budget accommodates small differences in per-call sponge ratcheting, adapter code, and domain-separator overhead without requiring micro-optimization.

## Design

### Architecture

**`crates/jolt-transcript`** (currently contains a hand-rolled Fiat-Shamir implementation — `src/{lib,transcript,blake2b,keccak,poseidon,blanket,digest}.rs` plus populated `tests/`, `fuzz/`, `benches/` — that this PR replaces in-place):

- Adds `spongefish` `0.7.x` and `light-poseidon` workspace dependencies. Enable only the spongefish hash features required for the selected backends; do not enable spongefish arkworks codec features unless they are proven compatible with Jolt's patched `ark-ff` / `ark-serialize` dependency graph.
- Defines `ProverTranscript` and `VerifierTranscript` traits with positional method signatures matching spongefish-native shape. Domain separation lives in the one-time `DomainSeparator` used at transcript construction. Production spongefish consumers (WhiR, sigma-rs) use this same positional style. Note: the current `crates/jolt-transcript/src/transcript.rs` is already positional per-call; the positional-API choice is structural preparation for the follow-up jolt-core migration, where jolt-core's labeled per-call style (`append_scalar(b"opening_claim", &x)` at ~148 callsites in `jolt-core/src/transcripts/`) will transform into positional `prover_message(&x)` calls. Within this PR's scope, no jolt-core callsite transformation is observable.
- Uses a concrete Spongefish domain setup before constructing prover/verifier states: `DomainSeparator::new(JOLT_TRANSCRIPT_PROTOCOL_ID)`, where the 64-byte protocol id is ASCII `a16z/jolt-transcript/v1` followed by zero padding; an explicit session decision (`session(...)` or `without_session()`); and a prefix-free instance encoding. The instance encoding is a local codec struct containing at least a version tag, sponge/backend id, challenge-width mode, and length-prefixed caller instance bytes. For the compatibility facade, `Transcript::new(label)` maps `label` into the session context with length-prefixing; the native split API requires callers to choose either an explicit session value or `without_session()` and to bind an instance value before calling `to_prover` / `to_verifier`.
- Provides local codec/newtype implementations for Jolt field/scalar/byte/message types instead of relying on spongefish's optional arkworks codecs. Jolt-local encodings must remain injective and prefix-free; the compatibility facade must preserve today's scalar endianness and label/count packing semantics for existing modular consumers.
- Challenge decoders implement `spongefish::Decoding<[H::U]>`. Per-sponge contract:
    - **Blake2b** and **Keccak**: both a 128-bit-truncating optimized decoder and a full-field decoder.
    - **Poseidon**: full-field decoder only, decoding to `ark_bn254::Fr` (254-bit). The 128-bit optimized API is not defined on `spongefish::ProverState<PoseidonSponge>` — attempting to use it is a compile error.
- Implements these traits directly on `spongefish::ProverState<Sponge, R>` / `spongefish::VerifierState<'_, Sponge>` via the orphan rule — no wrapper structs around the library types.
- Provides a new `PoseidonSponge` adapter making `light-poseidon` usable as a spongefish sponge (via `DuplexSpongeInterface`, or equivalently via a `Permutation` impl consumed through spongefish's built-in `DuplexSponge<P, W, R>` — implementer's choice). Use Circom-compatible BN254 parameters matching Jolt's intended Poseidon transcript configuration; preserve the compatibility facade for existing modular consumers. The follow-up gnark transpiler PR validates the exact absorb layout against the emitted verifier; within this PR's scope there is no observable downstream effect from the adapter-shape choice, so the implementer should pick the path that most cleanly matches `light-poseidon`'s native API surface.
- Introduces three new Cargo features: `transcript-blake2b`, `transcript-keccak`, `transcript-poseidon`. These are new to the crate; jolt-core's and downstream crates' identically-named features remain in place and continue to drive `jolt-core/src/transcripts/` unchanged.
- Rewrites the current `crates/jolt-transcript/src/{transcript,blake2b,keccak,poseidon,blanket,digest}.rs` legacy implementations in place. `lib.rs` is retained and re-exports both the new split trait surface and the existing compatibility facade.
- Removes only the tests/fuzz targets that no longer test Jolt-owned behavior; retains or replaces tests for local codec correctness, the compatibility facade, Poseidon adapter behavior, and NARG EOF rejection per Testing Strategy.

**`jolt-eval`:**

- Adds `jolt-transcript` as a dependency in `jolt-eval/Cargo.toml`.
- Adds `jolt-eval/src/invariant/transcript_symmetry.rs` defining three concrete named structs — `TranscriptConsistencyBlake2b`, `TranscriptConsistencyKeccak`, `TranscriptConsistencyPoseidon` — each with `#[invariant(Test, Fuzz, RedTeam)]`, `type Setup = ()`, and a shared `Input` enum. The three-concrete-struct pattern matches `SplitEqBindLowHighInvariant` / `SplitEqBindHighLowInvariant`; the `#[invariant]` macro is not designed for generic structs. The `Input` type is an enum covering the full trait surface: `PublicBytes(Vec<u8>)`, `PublicScalar(ark_bn254::Fr)`, `ProverBytes(Vec<u8>)`, `ProverScalar(ark_bn254::Fr)`, and `Challenge`. The scalar field is hardcoded to `ark_bn254::Fr`, matching the convention in `split_eq_bind.rs`. `check` derives a valid `DomainSeparator` from the exact generated operation sequence, runs that sequence on a `ProverTranscript` to produce a NARG byte string, constructs a `VerifierTranscript` against that NARG, replays the same sequence on the verifier, calls `check_eof()`, and asserts: (a) at each `ProverBytes`/`ProverScalar` op, the verifier-side `prover_message` returns the original value; (b) at each `Challenge` op, both sides' `verifier_message` outputs are identical.
- Registers the new module in `jolt-eval/src/invariant/mod.rs` and adds one `JoltInvariants` variant per sponge instantiation to the `JoltInvariants` dispatch enum in the same file (following the pattern of `SplitEqBindLowHigh`, `SplitEqBindHighLow`, etc.).
- Runs `jolt-eval/sync_targets.sh` to synchronize the fuzz target Cargo.toml entries (per `jolt-eval/README.md:229-239`).

**Direct modular consumers of `jolt-transcript`:**

- `jolt-sumcheck`, `jolt-openings`, and `jolt-crypto` currently depend on `jolt-transcript` directly, so this PR validates them explicitly.
- Default implementation path: preserve their existing imports, trait bounds, and callsites by keeping the `Transcript` / `AppendToTranscript` compatibility facade. If implementation proves this facade impractical, the PR must migrate these three crates in-scope and update this spec's test commands accordingly before merging.
- The compatibility facade is a source-compatibility layer only. It is not the future jolt-core migration API; future jolt-core work should use the split `ProverTranscript` / `VerifierTranscript` API and Spongefish NARG flow directly.

**Out of scope this PR (preserved exactly as today):**

- `jolt-core/src/transcripts/` — the hand-rolled `Transcript` trait and its three backends used by ~148 jolt-core callsites, the `JoltToDoryTranscript` bridge, the transpilable verifier, BlindFold, sumcheck, univariate skip, HyperKZG, etc.
- `JoltProof` — wire format unchanged.
- dory — keeps its own `DoryTranscript` trait; the `JoltToDoryTranscript` bridge wrapping jolt-core's transcript stays as-is.
- `transpiler/` and `zklean-extractor/` — their emitted verifier byte layouts are unaffected since they depend on jolt-core's transcript, which is untouched.

### Alternatives Considered

1. **Port `jolt-core/src/transcripts/` and `crates/jolt-transcript` together in one PR.** Rejected: maintainer's comment on #1455 (@moodlezoup, 2026-04-21) asked for a staged rollout — crate-only port first, jolt-core integration later — to keep each PR narrow-scope and reviewable. The earlier draft of this spec had the full migration; this revision narrows to just the crate.

2. **Keep Poseidon on the old `Transcript` trait (dual trait systems inside the same crate).** Rejected: spongefish's pluggable `DuplexSpongeInterface` means Poseidon can be a sponge like any other, and maintaining parallel transcript worlds in one crate would be a permanent maintenance burden.

3. **Keep per-call string labels by absorbing them as extra `public_message` calls on both sides.** Rejected: WhiR and sigma-rs (production spongefish consumers) both use purely positional calls; in a deterministic protocol flow, positional order already provides the domain separation that per-call labels would redundantly provide, and absorbing labels adds ~one extra sponge permutation per call for no soundness benefit.

4. **Wrapper structs around `spongefish::ProverState` / `VerifierState`.** Rejected: the orphan rule lets us impl our local traits directly on spongefish's types. Wrappers would add ceremony without carrying extra state.

5. **Super-trait `TranscriptCommon` for shared prover/verifier code.** Rejected: spongefish's `public_message` semantics already handle symmetric absorption — both sides independently call `public_message(&value)` with identical inputs, and sponge states move in lockstep. Shared binding code is cleaner duplicated at both callsites than abstracted; Fiat-Shamir symmetry is more eyeball-auditable when both sides' code is visible side by side.

6. **Keep all existing transcript property tests unchanged.** Rejected: tests that only duplicate spongefish's generic properties (determinism, order-sensitivity, no-panic behavior) provide little signal after the port. However, wholesale deletion is also rejected because this crate now owns compatibility encodings, local codecs, Poseidon adapter wiring, and NARG EOF checks. Retain or rewrite tests for those Jolt-owned behaviors.

7. **No in-tree correctness gauge at all; defer all verification to the follow-up jolt-core migration's `muldiv` e2e.** Rejected (after reviewer feedback): `jolt-eval` is specifically designed to provide mechanically checkable invariants for exactly this class of property, and the cost of registering one invariant (one file + one Cargo.toml dep) is small compared to the value of closing the staging gap.

## Documentation

No `book/` changes required. `CLAUDE.md`'s `## Architecture → transcripts/` subsection is not updated in this PR — it describes jolt-core's transcript code, which is unchanged. It gets updated in the follow-up jolt-core migration PR.

## References

- [arkworks-rs/spongefish](https://github.com/arkworks-rs/spongefish) and [spongefish crate docs](https://docs.rs/spongefish/latest/spongefish/) — Fiat-Shamir duplex-sponge library. This spec targets the current `0.7.x` API (`Encoding`, `NargSerialize`, `NargDeserialize`, `Decoding`, `VerificationResult`, `narg_string() -> &[u8]`, and `check_eof() -> VerificationResult<()>`).
- [spongefish `ProverState`](https://docs.rs/spongefish/latest/spongefish/struct.ProverState.html), [spongefish `VerifierState`](https://docs.rs/spongefish/latest/spongefish/struct.VerifierState.html), and [spongefish `DuplexSpongeInterface`](https://docs.rs/spongefish/latest/spongefish/trait.DuplexSpongeInterface.html) — API contracts used by this spec.
- [WizardOfMenlo/whir](https://github.com/WizardOfMenlo/whir) and [sigma-rs/sigma-proofs](https://github.com/sigma-rs/sigma-proofs) — production spongefish consumers using positional (label-less) calls.
- Closed dory PR #17 on `a16z/dory` — original spongefish integration attempt, redirected by the maintainer to `jolt-transcript`.
- `.claude/2026-04-17-jolt-transcript-spongefish-handoff.md` — handoff notes that triggered this spec.
- Maintainer's scope-shrink comment on #1455 by @moodlezoup (2026-04-21) — requested staged rollout, deferred jolt-core integration and transpiler updates to follow-up PRs.
- `jolt-eval/README.md` — invariant/objective framework and the `#[invariant(Test, Fuzz, RedTeam)]` macro used by the new `transcript_prover_verifier_consistency` module.
- `.github/workflows/bench-crates.yml` — Criterion baseline storage convention (`--save-baseline main_run` artifact + PR `--baseline main_run` comparison).
- Related jolt crate-extraction work:
    - jolt#1362 — workspace scaffolding (merged).
    - jolt#1363 — `jolt-field` crate (merged).
    - jolt#1365 — `jolt-transcript` crate extraction (merged; this spec ports its internal implementation).
    - jolt#1368 — `jolt-crypto` crate (merged).
    - jolt#1369 — `jolt-trace` crate (merged).
- jolt#1322 — original Poseidon transcript + gnark transpiler pipeline.
