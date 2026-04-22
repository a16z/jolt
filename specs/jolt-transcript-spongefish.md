# Spec: Port jolt-transcript to spongefish

| Field     | Value           |
| --------- | --------------- |
| Author(s) | @shreyas-londhe |
| Created   | 2026-04-17      |
| Status    | proposed        |
| PR        | 1455            |

## Summary

Port the internal implementation of the `crates/jolt-transcript` crate to use [spongefish](https://github.com/arkworks-rs/spongefish)'s duplex-sponge Fiat-Shamir construction. The crate currently holds a hand-rolled digest-based implementation; this PR replaces it with spongefish-backed `ProverTranscript` / `VerifierTranscript` traits and a new `PoseidonSponge` adapter over `light-poseidon`. The crate stays workspace-local and unused by `jolt-core`, dory, or the transpiler in this PR — integration into the jolt-core transcript call-path, the dory PCS bridge, and the gnark transpiler are explicitly deferred to follow-up PRs per maintainer guidance on #1455. This PR is step 1 of a staged rollout; scope is intentionally narrow so the trait surface and spongefish wiring can be reviewed in isolation. One downstream consumer is touched in-scope: `jolt-eval` gains a dependency on `jolt-transcript` so a new `transcript_prover_verifier_consistency` invariant can mechanically verify sponge symmetry across all three sponges.

## Intent

### Goal

Replace `crates/jolt-transcript`'s hand-rolled digest-based `Transcript` trait with spongefish-native `ProverTranscript` and `VerifierTranscript` traits, implemented directly on `spongefish::ProverState<Sponge>` / `spongefish::VerifierState<Sponge>`. Methods are positional, matching spongefish-native shape. Three sponges are feature-selectable within the crate: `spongefish::Blake2b512`, `spongefish::Keccak`, and a new `PoseidonSponge` adapter over `light-poseidon`. `jolt-eval` gains a new invariant that exercises the new trait surface across all three sponges. Crates outside `crates/jolt-transcript/` and `jolt-eval/` are not modified in this PR.

Key abstractions:

- **`ProverTranscript`** (new trait, replacing the current symmetric `Transcript` trait) — positional methods: `public_message<T>(&T)`, `prover_message<T>(&T)`, `verifier_message<T>() -> T`, `narg_string()`.
- **`VerifierTranscript`** (new trait) — positional methods: `public_message<T>(&T)`, `receive_prover_message<T>() -> T`, `verifier_message<T>() -> T`, `check_eof()`.
- Per-sponge challenge-width contract: Blake2b and Keccak expose both a 128-bit-truncating optimized decoder and a full-field decoder. Poseidon exposes only a full-field decoder (254-bit — the sponge's native field-element width). The 128-bit decoder is simply not defined on `PoseidonSponge`, so calling a `_optimized` method against it is a compile error.
- **`PoseidonSponge`** (new type) — adapter making `light-poseidon` usable as a spongefish sponge via `spongefish::DuplexSpongeInterface`. Circom-compatible BN254 parameters, matching today's `PoseidonTranscript` configuration.
- Trait impls live directly on `spongefish::ProverState<Sponge>` / `spongefish::VerifierState<Sponge>` (orphan rule allows it); no wrapper structs.

### Invariants

1. **Prover/verifier sponge symmetry (within the new crate).** Two `spongefish::ProverState<Sponge>` / `spongefish::VerifierState<Sponge>` instances constructed with identical `DomainSeparator` strings and absorbing the same sequence of values via `public_message` produce identical challenge streams via `verifier_message`. Holds for each of the three sponges. This invariant is encoded mechanically in `jolt-eval` (see Acceptance Criteria and Testing Strategy).

2. **No external behavior change, apart from a controlled jolt-eval addition.** `jolt-core/src/transcripts/`, the `JoltToDoryTranscript` bridge (`jolt-core/src/poly/commitment/dory/wrappers.rs:336`), the `transpiler/` and `zklean-extractor/` crates, and `JoltProof` are not modified. All their existing tests continue to pass unchanged. No downstream artifact (Solidity verifier, gnark verifier, Lean extraction) needs updating as a consequence of this PR. The only crate modified outside `crates/jolt-transcript/` is `jolt-eval`, which gains a new invariant module and a dependency on `jolt-transcript`.

### Non-Goals

1. **Integrating the new crate into `jolt-core`, dory, or the transpiler.** jolt-core's ~148 transcript callsites keep using `jolt-core/src/transcripts/`; the `JoltToDoryTranscript` bridge is untouched; `transpiler/` and `zklean-extractor/` are untouched. Three separate follow-up PRs will land these: (a) a16z/dory PR replacing dory's own `DoryTranscript` with a `crates/jolt-transcript` dependency; (b) a jolt-core migration PR replacing `jolt-core/src/transcripts/` with `jolt-transcript` imports and updating the 148 callsites; (c) a transpiler update PR regenerating the gnark verifier against spongefish's Poseidon absorb layout.

2. **Cryptographic redesign of what gets absorbed, at any layer.** Per-sponge challenge widths preserve today's semantics: Blake2b/Keccak's new traits expose both a 128-bit-optimized decoder and a full-field decoder; Poseidon's new traits expose only a full-field decoder. jolt-core's `challenge-254-bit` feature (gating `JoltField::Challenge` at `jolt-core/src/field/ark.rs:2`) is not modified and continues to behave exactly as today — since this PR does not connect jolt-core to the new crate, there is no cross-crate coupling on that feature.

3. **Guaranteeing stable spongefish NARG byte format across future spongefish releases.** `crates/jolt-transcript` pins the latest published `spongefish` at implementation time; future releases are picked up as normal dep updates.

4. **Performance improvement beyond no-regression.** Bar: `transcript_ops` micro-benchmark within 5% of the `main_run` CI baseline per `BenchmarkId`.

## Evaluation

### Acceptance Criteria

- [ ] `crates/jolt-transcript` replaces its current hand-rolled impl with spongefish-backed `ProverTranscript` / `VerifierTranscript` traits, implemented directly on `spongefish::ProverState<Sponge>` / `spongefish::VerifierState<Sponge>` (no wrapper structs).
- [ ] Three new Cargo features added to `crates/jolt-transcript/Cargo.toml` — `transcript-blake2b`, `transcript-keccak`, `transcript-poseidon` — selecting `spongefish::Blake2b512`, `spongefish::Keccak`, and the new `PoseidonSponge` adapter respectively. These features are new to the crate; the identically-named features in `jolt-core/Cargo.toml:45-48`, `jolt-sdk/Cargo.toml:44-46`, and `transpiler/Cargo.toml:26-28` are not modified and continue to drive `jolt-core/src/transcripts/` unchanged.
- [ ] Per-sponge challenge-width contract: Blake2b/Keccak expose both 128-bit-optimized and full-field decoders; Poseidon exposes full-field only (the 128-bit API is not defined on `PoseidonSponge`, so using it is a compile error).
- [ ] `jolt-core/src/transcripts/` is **not** modified; `jolt-core` does **not** depend on `jolt-transcript`; `JoltToDoryTranscript` is **not** modified; `transpiler/` and `zklean-extractor/` are **not** modified; `JoltProof` is **not** modified.
- [ ] `crates/jolt-transcript/tests/` and `crates/jolt-transcript/fuzz/` are deleted. Rationale in Testing Strategy.
- [ ] `crates/jolt-transcript/benches/transcript_ops.rs` is adapted to the new positional API and retained.
- [ ] A new jolt-eval invariant `transcript_prover_verifier_consistency` is added at `jolt-eval/src/invariant/transcript_symmetry.rs`, with one instantiation per sponge. The invariant covers the full trait surface — `public_message`, `prover_message`/`receive_prover_message`, and `verifier_message` — via an `Input` enum with variants `PublicBytes`, `PublicScalar`, `ProverBytes`, `ProverScalar`, `Challenge`. The `check` method runs the sequence on a `ProverTranscript` to build the NARG, constructs a `VerifierTranscript` against it, replays the sequence, and asserts both `receive_prover_message` round-trip equality and `verifier_message` challenge equality. The `#[invariant(Test, Fuzz, RedTeam)]` macro generates `#[test] fn seed_corpus()` and `#[test] fn random_inputs()` per instantiation; the generated fuzz targets compile; the seed corpus covers the empty sequence, single-message (each variant), 10-message mixed, and 1000-message mixed cases.
- [ ] `jolt-eval/Cargo.toml` gains a dependency on `jolt-transcript`. `jolt-eval/src/invariant/mod.rs` registers the new module. `jolt-eval/sync_targets.sh` is run so the fuzz target Cargo.toml entries are synchronized (per `jolt-eval/README.md` lines 229-239).
- [ ] `cargo nextest run -p jolt-eval` passes, exercising the new invariant's generated tests for all three sponge instantiations.
- [ ] `cargo nextest run -p jolt-core muldiv --features host` and `--features host,zk` continue to pass unchanged (workspace-build sanity check; jolt-core does not exercise the new code in this PR, so a per-sponge matrix is not meaningful here).
- [ ] `jolt-transcript` depends on the latest published `spongefish` release on crates.io at implementation time; version captured in `Cargo.lock`.

### Testing Strategy

**Existing tests that must keep passing, unchanged:**

- All `jolt-core` unit, integration, and e2e tests — no modifications, since jolt-core's transcript path is untouched. `muldiv` under `--features host` and `--features host,zk` included.
- All `transpiler/` and `zklean-extractor/` tests — no modifications, same reason.
- All dory-related tests via the existing `JoltToDoryTranscript` bridge — no modifications.

**Tests removed alongside the code they test:**

- `crates/jolt-transcript/tests/` (all four files and the shared `common/` macro) — tested generic Fiat-Shamir properties (determinism, domain separation, order sensitivity, clone behavior, prover/verifier consistency). These hold for any correct generic deterministic sponge regardless of parameter correctness — wrong width/rate/chunking in our adapter would produce "wrong but deterministic" output, which property tests still pass. They measure spongefish's properties (which spongefish tests upstream), not our adapter's parametrization.
- `crates/jolt-transcript/fuzz/` — fuzzed the no-panic guarantee, same reasoning.

**New tests added in this PR:**

- `jolt-eval`'s `transcript_prover_verifier_consistency` invariant, instantiated per sponge. The `#[invariant(Test, Fuzz, RedTeam)]` macro auto-generates two `#[test]` entries per instantiation (`seed_corpus`, `random_inputs`), plus a `libfuzzer_sys` fuzz target. Test entries run under `cargo nextest run -p jolt-eval`. The fuzz target builds in CI but is invoked ad-hoc via `cargo fuzz run -p jolt-eval <target>`.
- This invariant is the only in-tree correctness gauge for the new crate's spongefish wiring. Its differential-comparison shape (two instances absorbing the same messages must produce the same challenges) directly mechanizes Invariant #1. If an adapter-layer bug corrupts state but deterministically (e.g., wrong rate on `PoseidonSponge`), the invariant still passes — ultimate validation of absorb correctness happens in the follow-up jolt-core migration PR's `muldiv` e2e, which exercises the full protocol flow.

### Performance

**Regression budget: ≤5% wall-clock regression per `BenchmarkId` on `transcript_ops`**, measured against the `main_run` baseline artifact `criterion-baseline` stored by `.github/workflows/bench-crates.yml` on main pushes. The PR CI job at the same workflow downloads the baseline, runs the PR benchmarks with `--save-baseline pr_run`, and invokes `critcmp main_run pr_run --threshold 5` in the "Compare benchmarks" step to enforce the budget per individual `BenchmarkId` (e.g., `transcript_ops/absorb_scalar/Blake2b`), not against an aggregate.

Rationale: transcript operations are fundamentally hash calls (absorb, squeeze). The underlying hash primitive per sponge is unchanged — Blake2b, Keccak, and Poseidon all compute the same way they did pre-port. Spongefish's construction adds a thin domain-separation framework; it does not change the dominant cost. The 5% budget accommodates small differences in per-call sponge ratcheting and domain-separator overhead without requiring micro-optimization.

## Design

### Architecture

**`crates/jolt-transcript`** (currently contains a hand-rolled Fiat-Shamir implementation — `src/{lib,transcript,blake2b,keccak,poseidon,blanket,digest}.rs` plus populated `tests/`, `fuzz/`, `benches/` — that this PR replaces in-place):

- Adds `spongefish` and `light-poseidon` workspace dependencies.
- Defines `ProverTranscript` and `VerifierTranscript` traits with positional method signatures matching spongefish-native shape. Domain separation lives in the one-time `DomainSeparator` string used at transcript construction. Production spongefish consumers (WhiR, sigma-rs) use this same positional style. Note: the current `crates/jolt-transcript/src/transcript.rs` is already positional per-call; the positional-API choice is structural preparation for the follow-up jolt-core migration, where jolt-core's labeled per-call style (`append_scalar(b"opening_claim", &x)` at ~148 callsites in `jolt-core/src/transcripts/`) will transform into positional `prover_message(&x)` calls. Within this PR's scope, no callsite transformation is observable.
- Challenge decoders implement `spongefish::Decoding<[H::U]>`. Per-sponge contract:
    - **Blake2b** and **Keccak**: both a 128-bit-truncating optimized decoder and a full-field decoder.
    - **Poseidon**: full-field (254-bit) decoder only. The 128-bit optimized API is not defined on `spongefish::ProverState<PoseidonSponge>` — attempting to use it is a compile error.
- Implements these traits directly on `spongefish::ProverState<Sponge>` / `spongefish::VerifierState<Sponge>` via the orphan rule — no wrapper structs around the library types.
- Provides a new `PoseidonSponge` adapter making `light-poseidon` usable as a spongefish sponge (via `DuplexSpongeInterface`, or equivalently via a `Permutation` impl consumed through spongefish's built-in `DuplexSponge<P, W, R>` — implementer's choice). Circom-compatible BN254 parameters, matching today's `PoseidonTranscript`. Both paths produce absorb layouts that the follow-up gnark transpiler PR will validate against the emitted verifier; within this PR's scope there is no observable downstream effect from the choice, so the implementer should pick the path that most cleanly matches `light-poseidon`'s native API surface.
- Introduces three new Cargo features: `transcript-blake2b`, `transcript-keccak`, `transcript-poseidon`. These are new to the crate; jolt-core's and downstream crates' identically-named features remain in place and continue to drive `jolt-core/src/transcripts/` unchanged.
- Deletes the current `crates/jolt-transcript/src/{transcript,blake2b,keccak,poseidon,blanket,digest}.rs` legacy implementations. `lib.rs` is retained and re-exports the new trait surface.
- Deletes `crates/jolt-transcript/tests/` and `crates/jolt-transcript/fuzz/` per Testing Strategy.

**`jolt-eval`:**

- Adds `jolt-transcript` as a dependency in `jolt-eval/Cargo.toml`.
- Adds `jolt-eval/src/invariant/transcript_symmetry.rs` defining `TranscriptProverVerifierConsistency<Sponge>` with `#[invariant(Test, Fuzz, RedTeam)]` targets, one instantiation per sponge. The `Input` type is an enum covering the full trait surface: `PublicBytes(Vec<u8>)`, `PublicScalar(F)`, `ProverBytes(Vec<u8>)`, `ProverScalar(F)`, and `Challenge`. `check` runs the sequence on a `ProverTranscript` to produce a NARG byte string, constructs a `VerifierTranscript` against that NARG, replays the same sequence on the verifier, and asserts: (a) at each `ProverBytes`/`ProverScalar` op, the verifier's `receive_prover_message` returns the original value; (b) at each `Challenge` op, both sides' `verifier_message` outputs are identical.
- Registers the new module in `jolt-eval/src/invariant/mod.rs`.
- Runs `jolt-eval/sync_targets.sh` to synchronize the fuzz target Cargo.toml entries (per `jolt-eval/README.md:229-239`).

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

6. **Keep adapted versions of the existing property tests (determinism, domain separation, order-sensitivity, etc.) against `PoseidonSponge` to test the adapter.** Rejected: these tests verify properties that hold for any correct generic deterministic sponge, regardless of parameter correctness. Wrong width/rate/chunking in our `DuplexSpongeInterface` impl would produce "wrong but deterministic" output — property tests would still pass. They don't catch realistic adapter failure modes, so running them on our new code provides false confidence. The meaningful signal comes from the `jolt-eval` invariant (which tests the differential prover-verifier property the protocol relies on) plus the follow-up integration PR's `muldiv` e2e.

7. **No in-tree correctness gauge at all; defer all verification to the follow-up jolt-core migration's `muldiv` e2e.** Rejected (after reviewer feedback): `jolt-eval` is specifically designed to provide mechanically checkable invariants for exactly this class of property, and the cost of registering one invariant (one file + one Cargo.toml dep) is small compared to the value of closing the staging gap.

## Documentation

No `book/` changes required. `CLAUDE.md`'s `## Architecture → transcripts/` subsection is not updated in this PR — it describes jolt-core's transcript code, which is unchanged. It gets updated in the follow-up jolt-core migration PR.

## References

- [arkworks-rs/spongefish](https://github.com/arkworks-rs/spongefish) — Fiat-Shamir duplex-sponge library.
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
