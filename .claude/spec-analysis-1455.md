**Spec Analysis: Port jolt-transcript to spongefish (PR #1455)**

| Dimension | Score | Weight | Gap |
|-----------|-------|--------|-----|
| Goal Clarity | 0.88 | 0.35 | Trait method generic bounds & `narg_string()` return type not spelled out; `label`→`DomainSeparator` mapping implied |
| Constraint Clarity | 0.90 | 0.20 | Clear |
| Success Criteria | 0.87 | 0.30 | Invariant `Setup` associated type unspecified; sequence-validity (e.g. `Challenge` before any absorb) not stated |
| Context Clarity | 0.92 | 0.15 | Clear — all `file:line` anchors in the spec validate against HEAD (`wrappers.rs:336`, `Cargo.toml:45-48`/`44-46`/`26-28`, `field/ark.rs:2`, `README.md:229-239`) |
| **Ambiguity** | | | **11.3%** |

**Status: Approved** — the spec is clear enough for a one-shot implementation.

**Summary of what will be built:**
- Replace `crates/jolt-transcript`'s hand-rolled digest-based `Transcript` with spongefish-native `ProverTranscript` / `VerifierTranscript` traits, implemented directly on `spongefish::ProverState<Sponge>` / `spongefish::VerifierState<Sponge>` (no wrappers, via the orphan rule).
- Three feature-gated sponges: `transcript-blake2b` → `spongefish::Blake2b512`, `transcript-keccak` → `spongefish::Keccak`, `transcript-poseidon` → a new `PoseidonSponge` adapter over `light-poseidon` (Circom-compatible BN254).
- Per-sponge challenge-width contract preserved: Blake2b/Keccak expose both 128-bit-optimized and full-field decoders; Poseidon exposes full-field only (the 128-bit decoder is simply not defined on `PoseidonSponge`, so calling `_optimized` against it is a compile error).
- Delete `crates/jolt-transcript/tests/` and `crates/jolt-transcript/fuzz/` (rationale: they verify generic-sponge properties, not adapter parametrization). Adapt `benches/transcript_ops.rs` to the new positional API.
- Add `jolt-eval/src/invariant/transcript_symmetry.rs` (`TranscriptProverVerifierConsistency<Sponge>`) with `#[invariant(Test, Fuzz, RedTeam)]`, one instantiation per sponge. `Input` enum: `PublicBytes | PublicScalar | ProverBytes | ProverScalar | Challenge`. `check` runs the sequence on prover, replays on verifier against the NARG, asserts `receive_prover_message` round-trip equality and `verifier_message` equality. Seed corpus: empty, single-per-variant, 10-mixed, 1000-mixed.
- Add `jolt-transcript` dep to `jolt-eval/Cargo.toml`; register module in `invariant/mod.rs`; run `sync_targets.sh` to sync fuzz-target manifests.

**Key invariants:**
1. Prover/verifier sponge symmetry within the new crate (mechanized by the new invariant across all three sponges).
2. No external behavior change: `jolt-core/src/transcripts/`, `JoltToDoryTranscript`, `transpiler/`, `zklean-extractor/`, and `JoltProof` remain byte-identical; the three identically-named `transcript-*` features in `jolt-core`/`jolt-sdk`/`transpiler` continue to drive the old `jolt-core/src/transcripts/` unchanged.

**Critical evaluation criteria:**
- `cargo nextest run -p jolt-eval` (exercises the new invariant for all three sponge instantiations).
- `cargo nextest run -p jolt-core muldiv --features host` and `--features host,zk` continue to pass (workspace sanity only — jolt-core does not reach the new crate in this PR).
- `transcript_ops` Criterion benchmark stays within 5% of `main_run` baseline per `BenchmarkId` (enforced via `critcmp main_run pr_run --threshold 5` in `.github/workflows/bench-crates.yml`).

**Validation of spec anchors against HEAD:**
- `crates/jolt-transcript/src/{transcript,digest,blake2b,keccak,poseidon,blanket}.rs` present and hand-rolled digest-based ✓
- `crates/jolt-transcript/tests/` (4 files) and `fuzz/` populated ✓
- `crates/jolt-transcript/benches/transcript_ops.rs` present with `BenchmarkId::new("Blake2b"|"Keccak"|"Poseidon", ...)` groups ✓
- `jolt-eval/src/invariant/split_eq_bind.rs` serves as close structural model (same `#[invariant(Test, Fuzz, RedTeam)]` pattern, `type Setup = ()`, `type Input = ...`, `check(&self, _setup, input)`) ✓
- `jolt-eval/Cargo.toml` does **not** currently depend on `jolt-transcript` ✓
- No existing `transcript_symmetry.rs` or `TranscriptProverVerifierConsistency` ✓
- `jolt-core/src/poly/commitment/dory/wrappers.rs:336` hosts `JoltToDoryTranscript` ✓
- Features at `jolt-core/Cargo.toml:45-48`, `jolt-sdk/Cargo.toml:44-46`, `transpiler/Cargo.toml:26-28` ✓
- `challenge-254-bit` gate at `jolt-core/src/field/ark.rs:2` ✓
- `jolt-eval/README.md:229-239` describes `sync_targets.sh` ✓
- `.github/workflows/bench-crates.yml` uses Criterion `--save-baseline` and `critcmp` ✓
- `spongefish` is **not** yet a workspace dependency; `light-poseidon` already is ✓ (implementer needs to add `spongefish` to root `Cargo.toml`)

**Minor, non-blocking notes for the implementer** (judgement calls, not ambiguities that would block a one-shot impl):
- Generic bounds on `public_message<T>` / `prover_message<T>` / `verifier_message<T>` are not spelled out — follow spongefish's native `CommonUnitToBytes`/`Decoding` trait bounds (the spec names `spongefish::Decoding<[H::U]>` for challenge decoders).
- `narg_string()` return shape not specified — use spongefish's native `&[u8]` return.
- The invariant's `Setup` associated type is not named — default `type Setup = ()` matches `SplitEqBindLowHighInvariant`.
- Ordering constraints on the `Input` sequence (e.g., starting with `Challenge`) are not stated — spongefish's `DomainSeparator` defines the alphabet; the `check` method should construct both transcripts from the same `DomainSeparator` derived from the sequence.

**Next step:** Run `/implement-spec` to implement this spec.
