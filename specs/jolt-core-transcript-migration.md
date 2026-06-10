# Spec: jolt-core Transcript Migration to Spongefish Split-Trait Surface

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | @shreyas-londhe                |
| Created     | 2026-06-03                     |
| Status      | implemented                    |
| PR          | 1586                           |

## Summary

jolt-core carries its own hand-rolled Fiat-Shamir transcript stack (`jolt-core/src/transcripts/`, ~1,370 LOC across `transcript.rs`, `blake2b.rs`, `keccak.rs`, `poseidon.rs`) that duplicates the functionality now provided by the `crates/jolt-transcript` spongefish crate. The `jolt-transcript` PR (`specs/jolt-transcript-spongefish.md`) deliberately left jolt-core untouched and shipped a temporary **legacy facade** (`jolt-transcript/src/legacy.rs`) so that the modular consumers (`jolt-sumcheck`, `jolt-openings`, `jolt-crypto`) and the newer `jolt-verifier` / `jolt-dory` crates could keep compiling against a jolt-core-shaped API. This spec covers follow-up PR (b): migrate jolt-core — and **all** remaining facade consumers (`jolt-verifier`, `jolt-dory`, `jolt-sumcheck`, `jolt-openings`, `jolt-crypto`) — onto jolt-transcript's spongefish-native **split-trait surface** (`ProverTranscript` / `VerifierTranscript` / `OptimizedChallenge`), delete jolt-core's duplicate transcript stack, and delete the legacy facade so that nothing in the workspace routes through anything but the spongefish-native split traits. The problem being solved is duplication and divergence: two transcript implementations that must be kept byte-consistent by hand, plus a facade explicitly marked for retirement.

## Intent

### Goal

Replace jolt-core's in-house `crate::transcripts::Transcript` and its `Blake2bTranscript` / `KeccakTranscript` / `PoseidonTranscript` implementations with jolt-transcript's spongefish-native split-trait surface across every transcript callsite, migrate **all** remaining facade consumers (`jolt-verifier`, `jolt-dory`, `jolt-sumcheck`, `jolt-openings`, `jolt-crypto`) off the legacy `jolt_transcript::Transcript` facade onto the same split-trait surface, implement `OptimizedChallenge` for `PoseidonSponge`, and delete both `jolt-core/src/transcripts/` (~1,370 LOC) and `jolt-transcript/src/legacy.rs` (~287 LOC) so that a single spongefish-backed transcript surface — the split traits in `jolt-transcript` — is the only Fiat-Shamir API anywhere in the workspace.

Key abstractions and boundaries:
- **Adopted API:** `jolt_transcript::ProverTranscript<H>` (`public_message` / `prover_message` / `verifier_message` / `narg_string`), `jolt_transcript::VerifierTranscript<H>` (`public_message` / `prover_message -> VerificationResult<T>` / `verifier_message` / `check_eof`), and `jolt_transcript::OptimizedChallenge` (`challenge_128 -> Fr`). These are positional (no per-call label argument); domain separation lives in the one-time `DomainSeparator` at construction, and message identity comes from each message type's `Encoding` / `NargSerialize` impls.
- **Prover/verifier split:** the prover side accumulates a NARG byte-string via `prover_message`; the verifier side consumes that same byte-string and terminates with `check_eof`. This replaces jolt-core's single symmetric `Transcript` used on both sides.
- **New impl:** `OptimizedChallenge for ProverState<PoseidonSponge, R>` and the corresponding `VerifierState`, defining a 128-bit challenge for the field-native Poseidon sponge so jolt-core's optimized-challenge callsites compile for all three sponges.
- **Generic parameter:** the `ProofTranscript: Transcript` bound (e.g. `jolt-core/src/zkvm/prover.rs:175`) is re-pointed from `crate::transcripts::Transcript` to the jolt-transcript split traits.

### Invariants

1. **Prover/verifier transcript consistency (clean break).** For every proof, the verifier replaying the prover's emitted NARG byte-string must derive identical challenges and `check_eof` must succeed. This is the correctness gate — **not** byte-equality with the pre-migration jolt-core transcript. Proof bytes are expected to change.
2. **Soundness preserved.** The existing `jolt-eval` `soundness` invariant (RedTeam: for any deterministic guest program + input, only one `(output, panic)` pair is accepted by the verifier) must continue to hold. The transcript swap must not introduce a Fiat-Shamir weakness (e.g. missing domain separation, length-extension on `prover_message`).
3. **Per-sponge challenge width, prover/verifier-deterministic.** Challenge derivation is dispatched on the *sponge type*: Blake2b512/Keccak squeeze 128-bit optimized challenges (`OptimizedChallenge::challenge_u128`); PoseidonSponge squeezes genuine full-field `Fr` challenges and deliberately does **not** support `challenge_128` (per maintainer decision on #1586: truncation is costly for recursion and defeats Poseidon's purpose, so `transcript-poseidon` forces `challenge-254-bit` and Poseidon's `challenge_u128` stays `unimplemented!`). For every sponge, prover and verifier must produce the same challenge from the same transcript position.
4. **Single transcript implementation.** After this PR, no workspace crate references `crate::transcripts::*` (deleted) or `jolt_transcript::{Transcript, AppendToTranscript}` (legacy facade, deleted).

`jolt-eval` framework:
- **Modify existing:** `jolt-eval/src/invariant/transcript_symmetry.rs` defines **three** structs — `TranscriptConsistencyBlake2bInvariant`, `...Keccak...`, `...Poseidon...` (lines 201/229/257), driven by a shared `Op` enum (line 20) whose variants the prover/verifier pair replays. The `Op` enum currently has **no optimized-challenge variant** (the Poseidon `Challenge` op squeezes a full-field `ArkFr` via `verifier_message`, `transcript_symmetry.rs:81-83`). Add a new shared `Op::OptimizedChallenge` variant that calls `challenge_128`, exercised by the Blake2b/Keccak structs; the Poseidon struct **filters it out** (Poseidon has no 128-bit challenge — maintainer decision on #1586; its `challenge_u128` is `unimplemented!`). This is a shared-enum change across the three invariants, not a one-line addition.
- **Add via `/new-invariant`:** a `jolt_core_transcript_roundtrip` invariant — for a small jolt-core proof artifact (or a representative sequence of `public_message` / `prover_message` / `verifier_message` / `challenge_128` calls mirroring jolt-core's preamble + one sumcheck round), the verifier-side replay of the prover's NARG string derives identical challenges and `check_eof` succeeds. This mechanizes Invariant #1 below the full `muldiv` e2e so a regression is caught by a fast `jolt-eval` test, not only by the end-to-end prover.

### Non-Goals

1. **Byte-identical Fiat-Shamir output.** This is an explicit clean break. The migrated transcript uses spongefish's duplex-sponge + NARG layout; serialized proofs produced before this PR will not verify after it, and that is acceptable. No backward-compatibility shim for old proof bytes.
2. **Downstream verifier regeneration.** The transpilable verifier (`jolt-core/src/zkvm/transpilable_verifier.rs`, consumed by the gnark transpiler in `transpiler/`) and the Lean extractor (`zklean-extractor/`) consume the transcript byte layout and will need regeneration — these are coordinated follow-up PRs, named as non-goals in the parent spec, and are out of scope here. (There is no Solidity verifier — maintainer correction on this spec's review.) As implemented, the `transpiler` crate is temporarily removed from the workspace and `transpilable_verifier` is gated behind a default-off `transpiler` feature until that regeneration lands.
3. **The a16z/dory transcript PR.** jolt-dory's bridge (`crates/jolt-dory/src/transcript.rs`) is migrated to the split-trait surface in this PR, but the separate `a16z/dory` PR that replaces dory's own `DoryTranscript` with a `crates/jolt-transcript` dependency remains a distinct follow-up.
4. **`JoltProof` structural changes** beyond the unavoidable consequence of the transcript swap (challenge values / proof bytes changing). No new proof fields, no reorganization of proof stages.
5. **Performance optimization of the sponge itself.** Spongefish's duplex-sponge performance is taken as-is; this PR does not tune the permutation or batching.
6. **Behavioral changes to `jolt-sumcheck` / `jolt-openings` / `jolt-crypto` / `jolt-verifier` / `jolt-dory`.** These crates ARE migrated off the facade onto the split-trait surface in this PR (see Architecture → "Facade-consumer migration"), because deleting `legacy.rs` requires it. The migration is an API/role-split transformation only — no change to each crate's protocol logic, proof structure, or test semantics beyond the transcript-call rewrites and the (clean-break) challenge values they now derive.

## Evaluation

### Acceptance Criteria

- [ ] `jolt-core/src/transcripts/` is deleted in its entirety (`mod.rs`, `transcript.rs`, `blake2b.rs`, `keccak.rs`, `poseidon.rs`); `jolt-core/Cargo.toml` gains a dependency on `jolt-transcript`.
- [ ] `jolt-transcript/src/legacy.rs` is deleted. All its re-exports are removed from `jolt-transcript/src/lib.rs`: the `legacy::{...}` re-export block (`lib.rs:29-31`, including `SpongeTranscript`, `MAX_LABEL_LEN`), the `pub mod domain { Label, LabelWithCount, U64Word }` re-export (`lib.rs:37-39`, consumed by jolt-dory), and the `Blake2bTranscript`/`KeccakTranscript`/`PoseidonTranscript = SpongeTranscript<...>` aliases (`lib.rs:48+`). The split-trait surface is the only public transcript API remaining; jolt-sumcheck/jolt-openings/jolt-crypto (bound on the facade's `Transcript: Default + Clone + Sync + Send + 'static`) are migrated to it as part of this deletion.
- [ ] No workspace crate references `crate::transcripts::*`, `jolt_transcript::{Transcript, AppendToTranscript, SpongeTranscript, Label, LabelWithCount, U64Word, MAX_LABEL_LEN}`, or `jolt_transcript::domain::*`. Verified by `grep`.
- [ ] The `ProofTranscript` generic bound on jolt-core's prover/verifier (`jolt-core/src/zkvm/prover.rs:175` and the verifier counterpart) resolves to the jolt-transcript split traits.
- [x] Poseidon challenge width per the maintainer decision: `OptimizedChallenge::challenge_u128` stays `unimplemented!()` for `PoseidonSponge` (impl present only so generic bounds resolve); jolt-core's per-sponge `FsChallenge` gives Poseidon genuine full-field challenges and keeps Blake2b/Keccak at 128-bit.
- [ ] All facade consumers — `jolt-verifier`, `jolt-dory`, `jolt-sumcheck`, `jolt-openings`, `jolt-crypto` — import only the split-trait surface; no `T: Transcript<Challenge = …>` bound and no `AppendToTranscript` impl remains in any of them.
- [ ] jolt-core's `transcript-blake2b`/`-keccak`/`-poseidon` Cargo features forward to jolt-transcript's; `challenge-254-bit` is retained (it gates `JoltField::Challenge`, not `transcripts/`).
- [ ] `cargo nextest run -p jolt-core muldiv --cargo-quiet --features host` passes.
- [ ] `cargo nextest run -p jolt-core muldiv --cargo-quiet --features host,zk` passes.
- [ ] `cargo nextest run -p jolt-verifier -p jolt-blindfold -p jolt-eval --cargo-quiet` passes (the model-crate and invariant suites).
- [ ] `cargo nextest run -p jolt-sumcheck -p jolt-openings -p jolt-crypto --cargo-quiet` passes.
- [ ] `cargo clippy --all --features host -q --all-targets -- -D warnings` and `cargo clippy --all --features host,zk -q --all-targets -- -D warnings` are clean.
- [ ] The extended `transcript_prover_verifier_consistency` invariant and the new `jolt_core_transcript_roundtrip` invariant pass under `cargo nextest run -p jolt-eval`.

### Testing Strategy

Must continue passing:
- `cargo nextest run -p jolt-core muldiv --cargo-quiet --features host` and `--features host,zk` — the primary correctness gate. Because the migration is a clean break, this is where end-to-end prover/verifier consistency under the new transcript is proven (full protocol flow, Blake2b/Dory).
- `cargo nextest run -p jolt-verifier -p jolt-blindfold -p jolt-eval` — the model-crate, BlindFold, and invariant suites that exercise the split-trait surface and the soundness invariant.
- `cargo nextest run -p jolt-sumcheck -p jolt-openings -p jolt-crypto` — the modular consumers, proving the facade deletion did not strand them.
- Existing jolt-core transcript property tests (determinism, append-order sensitivity, label sensitivity) are deleted along with `jolt-core/src/transcripts/`; their intent is subsumed by jolt-transcript's own test suite (`crates/jolt-transcript/tests/`) plus the `jolt-eval` consistency invariant.

New tests:
- `jolt-eval`: add an `Op::OptimizedChallenge` variant to the shared `Op` enum in `transcript_symmetry.rs:20`, exercised by the Blake2b/Keccak invariants and filtered out by the Poseidon invariant (Poseidon has no 128-bit challenge — maintainer decision on #1586), asserting `challenge_128` prover/verifier agreement on the byte sponges.
- `jolt-eval`: add `jolt_core_transcript_roundtrip` (`/new-invariant`) — prover NARG string replayed by verifier yields identical challenges + `check_eof` for a representative jolt-core-shaped call sequence, with `Test` and `Fuzz` targets.
- `jolt-transcript`: add an `OptimizedChallenge` Poseidon unit test in the existing `poseidon_tests.rs` (squeeze width and prover/verifier agreement) mirroring the Blake2b/Keccak cases.

Mode coverage: both `--features host` and `--features host,zk` are required for the `muldiv` gate, since the ZK path (`prove_zk` / BlindFold) drives the transcript differently from the standard path.

### Performance

Expectation: **no measurable prover/verifier time regression**. Transcript absorb/squeeze is a negligible fraction of prover wall-clock relative to MSM and sumcheck binding, and the optimized 128-bit challenge path is preserved for all sponges (including the newly-added Poseidon impl), so no hot-path challenge widens.

`jolt-eval` framework:
- **Moves existing objectives:** `lloc` decreases — deleting `jolt-core/src/transcripts/` (~1,370 LOC) and `jolt-transcript/src/legacy.rs` (~287 LOC) outweighs the callsite edits (~180 method calls across 27 files, mostly in-place method-name/positional changes, net small addition). Net expected: a measurable `lloc` reduction in `jolt-core/src/`. `cognitive_complexity_avg` and `halstead_bugs`: minor, expected flat-to-down. Verify via `cargo run -p jolt-eval --bin measure-objectives -- --no-bench`.
- **Performance objectives:** `prover_time_fibonacci_100`, `prover_time_sha2_chain_100`, `prover_time_secp256k1_ecdsa_verify` expected flat (within noise). "No regression" suffices; verify with the existing Criterion benches (`cargo bench -p jolt-eval --bench prover_time_fibonacci`). No new objective required.

## Design

### Architecture

Affected modules:
- **Delete:** `jolt-core/src/transcripts/{mod,transcript,blake2b,keccak,poseidon}.rs`; `jolt-transcript/src/legacy.rs` and its `lib.rs` re-exports.
- **jolt-core/Cargo.toml feature rewiring:** jolt-core already declares `transcript-blake2b` / `transcript-keccak` / `transcript-poseidon` (`Cargo.toml:46-48`), and `transcript-poseidon` additionally enables `challenge-254-bit` (`Cargo.toml:45`); these currently gate `jolt-core/src/transcripts/`. Add `jolt-transcript.workspace = true` and **rewire each jolt-core feature to forward to jolt-transcript's identically-named feature** (`transcript-blake2b = ["jolt-transcript/transcript-blake2b"]`, etc.). **Keep `challenge-254-bit` unconditionally** — it does NOT gate `transcripts/`; it selects `JoltField::Challenge` (`Mont254BitChallenge` vs `MontU128Challenge`) in `field/ark.rs:44-49` and `field/tracked_ark.rs`, which is untouched by this PR. Do not remove it. Open question the implementer must resolve: whether `transcript-poseidon` should still enable `challenge-254-bit`, given the new `OptimizedChallenge for PoseidonSponge` squeezes a full `Fr` and truncates to 128 bits regardless of the field's `Challenge` width — i.e. confirm there is no soundness/consistency coupling between the Poseidon transcript's 128-bit truncation and `JoltField::Challenge` before changing that coupling. The default arm (no transcript feature) must keep resolving to Blake2b, matching `zkvm/mod.rs:323-368`.
- **Transcript construction (session / instance) — as implemented (supersedes the earlier `EmptyInstance` pin):** jolt-core constructs transcripts via `prover_transcript(b"Jolt", instance, sponge)` / `verifier_transcript(b"Jolt", instance, sponge, narg)` with `instance = fiat_shamir_instance(...)` — a 32-byte `Blake2b(CanonicalSerialize(statement))` digest over the full public statement (preprocessing digest, program I/O, `ram_K`, trace length, entry address, rw/one-hot configs, dory layout), per @mmaker's #1455 mandate. This **replaces** the per-field `fiat_shamir_preamble` scatter-absorbs on the production path (the preamble function is retained only for the cfg-gated, currently disabled `transpilable_verifier`, and must keep the same field set/order as `fiat_shamir_instance`). Soundness-critical: the digest must be byte-identical on prover and verifier — both sides call the one `fiat_shamir_instance` helper, the verifier recomputing it from the proof's public tail.
- **Generic bound:** `JoltCpuProver` (`jolt-core/src/zkvm/prover.rs:175`) and the verifier's `ProofTranscript: Transcript` bound (`jolt-core/src/zkvm/verifier.rs:229`, `:259`) re-point to the jolt-transcript split traits. The prover side is bound by `ProverTranscript`; the verifier side by `VerifierTranscript`. The type aliases `RV64IMACProver` / `RV64IMACVerifier` live at **`jolt-core/src/zkvm/mod.rs:323-368`** as **four `cfg`-gated pairs** (Blake2b default arm, Blake2b explicit, Keccak, Poseidon); **all four pairs** must be updated to the spongefish-backed `ProverState` / `VerifierState` instantiations.
- **Public vs prover message rule.** The split-trait surface distinguishes `public_message` (values both prover and verifier already know — they are absorbed by both sides and do NOT enter the NARG string) from `prover_message` (values only the prover holds — absorbed by the prover, serialized into the NARG string, and read back by the verifier via `prover_message::<T>()`). Mapping jolt-core's symmetric `append_*` calls requires classifying each site: preamble/public-parameter appends (`fiat_shamir_preamble`, `zkvm/mod.rs:277-320`) become `public_message`; commitment / proof-scalar / sumcheck-round appends become `prover_message`.
- **Complete callsite mapping (~165–220 method calls across 27 files using `use crate::transcripts`; the exact figure depends on counting method — treat as an estimate, not a contract).** Every method on jolt-core's `Transcript` trait (`jolt-core/src/transcripts/transcript.rs`) must have a defined target:

  | jolt-core method | split-trait target |
  |---|---|
  | `append_bytes(label, b)`, `append_scalar(label, x)`, `append_scalars`, `append_commitment`, `append_commitments`, `append_serializable`, `append_commitments_serializable` | `public_message(&x)` if both sides know it, else `prover_message(&x)` (per the rule above). Label arg dropped; domain separation moves to the message type's `Encoding` impl and the construction-time `DomainSeparator`. |
  | `append_u64(label, n)`, `append_label(label)` | `public_message(&n)` / folded into domain separation. `raw_append_*` (the low-level primitives behind the labeled methods) are removed, not mapped. |
  | `challenge_scalar() -> F` | `verifier_message::<Fr>()` (full-width field challenge). |
  | `challenge_vector(n) -> Vec<F>` | `n ×` `verifier_message::<Fr>()`. |
  | `challenge_u128() -> u128` | `verifier_message::<u128>()`. |
  | `challenge_scalar_optimized() -> F::Challenge` | `challenge_128() -> Fr` (now available for all three sponges). |
  | `challenge_vector_optimized(n)` | `n ×` `challenge_128()`. |
  | `challenge_scalar_powers(n)` | one `verifier_message::<Fr>()` then power expansion in a jolt-core helper. |

  Mapped for completeness but with **zero production callsites** (only internal to the deleted `transcripts/` impls and their tests, so no migration work): `challenge_scalar_128_bits` (helper behind `challenge_scalar_optimized`) → `challenge_128()`; `challenge_scalar_powers_optimized(n)` → one `challenge_128()` + power expansion.

  Clustering: `zkvm/mod.rs:277-320` (`fiat_shamir_preamble`), `zkvm/prover.rs`, `zkvm/verifier.rs`, `subprotocols/sumcheck.rs`, `subprotocols/univariate_skip.rs`, `poly/opening_proof.rs`, `poly/commitment/dory/commitment_scheme.rs`, `subprotocols/blindfold/*`. The optimized-challenge family (`challenge_scalar_optimized` / `challenge_vector_optimized`) is ~50 of the production calls — confirm by grep during implementation rather than trusting this estimate.
- **`JoltToDoryTranscript`:** the jolt-core bridge (`jolt-core/src/poly/commitment/dory/wrappers.rs:415`) is rewired to wrap a `ProverTranscript` / `VerifierTranscript` instead of jolt-core's `Transcript`. The jolt-dory bridge (`crates/jolt-dory/src/transcript.rs`) is moved from the facade onto the split-trait surface.
- **Poseidon challenge width — DECIDED (maintainer, #1586 review):** Poseidon does **not** get a 128-bit optimized challenge. Quoting @moodlezoup on this spec's review: "I think we can leave challenge_128 `unimplemented!` for Poseidon. For recursion, the truncation itself is costly and somewhat defeats the purpose of using Poseidon in the first place. So [...] `transcript-poseidon` should still enable `challenge-254-bit`." As implemented: `OptimizedChallenge::challenge_u128` for `PoseidonSponge` is `unimplemented!()` (kept so generic-over-sponge `OptimizedChallenge` bounds resolve, e.g. the jolt-eval symmetry invariant); jolt-core's `FsChallenge` is implemented **per sponge type** — Blake2b/Keccak route through the 16-byte `u128` squeeze, while the `PoseidonSponge` impls squeeze a genuine full-field `Fr` (`verifier_message::<ark_bn254::Fr>`) for both `challenge_field` and `challenge_optimized` (wrapped into the `#[repr(transparent)]` `Mont254BitChallenge`; `transcript-poseidon` force-enables `challenge-254-bit` via a `compile_error!` coupling). The modular `jolt_transcript::FsChallenge` vocabulary is likewise implemented only for the byte sponges, so instantiating a modular verifier over a Poseidon-backed state is a compile error. An earlier draft of this section pinned a low-128-bit truncation rule for Poseidon; that design was rejected in review and is superseded by the above.
- **Facade-consumer migration (jolt-verifier, jolt-sumcheck, jolt-openings, jolt-crypto).** This is a SECOND, structurally distinct migration that the facade deletion forces. These crates do **not** use jolt-core's `crate::transcripts::Transcript`; they bind the **facade** trait `jolt_transcript::Transcript` (`legacy.rs:31`), which has a different shape: an associated `type Challenge: TranscriptChallenge`, plus `challenge()`, `challenge_scalar()`, `challenge_vector(n)`, `challenge_scalar_powers(n)`, `append()`, `append_labeled(label, v)`, `append_values(label, vs)`, `append_bytes`, `state()`, and a companion `AppendToTranscript` trait. There are ~53 `T: Transcript<Challenge = F>` bound sites (jolt-verifier `verifier.rs:37,273,343` + all `stages/*`; jolt-sumcheck; jolt-openings; jolt-crypto) plus ~166 method usages. The transformation mirrors jolt-core's (role-split + positional), via this mapping:

  | facade construct | split-trait target |
  |---|---|
  | bound `T: Transcript<Challenge = F>` | role-split: prover-side code → `P: ProverTranscript<H> (+ OptimizedChallenge where 128-bit challenges are used)`; verifier-side code → `V: VerifierTranscript<H>`. The `Challenge = F` associated-type projection disappears — challenges are `Fr` from `verifier_message::<Fr>()` / `challenge_128()`. |
  | `append(&x)`, `append_labeled(label, &x)` | `public_message(&x)` or `prover_message(&x)` (public-vs-prover rule above). Label arg dropped → message type's `Encoding` / domain separator. |
  | `append_values(label, &xs)` | length-prefixed: a `public_message`/`prover_message` of `xs.len()` then each element (or one message over the slice via its `Encoding`). |
  | `append_bytes(&b)` | `prover_message(&BytesMsg(b))` / `public_message`. |
  | `challenge()`, `challenge_scalar()` | `verifier_message::<Fr>()`. |
  | `challenge_vector(n)` | `n ×` `verifier_message::<Fr>()`. |
  | `challenge_scalar_powers(n)` | one `verifier_message::<Fr>()` + power expansion. |
  | `state() -> [u8;32]` | test/debug only; map to a spongefish state peek or delete the call (verify no production use). |
  | `impl AppendToTranscript for F / Commitment / …` | delete the `AppendToTranscript` impls; the absorbed types implement spongefish `Encoding` / `NargSerialize` instead, consumed directly by `public_message` / `prover_message`. |

  Because the facade is used symmetrically in both prove and verify paths of these crates, each callsite must be classified by role exactly as in jolt-core. jolt-verifier specifically also has the `state()`-style helpers and labeled appends introduced during the upstream merge, rewritten accordingly.
- **`jolt-dory` bridge:** `crates/jolt-dory/src/transcript.rs` (uses `jolt_transcript::domain::{Label, LabelWithCount}` + `{AppendToTranscript, Transcript}`) is moved onto the split-trait surface, same transformation.

Interaction sketch (prover → NARG → verifier):
```
prover:   DomainSeparator(label) -> ProverState
          public_message(pp) ; prover_message(commitment) ; c = challenge_128()
          ...                                     -> narg_string()  ─┐
                                                                     │ bytes
verifier: DomainSeparator(label) -> VerifierState(narg_string)  <───┘
          public_message(pp) ; prover_message::<C>()? ; c = challenge_128()
          ... ; check_eof()?
```

### Alternatives Considered

1. **Migrate jolt-core onto the legacy facade, not the split-trait surface.** Rejected: the facade is explicitly a temporary source-compatibility layer (`jolt-transcript/src/legacy.rs:5-6`, `lib.rs:12`) marked for retirement, and the parent spec states "future jolt-core work should use the split `ProverTranscript` / `VerifierTranscript` API and Spongefish NARG flow directly." Migrating to the facade would consolidate implementations but leave the positional/NARG end-state unreached and the facade undeletable.
2. **Staged: facade first, split-trait later.** Rejected for this spec: the split-trait surface requires a prover/verifier split that is awkward to introduce half-way (a transcript used symmetrically on both sides cannot partially emit a NARG string). Doing the facade step first would mean two large rewrites of the same ~180 callsites. A single migration to the end-state is fewer net changes.
3. **Preserve byte-identical Fiat-Shamir output via a compatibility shim.** Rejected: spongefish's duplex-sponge + NARG byte layout differs structurally from jolt-core's hash-chain transcript; a byte-for-byte shim would defeat the purpose of going spongefish-native and would be a large, fragile surface. The clean break is acceptable because downstream verifiers are coordinated follow-ups and the correctness gate is internal prover/verifier consistency.
4. **Leave Poseidon without `OptimizedChallenge` (Blake2b/Keccak only).** Rejected: jolt-core calls `challenge_scalar_optimized` and supports a Poseidon-backed prover via the `transcript-poseidon` feature (`jolt-core/src/zkvm/mod.rs`); omitting the impl would make the Poseidon configuration fail to compile or silently lose the optimized path. Implementing it keeps all three sponge configurations first-class.
5. **Big-bang vs prover-first/verifier-second split.** Single big-bang chosen: the split-trait prover/verifier separation is atomic by nature, and a prover-first PR would need a temporary bridge with FS bytes stabilizing only after the second PR. One reviewable (if large) diff is preferred.

## Documentation

- `CLAUDE.md` `## Architecture → jolt-core` and `transcripts/` references must be updated: the `transcripts/` submodule is deleted, and the description of the three type parameters (`ProofTranscript: Transcript`) must point at `jolt_transcript`'s split traits. The ZK-mode and opening-accumulator notes that mention transcript append behavior should be revised to the NARG flow.
- No `book/` user-facing changes are required: this is an internal refactor of the proving-system plumbing with no change to the guest-facing zkVM API. (If the book documents the proof/transcript byte format anywhere, that section needs a clean-break note; verify during implementation.)

## Execution

Suggested order:
1. Implement and unit-test `OptimizedChallenge for PoseidonSponge` in `jolt-transcript` first (smallest, self-contained, unblocks jolt-core's optimized callsites). Pin and document the 128-bit squeeze rule.
2. Add `jolt-transcript` as a jolt-core dependency; introduce the new `RV64IMACProver` / `RV64IMACVerifier` aliases over `ProverState`/`VerifierState` alongside the old ones to keep the tree compiling mid-migration.
3. Migrate jolt-core callsites cluster-by-cluster, starting from `fiat_shamir_preamble` (`zkvm/mod.rs`) then prover, verifier, sumcheck, univariate-skip, opening-proof, dory wrappers, blindfold. Keep prover and verifier edits paired so `muldiv` can be run as a continuous gate.
4. Rewire `JoltToDoryTranscript` (jolt-core) and migrate `jolt-dory` + `jolt-verifier` off the facade.
5. Delete `jolt-core/src/transcripts/` and `jolt-transcript/src/legacy.rs`; remove facade re-exports.
6. Add/extend the `jolt-eval` invariants; run both clippy modes and the full test matrix.

Fiat-Shamir caution: because absorb order defines challenge values, migrate prover and verifier for each protocol stage together and run `muldiv` (host + zk) frequently — a transposed `prover_message` / `verifier_message` order is silently wrong until the e2e fails. The new `jolt_core_transcript_roundtrip` invariant exists to surface such transpositions faster than the full prover.

## References

- Parent spec: `specs/jolt-transcript-spongefish.md` — Non-Goals §1 names this as follow-up PR (b).
- PR #1455 (a16z/jolt) — the jolt-transcript spongefish port; maintainer guidance (@moodlezoup, 2026-04-21) requesting a staged rollout with jolt-core integration deferred.
- spongefish: https://github.com/arkworks-rs/spongefish — duplex-sponge Fiat-Shamir, NARG flow.
- `jolt-transcript/src/{prover,verifier,setup,codec}.rs` — the split-trait surface this migration targets.
- `jolt-eval/README.md` — invariant/objective framework used for the consistency invariants and LLOC objective.
