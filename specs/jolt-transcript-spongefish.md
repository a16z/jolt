# Spec: Port jolt-transcript to spongefish

| Field     | Value           |
| --------- | --------------- |
| Author(s) | @shreyas-londhe |
| Created   | 2026-04-17      |
| Status    | proposed        |
| PR        |                 |

## Summary

Jolt's current `Transcript` trait is a bespoke hash-based Fiat-Shamir construction with three backends (Blake2b, Keccak, Poseidon), each implementing custom label+length packing and challenge-extraction logic. This spec replaces that in-house construction with [spongefish](https://github.com/arkworks-rs/spongefish) — an audited duplex-sponge Fiat-Shamir library — and adopts its NARG-proof model so subprotocols (sumcheck, uni-skip, PCS) contribute to a single opaque byte string instead of directly appending typed messages onto a shared transcript struct. The result is a cleaner PCS boundary (dory contributes through a bridge into the same sponge), auditable domain-separation semantics, and alignment with the wider Arkworks spongefish ecosystem. This PR covers the Jolt side; a coordinated follow-up PR on `a16z/dory` makes dory consume `jolt-transcript` directly.

## Intent

### Goal

Replace Jolt's hand-rolled Fiat-Shamir transcript with spongefish's duplex-sponge construction across all three transcript variants, defining `ProverTranscript` and `VerifierTranscript` traits implemented directly on `spongefish::ProverState<Sponge>` / `spongefish::VerifierState<Sponge>` where `Sponge` is Cargo-feature-selected (`spongefish::Blake2b512`, `spongefish::Keccak`, or a new `PoseidonSponge: DuplexSpongeInterface` impl over `light-poseidon`). The new traits adopt spongefish's positional message style — per-call string labels (e.g., `b"opening_claim"`, `b"ram_K"`) are dropped. Domain separation lives only in the one-time `DomainSeparator` string used at transcript construction.

Key abstractions introduced:

- **`ProverTranscript`** (trait, in `crates/jolt-transcript`) — positional methods: `public_message<T>(&T)`, `prover_message<T>(&T)`, `verifier_message<T>() -> T`, plus the 128-bit-truncating challenge methods that preserve the current `challenge_*_optimized` performance profile, and `narg_string()`.
- **`VerifierTranscript`** (trait, in `crates/jolt-transcript`) — positional methods: `public_message<T>(&T)`, `receive_prover_message<T>() -> T`, `verifier_message<T>() -> T`, the same 128-bit-truncating challenge methods, and `check_eof()`.
- **`PoseidonSponge`** (type, in `crates/jolt-transcript`) — new `DuplexSpongeInterface` impl wrapping `light-poseidon`, so Poseidon plugs into spongefish like any other sponge.
- Trait impls live directly on `spongefish::ProverState<Sponge>` / `VerifierState<Sponge>` (orphan rule allows it); no wrapper structs.

### Invariants

1. **Prover/verifier sponge symmetry.** For every Jolt protocol path (standard and ZK), the prover and verifier absorb the same sequence of values into their sponges in the same order (positional — per-call labels no longer exist). This is the inductive property — challenge streams match on both sides, and protocol correctness and soundness carry over unchanged from the pre-port implementation.

2. **Functional equivalence for the Rust `JoltVerifier`.** Every Jolt proof produced by the off-chain prover continues to verify against the in-tree Rust `JoltVerifier`, across all three sponges (Blake2b / Keccak / Poseidon), in both `--features host` and `--features host,zk` modes.

### Non-Goals

1. **Consolidating dory onto `jolt-transcript` in this PR.** The end state is dory depending on `jolt-transcript` and sharing the same `ProverTranscript` / `VerifierTranscript` traits — but that change lives in a coordinated follow-up PR on `a16z/dory`. In the interim, the `JoltToDoryTranscript` bridge adapter (`jolt-core/src/poly/commitment/dory/wrappers.rs:336`) is updated to wrap the new spongefish-backed traits while dory keeps its current `DoryTranscript` trait. The bridge is removed once the follow-up lands.

2. **Updating `transpilable_verifier.rs`'s Solidity emission to match spongefish's Keccak absorb layout.** The Rust side migrates; the Solidity verifier update is a coordinated downstream follow-up.

3. **Updating `transpiler/` (gnark) and `zklean-extractor/` (Lean) to match spongefish's Poseidon absorb layout.** The Rust side migrates; the gnark and Lean verifier updates are coordinated downstream follow-ups.

4. **Cryptographic redesign of what gets absorbed.** This is a mechanical port of protocol semantics: the same values bind to the transcript in the same order. The only structural API change is that per-call string labels are dropped — in a deterministic protocol flow, positional order already provides domain separation, and production spongefish consumers (WhiR, sigma-rs) use purely positional calls. Labels survive only in the one-time `DomainSeparator` string used at transcript construction. No reconsideration of which values bind to the transcript or in what order.

5. **Guaranteeing stable spongefish NARG byte format across future spongefish releases.** Off-chain proofs are regenerated per release.

6. **Performance improvement beyond no-regression.** Bar: `muldiv` e2e no slower than current Blake2b.

## Evaluation

### Acceptance Criteria

- [ ] `crates/jolt-transcript` exposes `ProverTranscript` / `VerifierTranscript` traits implemented directly on `spongefish::ProverState<Sponge>` / `spongefish::VerifierState<Sponge>` (no wrapper structs).
- [ ] Three sponge backends selectable via Cargo feature: `transcript-blake2b` → `spongefish::Blake2b512`; `transcript-keccak` → `spongefish::Keccak`; `transcript-poseidon` → new `PoseidonSponge: DuplexSpongeInterface` over `light-poseidon`.
- [ ] `jolt-core/src/transcripts/` is deleted; `jolt-core` depends on `jolt-transcript`.
- [ ] `JoltToDoryTranscript` bridge (`jolt-core/src/poly/commitment/dory/wrappers.rs:336`) is updated to wrap the new traits and continues to satisfy dory's `DoryTranscript`.
- [ ] `muldiv` e2e passes under `--features host` for each of the three sponges.
- [ ] `muldiv` e2e passes under `--features host,zk` for each of the three sponges.
- [ ] `crates/jolt-transcript/tests/` and `crates/jolt-transcript/fuzz/` are deleted alongside the custom transcript implementation they were written to test. These tests exercised generic Fiat-Shamir properties (determinism, domain separation, challenge uniqueness, order sensitivity, clone independence) that are now owned by spongefish's own test suite; duplicating them in Jolt would be maintenance burden with no Jolt-specific correctness benefit.

- [ ] `crates/jolt-transcript/benches/transcript_ops.rs` is adapted to the new positional API and retained. Its role shifts from "micro-benchmarking our transcript implementation" to "regression gauge for spongefish operations under Jolt's usage pattern." Serves the Performance section's regression check.

- [ ] All `jolt-core` unit and integration tests remain green after mechanical API renames only: `append_bytes(&b)` → `public_message(&b)` / `prover_message(&b)`, `challenge_*` → `verifier_message::<T>()`, `Transcript::new(label)` → `DomainSeparator::new(...).instance(...).std_prover()`. No `jolt-core` test is deleted or `#[ignore]`d.

- [ ] `jolt-transcript` depends on the latest published `spongefish` release on crates.io at implementation time; the resolved version is captured in `Cargo.lock`.

### Testing Strategy

**Existing tests must keep passing:**

- `muldiv` e2e under `--features host` and `--features host,zk`, across all three Cargo feature sponges (`transcript-blake2b`, `transcript-keccak`, `transcript-poseidon`).
- All `jolt-core` unit and integration tests across the same matrix (with mechanical API renames as described in Acceptance Criteria).

**Tests removed alongside the code they test:**

- `crates/jolt-transcript/tests/` (all four files and the `common/` shared macro) — tested the custom transcript implementation's generic Fiat-Shamir properties; spongefish owns these tests for its own sponges upstream.
- `crates/jolt-transcript/fuzz/` — fuzzed the custom transcript's no-panic guarantee; spongefish owns this upstream.

**New tests:** None required. Correctness of the new code (the `PoseidonSponge: DuplexSpongeInterface` impl and the 128-bit-truncating decoder) is exercised by the `muldiv` e2e matrix — if either is wrong, e2e fails for the affected sponge or challenge path.

**Open question for maintainers:**

> How should CI treat `transpiler/go/e2e_test.go` and any Solidity integration tests that pin the current Poseidon / Keccak absorb byte layout? These will fail as a direct consequence of this port, since spongefish's domain separator and absorb semantics differ from the current hand-rolled layout. Options considered: (a) mark as `#[ignore]` until the downstream gnark / Solidity follow-ups land; (b) leave them failing, gating merge on coordinated downstream PR readiness; (c) maintainer's preference.

### Performance

No observable regression beyond benchmark noise on the `transcript_ops` micro-benchmark (`crates/jolt-transcript/benches/transcript_ops.rs`) and on the `muldiv` e2e wall-clock under `--features host` with `transcript-blake2b` (default).

Rationale: transcript operations are fundamentally hash calls (absorb, squeeze). The underlying hash implementation per sponge is unchanged — Blake2b is still Blake2b, Keccak is still Keccak, Poseidon is still Poseidon. Spongefish's construction adds a thin domain-separation framework over the sponge; it does not change the dominant cost.

## Design

### Architecture

**`crates/jolt-transcript`** (currently parked; this spec activates it):

- Adds `spongefish` and `light-poseidon` dependencies.
- Defines `ProverTranscript` and `VerifierTranscript` traits with positional method signatures matching spongefish-native shape: `public_message<T>(&T)` on both, `prover_message<T>(&T)` on prover / `receive_prover_message<T>() -> T` on verifier, `verifier_message<T>() -> T` for challenges on both. Challenge decoders implement `spongefish::Decoding<[H::U]>`; a custom 128-bit-truncating decoder preserves the performance profile of the current `challenge_*_optimized` family (63 hot-path callsites). `narg_string()` on prover, `check_eof()` on verifier.
- Implements these traits directly on `spongefish::ProverState<Sponge>` / `spongefish::VerifierState<Sponge>` — no wrapper structs.
- Provides a new `PoseidonSponge: DuplexSpongeInterface` impl wrapping `light-poseidon` so Poseidon plugs into spongefish.
- Retains the existing Cargo feature names `transcript-blake2b` / `transcript-keccak` / `transcript-poseidon`, each selecting the corresponding sponge type. `transcript-poseidon` continues to force-enable `challenge-254-bit`.
- Deletes the current `crates/jolt-transcript/src/{blake2b,keccak,poseidon,transcript}.rs` legacy implementations.

**`jolt-core`:**

- Deletes `src/transcripts/` (blake2b.rs, keccak.rs, poseidon.rs, transcript.rs, mod.rs).
- Adds `jolt-transcript.workspace = true` to `Cargo.toml`; forwards the three `transcript-*` features through.
- Updates ~148 transcript callsites. Generic bounds change from `<T: Transcript>` to `<T: ProverTranscript>` or `<T: VerifierTranscript>` as appropriate, AND each `append_*(label, &val)` / `challenge_*` call drops its label argument, becoming positional: `prover_message(&val)` / `public_message(&val)` / `verifier_message::<T>()`. Hot sites: `subprotocols/sumcheck.rs` (44 refs), `zkvm/prover.rs` (33), `zkvm/verifier.rs` (21), `zkvm/transpilable_verifier.rs` (30), `poly/commitment/hyperkzg.rs` (28), `subprotocols/univariate_skip.rs` (15), `subprotocols/blindfold/protocol.rs` (15), `poly/commitment/dory/commitment_scheme.rs` (12).
- `JoltToDoryTranscript` bridge (`poly/commitment/dory/wrappers.rs:336`) updated to wrap the new traits; dory interface unchanged.
- The shared preprocessing-binding code (currently a generic function at `zkvm/mod.rs:204-234` that binds preprocessing digest / memory layout / I/O) is split into two symmetric calls — one in `JoltProver::new()`, one in `JoltVerifier::new()`. Spongefish's `public_message` semantics mean both sides independently absorb the same values and their sponge states stay synchronized; duplicating the ~30 lines of binding code is cleaner than abstracting into a super-trait and keeps the symmetry eyeball-verifiable for Fiat-Shamir auditability.

**`JoltProof` structure** collapses to essentially:

```rust
pub struct JoltProof {
    narg: Vec<u8>,  // spongefish NARG byte string
    // plus any public inputs the verifier doesn't already know
}
```

Today's cfg-gated fields (`opening_claims: Claims<F>` in standard, `blindfold_proof: BlindFoldProof` in ZK) disappear from the struct — they become different prover-message sequences inside the NARG. The prover-side cfg gates remain (different code paths call different `prover_message` sequences), but the wire format unifies. Proof mode (standard vs ZK) is encoded in the spongefish domain separator at construction, not stored as a runtime field on the proof.

**`transpiler/`, `zklean-extractor/`:**

These depend on `jolt-core` features; they keep compiling because the Cargo feature names stay the same. Their emitted byte layouts change as a direct consequence of this port. Coordinating their downstream verifier updates is out of scope (Non-Goals 2 and 3).

### Alternatives Considered

1. **Keep Poseidon on the old `Transcript` trait (dual trait systems).** Rejected: spongefish's pluggable `DuplexSpongeInterface` means Poseidon can be a sponge like any other. No reason to maintain two parallel transcript worlds.

2. **`legacy-transcript-compat` feature flag that keeps old hand-rolled backends alive.** Rejected: contradicts "spongefish everywhere," adds permanent maintenance burden, defers downstream coordination indefinitely.

3. **Keep Keccak as a non-spongefish holdout for the EVM verifier.** Rejected: creates a forever-special-case in `transpilable_verifier.rs`. Better to coordinate the Solidity byte-layout update as a downstream follow-up (Non-Goal 2).

4. **Wrapper structs around `spongefish::ProverState` / `VerifierState`.** Rejected: the orphan rule lets us implement our local traits directly on spongefish's types. Wrappers would add ceremony without carrying extra state.

5. **Super-trait `TranscriptCommon` for shared prover/verifier code.** Rejected: spongefish's `public_message` semantics already handle symmetric absorption — both sides independently call `public_message(&value)` with identical inputs, and sponge states move in lockstep. The small amount of shared binding code (~30 lines, called once per proof) is cleaner duplicated at both callsites than abstracted; Fiat-Shamir symmetry is more eyeball-auditable when the two sides' code is visible side by side.

6. **Port in place in `jolt-core/src/transcripts/` without activating `crates/jolt-transcript`.** Rejected: the extracted crate exists per jolt#1365 and is the canonical future home once dory consumes it. Migrating code twice (in-place now, to the crate later) is strictly worse than doing it once.

7. **Keep per-call string labels by absorbing them as extra `public_message` calls on both sides.** Rejected: WhiR (`WizardOfMenlo/whir` — PCS used in production) and sigma-rs (`sigma-rs/sigma-proofs` — Sigma-protocols library) both use spongefish with purely positional calls; domain separation lives in the one-time protocol/session/instance string passed to `DomainSeparator`. In Jolt's deterministic protocol flow, positional order already provides the domain separation that per-call labels would redundantly provide, and absorbing labels adds ~one extra sponge permutation per transcript call on both prover and verifier for no soundness benefit.

## Documentation

No `book/` changes required. Existing conceptual descriptions of Fiat-Shamir (`book/src/how/blindfold.md:30-31,63,149`, `book/src/how/architecture/opening-proof.md:32`) are implementation-agnostic and remain accurate post-port. `CLAUDE.md`'s `## Architecture → transcripts/` subsection needs updating to reflect the new `crates/jolt-transcript` crate structure and spongefish-based implementation; that update lands with the implementation PR.

## References

- [arkworks-rs/spongefish](https://github.com/arkworks-rs/spongefish) — Fiat-Shamir duplex-sponge library.
- Closed dory PR #17 on `a16z/dory` — original spongefish integration attempt, redirected by the maintainer to `jolt-transcript`.
- `.claude/2026-04-17-jolt-transcript-spongefish-handoff.md` — handoff notes that triggered this spec.
- Related jolt crate-extraction work:
    - jolt#1362 — workspace scaffolding (merged).
    - jolt#1363 — `jolt-field` crate (merged).
    - jolt#1365 — `jolt-transcript` crate extraction (merged; this spec activates it).
    - jolt#1368 — `jolt-crypto` crate (merged).
    - jolt#1369 — `jolt-trace` crate (merged).
- jolt#1322 — original Poseidon transcript + gnark transpiler pipeline.
