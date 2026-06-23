# Akita Prover Integration Goal

## Goal Mode Prompt

Follow this file as the source of truth. Finish bringing Akita protocol and
verifier support into the stripped modular prover branch, then make
`crates/jolt-prover-legacy` produce verifier-native Akita proofs while preserving
the existing Dory path. Stay local: make milestone commits, but do not push.

## Mission

Finish bringing Akita protocol support into the stripped modular Jolt stack, then make
`crates/jolt-prover-legacy` produce Akita Jolt proofs accepted by `jolt-verifier`.
Preserve the existing Dory prover path throughout.

This branch should become the practical integration branch for Akita on top of the
post-`jolt-core` cleanup. The old `jolt-core`/compat/bridge architecture should not
come back.

## Starting Context

- Work on local branch `feat/prover-akita-integration`.
- This branch is based on the stripped prover branch where `jolt-core` became
  `crates/jolt-prover-legacy`.
- Do not push to remote.
- Commit locally at major milestones with clear commit messages.
- Use `feat/akita-protocol-integration` as the primary source for existing modular
  Akita/verifier work.
- Use `https://github.com/LayerZero-Research/jolt/tree/lz/akita` only as a structural
  reference for the old core-based Akita integration.
- Prefer this repository's modular stack over the old reference whenever they differ.

## Hard Requirements

- `jolt-prover-legacy` must keep producing valid Dory proofs.
- Akita must be added through an explicit feature/config path, not as a silent default
  replacement for Dory.
- Akita proofs must be verifier-native artifacts accepted by `jolt-verifier`.
- Avoid bridge, compat, and transitional wrapper layers unless there is no reasonable
  direct type path.
- Prefer direct use of modular types from `jolt-verifier`, `jolt-openings`,
  `jolt-claims`, `jolt-akita`, and related crates.
- Do not modify verifier logic unless needed for integration correctness or to fix a
  real protocol/interface bug.
- If a verifier/Akita/protocol bug blocks progress, fix it locally and document the
  reason in the commit message or benchmark/report notes.
- Use efficient prover algorithms where available. Avoid needless full witness
  materialization, repeated packing, repeated opening work, or obvious dense fallbacks
  when a sparse/batched/streaming path exists.

## Non-Goals

- Do not resurrect `jolt-core`.
- Do not preserve backwards compatibility with deleted core/compat APIs.
- Do not rename crates again unless the rename directly unblocks the Akita integration.
- Do not push branches or open/update PRs from goal mode.
- Do not block on Akita ZK if the imported modular Akita verifier path only supports
  transparent proofs. Document that limitation clearly and keep Dory ZK green.

## Execution Plan

1. Inspect the current branch state.
   - Verify the working tree and local commits.
   - Identify what Akita code has already landed in this branch.
   - Identify remaining gaps between `feat/akita-protocol-integration`, `lz/akita`,
     and this stripped modular stack.

2. Stabilize modular Akita support crates and verifier surface.
   - Bring in or reconcile required `jolt-akita`, `jolt-openings`, and `jolt-claims`
     changes.
   - Ensure lattice commitment payloads, packed opening validation, stage 8 lattice
     verification, and precommitted/advice openings compile cleanly.
   - Keep dependency and feature resolution clean.

3. Wire `jolt-prover-legacy` to produce Akita proof artifacts.
   - Keep the Dory proof path as-is except for necessary shared refactors.
   - Add a clear Akita prover entry point or feature-gated path.
   - Produce the commitment payloads, stage proofs, opening claims, packed validity
     proofs, and opening proofs that `jolt-verifier` expects.
   - Use native verifier/opening/claim structs directly where possible.

4. Add or update e2e coverage.
   - Dory transparent e2e must still pass.
   - Dory ZK e2e must still pass where it passed before.
   - Akita transparent e2e must pass.
   - Tampering tests should exercise prover-produced Akita proofs, not synthetic-only
     verifier fixtures.

5. Optimize only after correctness.
   - If Akita is correct but slow, inspect the hot path before changing algorithms.
   - Prefer known efficient paths from the Akita work or old `lz/akita` reference.
   - Keep optimizations local and measurable.

6. Benchmark after correctness.
   - Compare Dory vs Akita prover time using `sha2-chain` with input adjusted near
     `2^20` guest cycles.
   - Record commit hashes, command lines, feature flags, guest input, cycle count,
     hardware notes if available, and results.
   - Write results to a markdown report on this branch.
   - Expected outcome: Akita Jolt should be significantly faster than Dory Jolt.
   - If Akita is not significantly faster, investigate and fix reasonable prover
     bottlenecks before final reporting.

## Required Checks

Use `cargo nextest`, not `cargo test`.

Run focused checks first while iterating, then broaden before final completion:

```bash
cargo fmt -q
cargo check -p jolt-prover-legacy --features host -q
cargo check -p jolt-prover-legacy --features host,akita -q
cargo clippy -p jolt-prover-legacy --features host -q --all-targets -- -D warnings
cargo clippy -p jolt-prover-legacy --features host,akita -q --all-targets -- -D warnings
```

Before declaring the goal complete, also run the repo-level checks required by
`AGENTS.md` when feasible:

```bash
cargo clippy --all --features host -q --all-targets -- -D warnings
cargo clippy --all --features host,zk -q --all-targets -- -D warnings
cargo nextest run --cargo-quiet
cargo nextest run -p jolt-prover-legacy muldiv --cargo-quiet --features host
cargo nextest run -p jolt-prover-legacy muldiv --cargo-quiet --features host,zk
```

Add Akita-specific e2e and tampering checks once the Akita path is wired.

## Commit Strategy

Commit major milestones locally. Suggested commit boundaries:

1. Import/reconcile modular Akita verifier and support crate changes.
2. Add Akita prover data plumbing and proof assembly.
3. Add Akita e2e/tampering tests.
4. Add performance fixes, if needed.
5. Add benchmark report.

Keep commits reviewable. Avoid combining unrelated verifier fixes, prover rewrites,
and benchmark reporting into one commit.

## Done Criteria

- `jolt-prover-legacy` produces Dory proofs accepted by `jolt-verifier`.
- `jolt-prover-legacy` produces Akita proofs accepted by `jolt-verifier`.
- Dory transparent and ZK checks remain green.
- Akita transparent e2e is green.
- Akita tampering coverage uses prover-produced proofs.
- No `jolt-core`, compat, or bridge-style proof conversion path is reintroduced.
- Local milestone commits exist.
- A markdown benchmark report compares Dory vs Akita on `sha2-chain` near `2^20`
  guest cycles and documents any caveats.
