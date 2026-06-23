# Akita Prover Integration Goal

Goal: Finish integrating Akita Jolt into the stripped modular Jolt stack.

## Starting Point

- Work on the local branch `feat/prover-akita-integration`, based on the stripped `jolt-prover-legacy` branch.
- Do not push to remote.
- The stable prover crate is now `crates/jolt-prover-legacy`; do not resurrect `jolt-core`.
- Use `feat/akita-protocol-integration` as the primary source for modular Akita/verifier work.
- Use `https://github.com/LayerZero-Research/jolt/tree/lz/akita` only as a structural reference for how Akita proving was wired in the old core-based stack.

## Main Objective

Bring in the Akita protocol/verifier changes in the most sensible modular form, then modify `jolt-prover-legacy` so it can produce Akita-protocol Jolt proofs accepted by `jolt-verifier`, while preserving the existing Dory proof path.

## Scope And Constraints

- Keep Dory as the default/current path and ensure existing Dory e2e tests still pass.
- Add Akita behind an explicit feature/config path.
- Do not reintroduce bridge/compat/core-style architecture unless absolutely required.
- Prefer native modular types from `jolt-verifier`, `jolt-openings`, `jolt-claims`, and `jolt-akita`.
- Avoid modifying verifier logic unless needed to finish the integration or fix a real protocol/interface bug.
- If an Akita/verifier/protocol bug blocks correctness, fix it locally and document why.
- Use efficient prover algorithms where available: streaming commitments, batched openings, sparse/one-hot handling, and avoid unnecessary full witness materialization.
- Commit locally at major milestones with clear messages. Do not push.

## Implementation Milestones

1. Import and stabilize the modular Akita support crates/layers:
   - `jolt-akita`
   - expanded `jolt-openings` packing/batch-opening support
   - necessary `jolt-claims` lattice/formula changes
   - necessary small support changes in field/poly/sumcheck/Dory/HyperKZG
   - keep workspace/dependency resolution clean.

2. Bring in the Akita verifier surface:
   - lattice commitment payloads
   - Akita packing/opening validation
   - stage 8 lattice/packed opening verification
   - precommitted/advice opening handling as needed
   - avoid old `jolt-core` or deleted compat paths.

3. Modify `jolt-prover-legacy` to produce verifier-native Akita proof artifacts:
   - preserve current Dory proof generation
   - add Akita proof generation through the modular opening/packing APIs
   - ensure prover output is directly accepted by `jolt-verifier`
   - avoid conversion wrappers unless there is no cleaner path.

4. Add e2e harness/tests:
   - Dory e2e still passes, including existing transparent and ZK checks where applicable.
   - Akita transparent e2e passes.
   - If Akita ZK is unsupported by the current Akita stack, explicitly document that and do not block on it unless the imported verifier path already supports it.

5. Validate:
   - `cargo fmt -q`
   - clippy in normal and ZK modes, matching repo instructions
   - targeted e2e tests for Dory and Akita
   - broader tests as needed for touched crates.

6. Benchmark after correctness:
   - Compare Dory vs Akita prover time on `sha2-chain` with guest input adjusted near `2^20` cycles.
   - Record methodology, command lines, commit hashes, feature flags, and results in a markdown report on this branch.
   - Expected result: Akita Jolt should be significantly faster than Dory Jolt.
   - If Akita is not faster, investigate prover bottlenecks and apply reasonable optimizations before reporting final results.

## Done Criteria

- Local commits exist for major milestones.
- Akita and Dory e2e tests pass.
- Existing Dory behavior is not regressed.
- Akita proofs are produced by `jolt-prover-legacy` and accepted by `jolt-verifier`.
- Benchmark report for `sha2-chain ~2^20 cycles` compares Akita vs Dory and documents any remaining caveats.
