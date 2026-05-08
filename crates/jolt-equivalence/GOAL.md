# Jolt-on-Bolt E2E Perf Goal

This task tracks the work required to make the generated Jolt-on-Bolt prover
PR-ready against `jolt-core` on the full standard `Fr` path.

## Readiness Target

The blocking PR-readiness target is:

```text
sha2-chain 2^20 e2e prover time: Bolt <= 1.20x jolt-core
```

This target is measured by the ignored release perf oracle:

```bash
source .bolt-dev-env
cargo nextest run -p jolt-equivalence --test bolt_perf --release \
  --run-ignored only bolt_sha2_chain_2_20_core_vs_bolt_perf_oracle \
  --no-fail-fast --no-capture
```

The `2^16` perf oracle is a faster smoke signal, but it is not sufficient for
PR readiness:

```bash
source .bolt-dev-env
cargo nextest run -p jolt-equivalence --test bolt_perf --release \
  --run-ignored only bolt_sha2_chain_2_16_core_vs_bolt_perf_oracle \
  --no-fail-fast --no-capture
```

## Current Baseline

Latest local release measurements on `refactor/crates`:

```text
sha2-chain 2^16:
  prove_ms: core=1923.850, bolt=2326.328, ratio=1.209x
  verify_ms: core=88.692, bolt=119.435, ratio=1.347x
  proof_bytes: core=80209, bolt=111198, ratio=1.386x
  peak_rss_mb: core=210, bolt=1026, ratio=4.886x

sha2-chain 2^20:
  setup_ms: core=2384.928, bolt=10279.676, ratio=4.310x
  prove_ms: core=9571.642, bolt=18719.842, ratio=1.956x
  verify_ms: core=104.248, bolt=135.351, ratio=1.298x
  proof_bytes: core=89041, bolt=121398, ratio=1.363x
  peak_rss_mb: core=1420, bolt=4954, ratio=3.489x
```

The current `2^20` prover path is therefore about `1.63x` slower than the
PR-readiness budget allows. The first milestone is to cut the Bolt prover from
about `18.7s` to about `11.5s` or lower on the same local machine.

## Required Semantic Gates

Performance work must preserve all equivalence and soundness gates:

```bash
source .bolt-dev-env
cargo nextest run -p jolt-equivalence --cargo-quiet --test-threads 1 --no-fail-fast
cargo nextest run -p jolt-witness --cargo-quiet
cargo nextest run -p jolt-kernels --cargo-quiet
cargo clippy -p jolt-equivalence -p jolt-witness -p jolt-kernels --all-targets -- -D warnings
```

The perf work is not acceptable if any of these regressions occur:

- Bolt proof artifacts stop being accepted by `jolt-core`.
- Bolt prover and verifier transcripts diverge.
- Generated verifier tamper checks become weaker or are removed.
- Internal generated/kernel artifact parity is weakened.
- `jolt-equivalence` gains proof-specific shortcuts that make Bolt look closer
  to core without moving real protocol work into the owning crates.

## Architecture Boundary

The implementation route is:

```text
Bolt IR/codegen -> jolt-kernels -> generated jolt-prover / jolt-verifier
```

Compiler changes are in scope and should be considered first-class work when
they make the generated prover/verifier simpler, faster, or closer to the core
algorithm. It is fine to adjust Bolt IR, stage schemas, generated metadata, or
Rust emission if that is the cleanest way to expose the right kernel shape or
avoid expensive generated glue.

Protocol semantics and performance-critical algorithms should live in one of:

- `crates/jolt-kernels`
- `crates/jolt-witness`
- `crates/bolt` IR/codegen/schema/pass/emit layers
- checked-in generated `crates/jolt-prover`
- checked-in generated `crates/jolt-verifier`

`crates/jolt-equivalence` is only the oracle harness. It may run core and Bolt,
compare public artifacts, check transcript parity, enforce tamper rejection,
and report perf metrics. It should not reconstruct prover semantics to hide
Bolt inefficiencies.

## Optimization Plan

1. Profile the `2^20` prover path with generated spans enabled.

   ```bash
   JOLT_BOLT_PERF_TRACE=1 cargo nextest run -p jolt-equivalence --test bolt_perf --release \
     --run-ignored only bolt_sha2_chain_2_20_core_vs_bolt_perf_oracle \
     --no-fail-fast --no-capture
   ```

2. Attack the largest Bolt prover deltas first, especially Stage 6 and any
   generated dense-polynomial paths that should mirror core sparse/specialized
   algorithms. Prefer compiler/codegen changes when the generated artifact
   shape is the reason a kernel cannot use the core-equivalent algorithm.

3. Keep improvements in reusable kernels, compiler IR/codegen, or generated
   code. Avoid one-off harness-side fixes.

4. After each meaningful optimization, rerun:

   ```bash
   cargo nextest run -p jolt-equivalence --cargo-quiet --test-threads 1 --no-fail-fast
   cargo nextest run -p jolt-equivalence --test bolt_perf --release \
     --run-ignored only bolt_sha2_chain_2_20_core_vs_bolt_perf_oracle \
     --no-fail-fast --no-capture
   ```

5. Once `2^20` prover ratio is consistently `<= 1.20x`, tighten the perf oracle
   threshold so CI enforces the target.

## Secondary Targets

These are not the first PR-readiness blocker, but should trend toward core:

```text
sha2-chain 2^20 setup_ms <= 1.20x
sha2-chain 2^20 verify_ms <= 1.20x
sha2-chain 2^20 proof_bytes <= 1.20x
sha2-chain 2^20 peak_rss_mb <= 1.20x
```

If one secondary metric remains above `1.20x`, document the reason and the
remaining owner before opening the PR stack. The prover `2^20` target is not
optional.

## Non-Goals

- Do not remove Bolt tamper coverage to improve runtime.
- Do not weaken core acceptance or transcript parity checks.
- Do not add perf-only behavior to `jolt-equivalence`.
- Do not optimize by diverging from full standard `Fr` semantics.
- Do not declare PR readiness from `2^16` numbers alone.
