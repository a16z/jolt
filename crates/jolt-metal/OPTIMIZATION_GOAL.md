# Optimization Goal

## Target

**5x end-to-end speedup of Hybrid over CPU on all `sumcheck_e2e` scenarios.**

| Scenario | Current Speedup | Target Speedup |
|----------|----------------|----------------|
| toom_D4_P1 | ~2.0x | **5.0x** |
| toom_D4_P3 | ~2.1x | **5.0x** |
| toom_D8_P1 | ~1.8x | **5.0x** |
| eq_product | ~1.6x | **5.0x** |
| hamming | ~1.7x | **5.0x** |

## Rules

1. **No benchmark modifications.** `sumcheck_e2e.rs` is frozen. The benchmark measures
   real `SumcheckProver::prove()` latency through the full protocol loop — that's the
   number that matters. Changing NUM_VARS, HYBRID_THRESHOLD, sample_size, or measurement
   methodology is cheating.

2. **No correctness regressions.** All changes must pass:
   ```bash
   cargo nextest run -p jolt-metal --cargo-quiet
   cargo nextest run -p jolt-compute --cargo-quiet
   cargo nextest run -p jolt-zkvm --cargo-quiet
   ```

3. **Changes must be general.** Optimizations apply to the backend and kernel infrastructure,
   not special-cased for benchmark inputs. If it speeds up the benchmark but would hurt
   real proving workloads, it doesn't count.

4. **All intermediate results count.** Every experiment that moves any scenario closer to
   5x is valuable and gets logged in `OPTIMIZATION_LOG.md`, even if it only improves one
   scenario by 3%.

## Scope of Allowed Changes

Anything in these crates is fair game:
- `jolt-metal` — Metal backend, shaders, kernel compilation, dispatch
- `jolt-compute` — ComputeBackend trait, HybridBackend, buffer management
- `jolt-cpu` — CPU backend (making CPU faster also helps by raising the baseline
  that Hybrid competes against — wait, no, that hurts the ratio. CPU optimizations
  are allowed but only if they also improve the hybrid path.)
- `jolt-zkvm/src/evaluators/` — KernelEvaluator, witness construction
- `jolt-sumcheck` — SumcheckProver protocol loop
- `jolt-compiler` — CompositionFormula, expressions

NOT in scope (would change the problem, not solve it):
- Changing the field (BN254 → Goldilocks)
- Changing the polynomial commitment scheme
- Changing the sumcheck protocol structure
- Modifying benchmark parameters or measurement methodology

## Why 5x?

The current ~2x speedup means the GPU roughly doubles available Montgomery throughput.
To reach 5x, we need to either:
- Make GPU Montgomery mul 4x more efficient than CPU (occupancy + ILP improvements)
- Eliminate ~60% of non-kernel overhead (dispatch, transcript, reconstruction)
- Some combination of both

This is ambitious but not impossible. The theoretical bandwidth limit suggests 20-40x
is possible if Montgomery latency could be hidden. Realistic targets within the current
architecture are 3-5x. Hitting 5x likely requires at least one architectural insight
(pipelining, async dispatch, or a fundamentally better kernel strategy).

## Completion Criteria

The goal is met when all five scenarios show ≥5.0x speedup in three consecutive
benchmark runs on the same hardware. Record the final results in `OPTIMIZATION_LOG.md`.
