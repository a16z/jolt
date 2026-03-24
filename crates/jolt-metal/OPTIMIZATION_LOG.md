# Optimization Log

Tracks experiments, benchmark results, and baselines for the Metal hybrid backend.

- **Goal**: 5x e2e speedup on all `sumcheck_e2e` scenarios — see [`OPTIMIZATION_GOAL.md`](OPTIMIZATION_GOAL.md)
- **Analysis & playbook**: [`OPTIMIZATION_FRAMEWORK.md`](OPTIMIZATION_FRAMEWORK.md)

## Hardware

Record your hardware before running baselines:
```bash
system_profiler SPHardwareDataType | grep -E "Chip|Memory|Cores"
```

## Baselines

### How to capture

```bash
git rev-parse --short HEAD  # record commit
cargo bench -p jolt-metal --bench sumcheck_e2e -q 2>&1 | tee baselines/e2e_$(date +%Y%m%d).txt
cargo bench -p jolt-metal --bench metal_vs_cpu -q 2>&1 | tee baselines/isolated_$(date +%Y%m%d).txt
cargo bench -p jolt-metal --bench threshold -q 2>&1 | tee baselines/threshold_$(date +%Y%m%d).txt
```

### E2E Sumcheck (sumcheck_e2e, NUM_VARS=20, HYBRID_THRESHOLD=2^12)

| Date | Commit | Hardware | Scenario | CPU (ms) | Hybrid (ms) | Speedup |
|------|--------|----------|----------|----------|-------------|---------|
| 2026-03-24 | 29100c2c2 | M1 Pro 8c/16GB | toom_D4_P1 | 101.4 | 60.2 | 1.68x |
| 2026-03-24 | 29100c2c2 | M1 Pro 8c/16GB | toom_D4_P3 | 263.1 | 132.0 | 1.99x |
| 2026-03-24 | 29100c2c2 | M1 Pro 8c/16GB | toom_D8_P1 | 351.5 | 153.8 | 2.29x |
| 2026-03-24 | 29100c2c2 | M1 Pro 8c/16GB | eq_product | 49.5 | 26.7 | 1.85x |
| 2026-03-24 | 29100c2c2 | M1 Pro 8c/16GB | hamming | 66.2 | 41.0 | 1.61x |

### Isolated Kernels (metal_vs_cpu)

| Date | Commit | Hardware | Kernel | Size | Metal (ms) | CPU (ms) | Speedup |
|------|--------|----------|--------|------|------------|----------|---------|
| | | | pairwise_reduce D4 | 2^14 | | | |
| | | | pairwise_reduce D4 | 2^20 | | | |
| | | | pairwise_reduce D8 | 2^14 | | | |
| | | | pairwise_reduce D8 | 2^20 | | | |
| | | | interpolate | 2^14 | | | |
| | | | interpolate | 2^20 | | | |
| | | | sumcheck_round D4 | 2^20/4r | | | |
| | | | sumcheck_round D8 | 2^20/4r | | | |
| | | | sumcheck_round_h2l D4 | 2^20/4r | | | |
| | | | sumcheck_round_h2l D8 | 2^20/4r | | | |

### Threshold Crossover Points

| Date | Commit | Hardware | Kernel | Metal > CPU at | Optimal switchover |
|------|--------|----------|--------|----------------|--------------------|
| | | | pairwise_reduce D4 | | |
| | | | pairwise_reduce D8 | | |
| | | | interpolate | | |

---

## Experiments

### Template

```
### EXP-NNN: <title>
**Date**: YYYY-MM-DD
**Branch**: opt/<name>
**Commit**: <hash>
**Hardware**: <chip>
**Framework ref**: O<N> from OPTIMIZATION_FRAMEWORK.md

**Hypothesis**: <what bottleneck, expected gain, why>

**Changes**: <files modified, what was changed>

**Results**:
| Benchmark | Before | After | Delta |
|-----------|--------|-------|-------|

**Analysis**: <did it match hypothesis? why/why not?>

**Decision**: MERGED / DISCARDED
**Reason**: <why>
```

---

### EXP-001: Instrument and profile (O0c)
**Date**: 2026-03-24
**Branch**: refactor/crates
**Commit**: ba2779c4b
**Hardware**: M1 Pro 8c/16GB
**Framework ref**: O0c

**Hypothesis**: Profile hybrid toom_D8_P1 to identify the real bottleneck distribution across GPU reduce, GPU bind/interpolate, CPU rounds, and dispatch overhead.

**Changes**: Added `#[tracing::instrument]` to 5 Metal dispatch methods (device.rs), per-round `tracing::debug_span` to SumcheckProver loop (prover.rs), reduce/reconstruct spans to KernelEvaluator (kernel.rs), env-gated Chrome trace subscriber to sumcheck_e2e.rs.

**Results** (median of 25 hybrid runs, toom_D8_P1, NUM_VARS=20):

| Component | Time (ms) | % of prove |
|-----------|-----------|------------|
| **fused_interpolate_reduce (8 rounds)** | **84.7** | **56.8%** |
| **MetalBackend::pairwise_reduce (round 0)** | **57.3** | **38.5%** |
| reconstruct (Toom-Cook interp) | 3.7 | 2.5% |
| CpuBackend::pairwise_reduce (rounds 8-19) | 1.5 | 1.0% |
| transcript | 0.06 | 0.0% |
| **Total hybrid prove** | **149.0** | **100%** |
| Total CPU prove (median) | 263.1 | — |
| **Measured speedup** | **1.77x** | — |

Per-round breakdown (hybrid):

| Round | round_poly (ms) | bind (ms) | Backend | Notes |
|-------|----------------|-----------|---------|-------|
| 0 | 57.6 | 41.5 | Metal | reduce=57ms, fused bind |
| 1 | 0.2 | 20.5 | Metal | cached reduce, fused bind |
| 2 | 0.2 | 10.6 | Metal | halving pattern |
| 3 | 0.2 | 5.6 | Metal | |
| 4 | 0.2 | 3.1 | Metal | |
| 5 | 0.2 | 1.7 | Metal | |
| 6 | 0.2 | 1.1 | Metal | |
| 7 | 0.2 | 1.3 | Metal | migration round |
| 8-19 | ~0.2ea | ~0.4ea | CPU | fast, not worth optimizing |

**Analysis**:
1. **fused_interpolate_reduce is the #1 bottleneck at 57%** — not pairwise_reduce as initially hypothesized
2. Round 0 alone is 66% of hybrid time (reduce + first fused bind)
3. GPU rounds 1-7 are dominated by fused_interpolate_reduce halving pattern
4. CPU rounds (8-19) total only 3.5% — migration threshold is well-calibrated
5. Transcript overhead is negligible — no optimization needed there
6. **Target for 5x**: Total must drop from 149ms to 52.6ms → GPU kernels must get ~2.8x faster
7. Cooperative threading (O0a) targets BOTH fused_interpolate_reduce and pairwise_reduce

**Decision**: KEEP (instrumentation merged)

---

### EXP-002: (next experiment goes here)
