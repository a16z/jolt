## Grumpkin GLV Decomposition Progress

### Goal
Implement GLV scalar decomposition for Grumpkin and integrate it into MSM to
reduce doublings and improve overall MSM performance.

### Current State
- Grumpkin has division inlines for base/scalar fields.
- GLV endomorphism constants (`beta`, `lambda`) and the general endomorphism map
  are now defined in `jolt-inlines/grumpkin/src/sdk.rs`.
- GLV decomposition inline is registered and has host/guest implementations.

### Tasks
- [x] Confirm Grumpkin endomorphism constants (`beta`, `lambda`) and formulas.
- [x] Add `GrumpkinPoint::endomorphism()` for arbitrary points.
- [x] Add GLV inline constants in `jolt-inlines/grumpkin/src/lib.rs`.
- [x] Implement guest-side `GrumpkinPoint::decompose_scalar()` using the new
      GLV inline + correctness check (similar to secp256k1).
- [x] Implement host-side fallback `decompose_scalar()` (bigint path).
- [x] Implement `sequence_builder` + `advice` for GLV decomposition.
- [x] Register GLV inline in `host.rs`.
- [x] Add tests for GLV decomposition and GLV-based scalar mul correctness.
- [x] Integrate GLV into MSM benchmark and record cycle deltas.
- [x] Add fixed-base (generator) precompute table in the MSM benchmark for fast scalar mul + point generation.

### Benchmarks (MSM_SIZE = 1024, GLV_WINDOW = 12)
- scalar_mul_256bit: 394,008 RV64IMAC cycles (406,958 virtual)
- scalar_mul_glv_2x128: 225,129 RV64IMAC cycles (236,315 virtual)
- msm_setup: 9,782,897 RV64IMAC cycles (11,117,945 virtual)
- msm (no GLV, no Pippenger): 423,651,173 RV64IMAC cycles (439,598,630 virtual)
- msm_glv: 237,894,611 RV64IMAC cycles (249,198,175 virtual)
- msm_pippenger: 180,472,760 RV64IMAC cycles (272,948,304 virtual)
- msm_glv_pippenger: 105,865,710 RV64IMAC cycles (155,068,026 virtual)

### Benchmarks (MSM_SIZE = 2048, GLV_WINDOW = 12)
- scalar_mul_256bit: 394,011 RV64IMAC cycles (406,962 virtual)
- scalar_mul_glv_2x128: 225,129 RV64IMAC cycles (236,315 virtual)
- msm_setup: 19,218,310 RV64IMAC cycles (21,887,675 virtual)
- msm (no GLV, no Pippenger): 847,441,260 RV64IMAC cycles (879,300,749 virtual)
- msm_glv: 475,663,443 RV64IMAC cycles (498,259,783 virtual)
- msm_pippenger: 206,746,912 RV64IMAC cycles (300,297,496 virtual)
- msm_glv_pippenger: 135,028,466 RV64IMAC cycles (186,059,682 virtual)

### Window Sweep (MSM_SIZE = 1024, msm_glv_pippenger)

| GLV_WINDOW | RV64IMAC | Virtual | Total |
|------------|----------|---------|-------|
| 7 | 51,167,509 | 56,501,923 | 107,669,432 |
| **8** | **49,167,132** | **55,906,558** | **105,073,690** |
| 9 | 50,293,472 | 60,566,281 | 110,859,753 |
| 10 | 57,290,195 | 73,217,503 | 130,507,698 |
| 11 | 73,673,369 | 101,369,899 | 175,043,268 |
| 12 | 105,865,712 | 155,068,028 | 260,933,740 |

**Finding:** GLV_WINDOW=8 is optimal for MSM_SIZE=1024.
- 2.5x faster than GLV_WINDOW=12 (105M vs 261M total cycles)
- Sweet spot balances: fewer buckets (2^8=256) vs more windows (128/8=16)

### Window Sweep (MSM_SIZE = 2048, msm_glv_pippenger)

| GLV_WINDOW | RV64IMAC | Virtual | Total |
|------------|----------|---------|-------|
| 7 | 98,037,230 | 106,184,563 | 204,221,793 |
| 8 | 91,202,727 | 100,513,695 | 191,716,422 |
| **9** | **87,456,616** | **100,064,771** | **187,521,387** |
| 10 | 91,876,698 | 109,993,062 | 201,869,760 |

**Finding:** GLV_WINDOW=9 is optimal for MSM_SIZE=2048.
- Larger MSM amortizes bucket cost → optimal window shifts up
- 1.7x faster than GLV_WINDOW=12 at MSM_SIZE=2048 (188M vs 321M)

### Fixed-Base Precompute

**Idea:** For a fixed generator G, precompute `Table[w][d] = d * (2^(w*window)) * G` for all windows w and digits d.
Then scalar multiplication becomes pure **lookups + additions** — no doublings needed.

#### Precompute Table Sizes

| w | windows | entries/window | total entries | memory |
|---|---------|----------------|---------------|--------|
| 8 | 32 | 256 | 8,192 | 0.5 MiB |
| 10 | 26 | 1,024 | 26,624 | 1.6 MiB |
| 11 | 24 | 2,048 | 49,152 | 3.0 MiB |
| 12 | 22 | 4,096 | 90,112 | 5.5 MiB |
| 13 | 20 | 8,192 | 163,840 | 10.0 MiB |
| 14 | 19 | 16,384 | 311,296 | 19.0 MiB |

*(Each affine point = 64 bytes: two 256-bit field elements)*

#### Fixed-Base Window Sweep (MSM_SIZE=1024, msm_fixed_base_table)

| w | RV64IMAC | Virtual | **Total** | vs GLV+Pippenger |
|---|----------|---------|-----------|------------------|
| 8 | 40,208,584 | 42,896,149 | 83,104,733 | 1.26x |
| 9 | 36,403,488 | 38,911,414 | 75,314,902 | 1.40x |
| 10 | 33,061,817 | 35,421,326 | 68,483,143 | 1.53x |
| 11 | 29,854,092 | 32,072,225 | 61,926,317 | 1.70x |
| 12 | 27,726,692 | 29,848,563 | 57,575,255 | 1.82x |
| 13 | 25,693,311 | 27,726,498 | 53,419,809 | 1.97x |
| **14** | **23,980,749** | **25,937,710** | **49,918,459** | **2.10x** |

**Finding:** FIXED_BASE_WINDOW=14 is optimal for MSM_SIZE=1024, giving **49.9M** total cycles.
- **2.1x faster** than GLV+Pippenger (50M vs 105M)
- Trade-off: 19 MiB precompute table (heap-allocated via `Box`)

#### MSM Comparison Summary (MSM_SIZE=1024)

| Method | Total Cycles | vs GLV+Pippenger |
|--------|--------------|------------------|
| Pippenger (256-bit, no GLV) | 453M | 0.23x |
| **GLV + Pippenger (w=8)** | **105M** | **1.0x (baseline)** |
| Fixed-base table (w=8) | 83M | 1.26x |
| Fixed-base table (w=14) | 50M | 2.10x |

### References
- Grumpkin inline constants: `jolt-inlines/grumpkin/src/lib.rs`
- Grumpkin GLV constants and endomorphism: `jolt-inlines/grumpkin/src/sdk.rs`
