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

### Fixed-base (generator) precompute (FIXED_BASE_WINDOW = 8)

Fixed-base window tables let us compute scalar * G with **no doublings** (only lookups + additions).

Using `examples/grumpkin-msm-bench` with `MSM_SIZE=1024`, `GLV_WINDOW=8`, `FIXED_BASE_WINDOW=8`:
- fixed_base_precompute_g_w8: 10,573,780 RV64IMAC cycles (11,105,622 virtual)
- scalar_mul_fixed_base_table_256_w8: 38,343 RV64IMAC cycles (40,935 virtual)
  - vs scalar_mul_256bit: 394,014 RV64IMAC cycles (406,967 virtual) → ~10.1x faster (total cycles)
  - vs scalar_mul_glv_2x128: 225,133 RV64IMAC cycles (236,321 virtual) → ~5.8x faster (total cycles)
- msm_setup: 9,782,897 RV64IMAC cycles (11,117,945 virtual)
  - note: `msm_setup` now uses the fixed-base table to generate points (previously it did 1024x double-and-add scalar mul).
- msm_fixed_base_table_256_w8 (RUN_FIXED_BASE_ONLY): 40,043,310 RV64IMAC cycles (42,730,875 virtual)

### References
- Grumpkin inline constants: `jolt-inlines/grumpkin/src/lib.rs`
- Grumpkin GLV constants and endomorphism: `jolt-inlines/grumpkin/src/sdk.rs`
