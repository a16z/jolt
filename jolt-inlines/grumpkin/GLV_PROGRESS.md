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

### Benchmarks (MSM_SIZE = 1024, GLV_WINDOW = 12)
- scalar_mul_256bit: 394,008 RV64IMAC cycles (406,958 virtual)
- scalar_mul_glv_2x128: 225,129 RV64IMAC cycles (236,315 virtual)
- msm_setup: 158,250,252 RV64IMAC cycles (251,829,042 virtual)
- msm (no GLV, no Pippenger): 423,651,173 RV64IMAC cycles (439,598,630 virtual)
- msm_glv: 237,894,611 RV64IMAC cycles (249,198,175 virtual)
- msm_pippenger: 180,472,760 RV64IMAC cycles (272,948,304 virtual)
- msm_glv_pippenger: 105,865,710 RV64IMAC cycles (155,068,026 virtual)

### Benchmarks (MSM_SIZE = 2048, GLV_WINDOW = 12)
- scalar_mul_256bit: 394,011 RV64IMAC cycles (406,962 virtual)
- scalar_mul_glv_2x128: 225,129 RV64IMAC cycles (236,315 virtual)
- msm_setup: 316,427,460 RV64IMAC cycles (503,490,600 virtual)
- msm (no GLV, no Pippenger): 847,441,260 RV64IMAC cycles (879,300,749 virtual)
- msm_glv: 475,663,443 RV64IMAC cycles (498,259,783 virtual)
- msm_pippenger: 206,746,912 RV64IMAC cycles (300,297,496 virtual)
- msm_glv_pippenger: 135,028,466 RV64IMAC cycles (186,059,682 virtual)

### Window Sweep (MSM_SIZE = 1024, msm_glv_pippenger)

| GLV_WINDOW | RV64IMAC | Virtual | Total |
|------------|----------|---------|-------|
| 12 | 105,865,712 | 155,068,028 | **260,933,740** |
| 10 | 57,290,195 | 73,217,503 | **130,507,698** |

**Finding:** GLV_WINDOW=10 is ~2x faster than GLV_WINDOW=12 for MSM_SIZE=1024.
Smaller window reduces bucket accumulation cost (2^w additions per window).

### References
- Grumpkin inline constants: `jolt-inlines/grumpkin/src/lib.rs`
- Grumpkin GLV constants and endomorphism: `jolt-inlines/grumpkin/src/sdk.rs`
