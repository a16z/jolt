# Metal W3 st0 commitment kernel

**RETAIN:** `jk_g1_seg_sum` now accumulates in XYZZ coordinates. Same 256-entry
host segments; mixed add drops from 7M+4S to 8M+2S (11→10 Fq Montgomery
products), then converts each segment once to the existing Jacobian output.

## Attribution

| st0 slice @2^27 | time | exposure |
|---|---:|---|
| certified st0 wall | **17.98 s** | stage wall |
| Metal device-active union | **14.28 s** (79.4%) | dominant wall resource; G1 and Miller queues overlap |
| `jk_g1_seg_sum` family | **~15–16 s inclusive device work** | 2^24 CB trace: ~1.9–2.0 s across 128 superchunks, linearly modeled; overlaps tier-2/Miller |
| tier-2 host preparation | **0.457 s** | `DoryScheme::prepare_tier2`, Rayon-joined with base preparation |
| tier-2 pair absorption | pipeline-resident | consumes each G1 result and feeds Miller; cap sweep prices its host cost below |
| trace-walk feed | **13.337 s** | `TraceRecord::collect`; background Rayon pool, fully hidden (`join = 0`) behind st0 |

The kernel is compute-bound: retained same-window scale-24 median **17.3 GB/s
useful** and **2.52 Gmul/s**, versus 357/485 GB/s contended/isolated bandwidth
roofs and 11.30 Gmul/s Fq roof. Bandwidth reaches 4.8%/3.6% of the roofs;
Montgomery throughput reaches 22.3%.

## Retention A/B

All samples: production SHA2 witness/grid, `gpu_lock()`, one binary, interleaved
A/B, device `GPUStartTime→GPUEndTime`; setup and oracle untimed.

| objective | serial Jacobian | XYZZ | delta |
|---|---:|---:|---:|
| isolated G1, 2^22, pooled median | 39.7 ms | 19.3 ms | **−51.4%** |
| isolated G1, 2^24, pooled median | 21.3 ms | 18.8 ms | **−11.6%** |
| full commitment, 2^24, pooled median | 18.10 s | 16.18 s | **−10.6%, −1.91 s** |
| modeled certified st0 | 17.98 s | 16.08 s | **−1.90 s** (conservative proportional model) |

The 2^24 window was package-derated but tightly paired: full-pipeline serial
17.93–18.13 s, XYZZ 16.08–16.23 s. The proportional 2^27 model clears both
retention bars: 10.6% dominant/full slice and 1.90 s stage wall.

## Doors priced

| candidate | isolated kernel | full commitment | verdict |
|---|---:|---:|---|
| segment cap 128 | ~−27% | +3.6% | reject: tier-2 host absorption grows |
| segment cap 64 | ~−31% | +12.4% | reject |
| segment cap 32 | −44% | ~+38% | reject |
| four threads/segment + TG reduction | +32% | — | reject: 24 KiB TG state cuts occupancy |
| two accumulators/thread | +21% | +3.8% | reject: register pressure |
| **XYZZ mixed add** | **−11.6…−51.4%** | **−10.6%** | **retain** |

## Oracle and gates

- Harness compares every XYZZ tier-1 row sum with the serial Jacobian kernel
  before timing; signed increment indices included by targeted tests.
- `metal_commit_matches_optimized`: full commitments and Dory hints exactly
  equal the host optimized build; transcript inputs therefore unchanged.
- `seg_sums_match_arkworks`, `signed_seg_sums_match_arkworks`, and
  `seg_sum_edge_cases`: pass.
- `cargo fmt`; clippy `-D warnings` for `jolt-kernels` and `jolt-eval`, with
  and without Metal where the target exists: pass.
