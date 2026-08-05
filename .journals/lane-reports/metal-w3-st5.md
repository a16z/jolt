# Metal wave 3 — stage 5 InstructionReadRAF phase scan

## Verdict

**GO.** `jk_irr_phase_scan` is the dominant individual stage-5 kernel. Replacing its 32-turn serialized scatter with collision-only SIMD aggregation cuts the isolated production-shape kernel by **52.6% at 2^22** and **54.0% at 2^24**. Exact device/CPU parity passes for optimized and env-restored legacy arms; proof/wire bytes are unchanged.

## Attribution

Wave-1 stage vector at 2^27: 14.669 s total, including 9.709 s `InstructionReadRaf::prove_round` and 3.016 s `RegistersValEvaluation::prove_round`. A 2^22 `JOLT_METAL_CB_TRACE=1 JOLT_IRR_SPLIT_TRACE=1` run split the previously bundled scanner passes:

| owner | command buffers | summed GPU time | share of traced IRR scanner |
|---|---:|---:|---:|
| `jk_irr_phase_scan` + tiny `jk_irr_reduce` | 16 | ~174 ms | 51.1% |
| `jk_irr_suffix_scan` + reduce | 16 | ~80 ms | 23.5% |
| `jk_irr_cycle_round` | 22 | ~74 ms | 21.7% |
| cycle initialization | 1 (9 dispatches) | ~12.6 ms | 3.7% |

Individual phase-scan buffers were typically 10–11 ms before the change. `jk_irr_phase_scan` therefore owns the largest kernel mass, rather than RegistersValEvaluation or command-buffer binds.

## Harness

`jolt-eval/benches/metal/instruction_read_raf_phase_scan.rs`:

- production `TARGET_SIMDGROUPS=512` scanner geometry at 2^22 and 2^24;
- full-width randomized 128-bit lookup indices and production RAF-family mix;
- fixture construction, CPU oracle, Metal buffer wrapping, and warm-up outside timing;
- exact final-cell oracle against CPU field arithmetic before timing;
- legacy and optimized kernels measured in the same process under `gpu_lock()`.

## Change

The old non-uniform scatter performed 32 barrier-ordered device-memory read/modify/writes per tile. Most random rows target distinct `(family, chunk)` cells, so those turns serialized independent work.

The new path first exchanges cheap 9-bit scatter keys. Unique lanes update concurrently; only colliding lanes shuffle and sum the three field values, with one leader per collided key writing memory. The existing uniform-tile reduction remains intact. This avoids shuffling 24 limbs for the common unique-key case and removes the 32-turn barrier chain.

Rejected probes:

| probe | result | decision |
|---|---:|---|
| full-key SIMD field reduction | ~20.7 ms at 2^22, ~84.1 ms at 2^24 | rejected: 2.3x slower from unconditional limb shuffles/adds |
| 256 / 512 / 1024 simdgroups | ~7.3 / ~5.7 / ~6.4 ms at 2^22 | retain production 512: best occupancy/register-pressure balance |
| threadgroup atomic spin lock | SIMD forward-progress deadlock | rejected |

`JOLT_IRR_PHASE_SCAN_LEGACY=1` restores the structural legacy kernel. `JOLT_IRR_SPLIT_TRACE=1` preserves the attribution-only command-buffer split.

## Retention

Thirty paired samples per geometry, alternating AB/BA order; intervals are 95% t CIs from the same-window run:

| rows | legacy | optimized | reduction | effective Montgomery products/s |
|---:|---:|---:|---:|---:|
| 2^22 | 10.481 ± 0.020 ms | 4.966 ± 0.018 ms | 52.6% | 0.667 → 1.407 G/s |
| 2^24 | 41.116 ± 0.032 ms | 18.911 ± 0.184 ms | 54.0% | 0.680 → 1.479 G/s |

The second and final retention run, Criterion `--quick`, corroborated with non-overlapping intervals: 2^22 legacy `[10.447, 10.493]` ms vs optimized `[4.971, 4.983]` ms; 2^24 legacy `[41.171, 41.194]` ms vs optimized `[18.755, 19.079]` ms.

The effective rate counts the harness mix's 5/3 scalar Montgomery multiplications per row. Optimized throughput is only 13.1% of the measured 11.30 Gmont-mul/s device roof: the kernel was scatter-serialization-bound, not at the ALU roof.

## 2^27 model

Share-calibrated model: `9.709 s × (174 / 340.6) × 54.0% = 2.68 s` saved. On the certified 14.81 s stage vector, modeled stage time is **~12.13 s**. Raw 2^22 linear scaling gives ~3.0 s saved; 2.68 s is the retained conservative model and exceeds the 0.4 s bar by 6.7x.

## Verification

- exact optimized + legacy device/CPU fixture parity at 2^22 and 2^24: pass
- `cargo fmt -q --message-format=short`: pass
- clippy `-D warnings`: pass for `jolt-kernels` without Metal, `jolt-kernels` with `metal,bench-utils`, and `jolt-eval --features metal --all-targets`
- targeted InstructionReadRAF nextest with optimized kernel: 10/10 pass
- targeted InstructionReadRAF nextest with `JOLT_IRR_PHASE_SCAN_LEGACY=1`: 10/10 pass
