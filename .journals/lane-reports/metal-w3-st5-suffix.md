# Metal wave 3 — stage 5 InstructionReadRAF suffix scan

## Verdict

**GO.** `jk_irr_suffix_scan` was the new dominant unoptimized device kernel after the phase-scan merge. Collision-only SIMD scatter cuts the isolated kernel **35.1% at 2^22** and **35.3% at 2^24** with disjoint 95% CIs. A 2048-simdgroup production schedule then cuts scan+reduce another 17.4% at both sizes. Modeled stage-5 saving: **1.26 s at 2^27**, taking the post-phase model from 12.13 to **~10.87 s**.

## Attribution

Post-phase-fix `JOLT_METAL_CB_TRACE=1 JOLT_IRR_SPLIT_TRACE=1`, exact 2^22 scanner fixture:

| owner | CBs | GPU time | calibrated 2^27 mass |
|---|---:|---:|---:|
| `jk_irr_suffix_scan` + reduce | 16 | 92.413 ms | 2.63 s |
| optimized `jk_irr_phase_scan` + reduce | 16 | 90.523 ms | 2.58 s |
| `jk_irr_cycle_round` | 22 | 54.738 ms | 1.56 s |
| `jk_irr_cycle_init` | 1 / 9 dispatches | 5.623 ms | 0.16 s |

RegistersValEvaluation's 3.016 s is CPU span mass, not exposed stage wall: the wave-1 mixed-tail trace placed all but ~31 ms under asynchronous InstructionReadRAF device work. Address-message assembly/binds are dispersed host work; the measured device prefix was 94.2% GPU-equivalent. `jk_irr_suffix_scan` was therefore the largest new critical-path kernel.

Final 2048-group trace: suffix scan+reduce **48.288 ms**, down 47.7%; phase 89.910 ms, cycle 55.827 ms, init 5.643 ms.

## Harness

`jolt-eval/benches/metal/instruction_read_raf_suffix_scan.rs` uses production 2^22/2^24 bucket geometry, five real lookup-table suffix decompositions, randomized 128-bit indices/field weights, `gpu_lock()`, and setup/buffer wrapping outside timing. Before timing, optimized and legacy kernels reduce their partials and match CPU field sums exactly.

## Change

- Replace each suffix's 32 barrier-serialized RMW turns with key-first SIMD collision detection: unique chunks write concurrently; only colliding chunks shuffle/reduce field limbs.
- Skip zero-valued suffix emissions and clear only the table's live suffix rows instead of all eight capacity rows.
- Raise suffix-only scheduling 512 → 2048 simdgroups; 4096 gives overlapping combined CIs at ~2× partial memory.
- `JOLT_IRR_SUFFIX_SCAN_LEGACY=1` restores both the old kernel and 512-group schedule. Protocol/wire bytes are unchanged.

Rejected: hoisting chunk-collision discovery outside the suffix loop regressed the stable 2^22 candidate from 3.796 to 4.125 ms; extra live state outweighed the repeated key shuffles.

## Retention

Thirty alternating AB/BA pairs, same process/window:

| rows | legacy 512 | candidate 512 | isolated reduction |
|---:|---:|---:|---:|
| 2^22 | 5.852 ± 0.010 ms | 3.796 ± 0.009 ms | 35.1% |
| 2^24 | 22.672 ± 0.171 ms | 14.671 ± 0.233 ms | 35.3% |

Schedule probe, candidate scan+reduce:

| rows | 512 groups | 2048 groups | reduction |
|---:|---:|---:|---:|
| 2^22 | 3.856 ± 0.008 ms | 3.186 ± 0.022 ms | 17.4% |
| 2^24 | 14.644 ± 0.233 ms | 12.091 ± 0.183 ms | 17.4% |

The fixture performs ~0.514 scalar Montgomery products per row: the final 2^24 scan reaches ~0.77 Gmont-mul/s, only 6.8% of the 11.30 Gmont-mul/s roof. Indexed gathers/scatter serialization—not field ALU—were limiting. The 2048 schedule adds ~0.10 GB of partials at production scale.

## Model

`9.709 s × (92.413 − 48.288) / 340.6 = 1.258 s` saved. The isolated 35.3% kernel bar models 0.93 s even without the retained occupancy gain; both exceed the 0.4 s stage bar.

## Verification

- exact optimized + legacy device/CPU fixture parity at 2^22 and 2^24: pass
- final 2^22 full scanner attribution parity: pass
- `cargo fmt -q --message-format=short`: pass
- clippy `-D warnings`: pass for `jolt-kernels` without Metal, `jolt-kernels` with `metal,bench-utils`, and `jolt-eval --features metal --all-targets`
- targeted InstructionReadRAF nextest with optimized kernel: 10/10 pass
- targeted InstructionReadRAF nextest with `JOLT_IRR_SUFFIX_SCAN_LEGACY=1`: 10/10 pass
