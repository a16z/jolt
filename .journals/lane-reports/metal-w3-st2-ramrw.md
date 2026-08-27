# Metal W3 stage-2 RAM read/write

**Verdict: NO-GO; harness and env-gated experiments retained, production schedule unchanged.** GPU CSR construction clears its isolated slice bar but models only **0.219–0.259 s** off the certified stage, below the **0.3 s** gate. Bind+message fusion cuts command buffers almost in half and moves no wall time.

## Certified attribution (`2^27`)

Source: `.journals/artifacts/monitor-2to27-20260804.json`; nested spans are not double-counted.

| stage-2 slice | wall | stage share |
|---|---:|---:|
| `SpartanProductUniskip::prepare` | 0.538 s | 18.2% |
| `RamReadWriteChecking::prepare` | 0.501 s | 16.9% |
| `RamRafEvaluation::prepare` | 0.392 s | 13.3% |
| `ProductRemainder::prepare` | 0.317 s | 10.7% |
| `InstructionClaimReduction::prove_round` | 0.451 s | 15.2% |
| `RamReadWriteChecking::prove_round` | 0.384 s | 13.0% |
| `ProductRemainder::prove_round` | 0.163 s | 5.5% |
| other member work + batch glue | 0.197 s | 6.7% |
| **stage 2** | **2.958 s** | **100%** |

RAM-RW prepare separates at the `RamValFinal` oracle boundary:

| interval | wall | RAM-RW prepare share | contents |
|---|---:|---:|---|
| entry → oracle begin | 0.378 s | 75.6% | address validation, sparse CSR build, `RamInc` |
| oracle lookup | 0.005 s | 1.0% | `RamValFinal` |
| oracle end → return | 0.117 s | 23.3% | `val_init` reconstruction, device allocations |

RAM-RW rounds contain 81 traced begin/collect envelopes but 40 logical rounds. The first 15 logical rounds consume **0.356 s / 92.6%** and stay device-resident; the transition plus host tail consumes 0.029 s.

## Exact device schedule

At `log_t=27`, rows remain on Metal through messages 0…14. Round 15 performs the final device bind to 4096 rows, converts the sparse state, then continues on the host.

```text
r0:       message -> wait
r1..r14:  host counts prefix-scan -> bind -> wait -> message -> wait
r15:      host counts prefix-scan -> bind -> wait -> host transition/message
```

Legacy total: **15 message dispatches + 15 bind dispatches = 30 command buffers / 30 waits / 15 host scans**. Fusion: **30 dispatches in 16 command buffers / 16 waits**, with all host scans unchanged. Stage-2 batch membership makes rounds 0…12 RAM-RW-only; those 13 rounds consume 0.326 s, so detached command buffers have no peer work to cover their waits.

## Door A — GPU CSR construction

`jk_ram_rw_build` writes the existing 88-byte `RawRamRwEntry` representation directly from `RamAccessColumns`. The host still builds byte-identical offsets; the detached build overlaps `RamInc`. `JOLT_RAMRW_GPU_PREPARE=1` enables the experiment; default is legacy.

The Criterion fixture uses production `T`, one sparse entry per active cycle, 1-in-8 RAM-access density, fixed RAM address range, and the existing early-handoff geometry. The setup oracle compares every entry and offset byte before timing.

| size | serial A / B | GPU A / B | reduction range |
|---|---:|---:|---:|
| `2^22` | 14.658 / 17.288 ms | 7.480 / 5.968 ms | 49.0–65.5% |
| `2^24` | 70.249 / 69.662 ms | 22.113 / 29.372 ms | **57.8–68.5%** |

All paired confidence intervals are disjoint. An earlier dense 7-in-8 sensitivity arm measured 81–83% at `2^24`; it is intentionally not used for retention.

Stage calibration uses the certified 0.378 s parent interval, not optimistic linear scaling of the synthetic slice: `0.378 × 57.8–68.5% = 0.219–0.259 s`. Modeled stage 2: **2.699–2.739 s** (7.4–8.8% faster), missing the 0.3 s gate by 0.041–0.081 s. Peak prepare allocation is unchanged to first order: the GPU output replaces the host entry vector; no additional `T`-sized representation exists.

## Door B — bind+message fusion

`JOLT_RAMRW_FUSED=1` encodes the existing bind and following message into one command buffer with a buffer barrier. CSR layout, field arithmetic, dispatch count, and host scans are unchanged.

| size | legacy A / B | fused A / B | result |
|---|---:|---:|---:|
| `2^22` | 109.40 / 110.77 ms | 110.31 / 110.78 ms | overlap; ~flat |
| `2^24` | 461.47 / 464.75 ms | 466.26 / 459.78 ms | overlap; ~flat |

The `2^24` fixture pins **23 → 12 command buffers** and **23 → 23 dispatches**. With fixed kernel work, halving waits produces no measurable gain; the dominant round slice is sparse merge/field work, not host synchronization. Door rejected.

## Gates

- GPU CSR vs serial CSR byte oracle: pass.
- Legacy and fused round polynomials vs CPU twin: pass.
- Legacy/fused dispatch-count pins: pass.
- Existing optimized-vs-Metal handoff parity, candidate knobs enabled: pass.
- Transcript bytes: unchanged by construction; exact round-polynomial parity pins the absorbed coefficients.
- Retention: CSR slice ≥12% pass; modeled stage gain ≥0.3 s **fail**. Fusion dominant slice ≥12% **fail**.
