# C2 acceptance: projected 10 MHz average reached

2026-09-05. **Accept C2.** The rebuilt, integrated evaluator reaches
**10.498382 MHz** arithmetic-average projected M5 Max padded throughput,
4.9838% above the frozen 10 MHz target. This is an M4 Max measurement with
the campaign's unchanged 1.13 projection, not an M5 hardware measurement.
Fibonacci remains below 10 MHz individually; the objective is the average.

## Frozen paired matrix

One M4 Max, serial, no subagents. Every workload used parent, candidate,
candidate, parent with 120-second inter-run gaps and an initial 120-second
cooldown. All twelve proofs printed `PROOF_VERIFIED backend=metal value=true`.
No watchdog aborts, process swaps, exclusions or guard failures occurred.

| Workload | Parent samples s | C2 samples s | Parent mean s | C2 mean s | Saved s | Projected M5 MHz |
|---|---|---|---:|---:|---:|---:|
| BTreeMap | 29.50, 29.25 | 26.88, 26.82 | 29.375 | 26.850 | 2.525 | 11.2973 |
| Fibonacci | 35.02, 35.32 | 31.98, 31.84 | 35.170 | 31.910 | 3.260 | 9.5059 |
| SHA-2 | 32.37, 32.51 | 28.32, 28.42 | 32.440 | 28.370 | 4.070 | 10.6920 |

The score is `sum(1.13 * 2^28 / (1e6 * mean_wall_s)) / 3`, not the reciprocal
of the average wall. Fresh paired parent: 9.433831 MHz; historical accepted
matrix: 9.251347 MHz. Use the fresh controls for C2's improvement claim.
Fibonacci repeats the earlier candidate-pair saving of 3.265 s with 3.260 s
after integration. Both new parent controls lie within 34.5–37.5 s.

Using the slower C2 sample for each workload still gives 10.480973 MHz.
This is a two-sample sensitivity check, not a confidence interval. The
break-even M5/M4 speed factor is 1.076356; the frozen assumption is 1.13.
Peak C2 RSS is 89,801,654,272 bytes (83.63 GiB), below the 90 GiB limit.

## What supplied the gain

C2 combines the D128/rank-3 root commitment with device D-role relation
rows and the row-aligned D128 decompose-fold. Nominal commitment coefficient
updates fall 25% relative to D512/rank-1. Evaluation must stay on the fast
path for that reduction to survive end to end. No challenge, SIS policy,
digit basis, transcript or verifier relation was changed. CPU and Metal
now select the same already-admitted K256 schedule rows.

Separate Fibonacci Chrome diagnostics, not paired scores, show Stage0
17.154852 → 12.755944 s and Stage8 6.229150 → 6.745183 s: 3.882875 s saved
across the combined boundary. The original commit-only C lost its gain to
a large CPU fallback in evaluation. Retained C2's device D-role and fold
address those measured costs. A quotient construction remains on the CPU;
existing NTT prewarming hides its cold-cache setup. No quotient shortcut
or hypothetical domain-halving saving is credited.

## Correctness and production gates

- All 327 current `jolt-kernels --features metal` tests passed, none skipped.
  The earlier handoff's 328 count was not this checkout's inventory.
- Final-pin adapter/catalog tests: 43 passed, one pre-existing ignored test.
  Final fork Metal-family tests: 32 passed; serial packing tests: 17 passed.
- Both exact Jolt clippy modes (`host` and `host,zk`) and clean committed-tree
  fmt passed. All four required fork clippy configurations passed.
- Both Jolt catalogs regenerated; CPU artifact unchanged, Metal artifact
  byte-identical to CPU. Fork schedule regeneration changed no tables.
- Full production CPU/Metal commitment comparisons matched all 12,582,912
  coefficients on each workload, then completed verified Metal proofs.
  CPU commit rates B/F/S were 15.407794 / 16.183877 / 16.188270 ns per hot
  entry, all below the unchanged 19 ns ceiling. These diagnostic CPU
  components are not reruns of the frozen full-proof CPU references.
- Clean diagnostic Stage0 B/F/S: 7.726366 / 12.755944 / 11.053317 s, below
  the unchanged 9.8 / 13.5 / 14.4 s production ceilings.
- Synthetic slope 2.857 ns/hot and intercept 2.951 s pass the slope ceiling
  3.2 ns/hot and user-approved corrected multi-task intercept ceiling 3.2 s.
  The correction's known-result timing and twofold traffic-accounting error
  are documented in [the floor spec](akita-metal-d128-rank3-root-floor.md).

The new shared T28 row digest is
`e140678a759cb3645c549471db4647b92b87c283bf025abba0b735462cab5c26`;
the retired Metal digest is
`968284aa9aec7860b6b06816b9069fc67d6272eab071123501fa90bf271e41c0`.
The old Metal catalog rejects the new selection and the new CPU/Metal
catalogs reject the retired selection. This is catalog-scoped: the old
generic dispatcher already supported the CPU catalog and its D128 row.
Do not claim every old generic verifier rejects every new proof.

## Reconstruction and limitations

Measured Jolt runtime revision: `1ad8a6b436302d835c9bc75fa1ade8607fa02770`;
Akita pin: `d756e3a67954fb39efa03809ec59971b15303860`.
Parent: Jolt `b160c87ea50096085e92cf33fdc8c4f5923b436f`, Akita `369a1c7ff`.
Subsequent Jolt acceptance changes are documentation/evidence only.

```sh
cargo build --release -p jolt-prover --example modular_benchmark --features prover-fixtures,metal,profiling
modular_benchmark --name <btreemap|fibonacci|sha2-chain> --scale 28 --backend metal --format none
```

Candidate binary SHA256:
`f38326dbde98ffc1518e0f094a75f2c86ce1d102f2cbccf142cd553e88fd993f`.
Parent binary SHA256:
`d113abcfdfa9a1161128cec8a058786540d6de1b123428e6a45dc2659187932a`.
The [machine-readable evidence](akita-metal-c2-acceptance-2026-09-05.json)
records every raw-output hash, sample, RSS, binary and scorer identity.
Local raw logs are under
`benchmark-runs/akita-10mhz-studies/runs/C2-coupled-matrix-*`.

Could not verify complete fork preflight/CI as green: inherited backend/
runtime file-size limits, a `recursive_commit` error-owner check, and two
unused Metal dependencies still fail auxiliary audits; `typos` is absent.
These are reported debt, not passing checks. The user's uncommitted census
example also has an unrelated fmt issue and was left untouched; clean-tree
validation checked an identical committed tree. No M5 hardware run or
larger statistical replication was performed. Diagnostic CPU-shadow code
and the old E6 prototype are excluded from the landing diff.
