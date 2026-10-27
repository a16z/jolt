# Whiteout benchmark, idle rerun (P0b)

Parent: `specs/whiteout-optimization-campaign.md` (P0, and the O6/P9
decision rule). Method: `specs/whiteout-o0-profile.md`, unchanged.

**Code.** jolt whiteout/dev `173821f29` (after O0–O5 and M1), Akita
`~/akita-whiteout` `6ca22bd57`. Built with `CARGO_PROFILE_RELEASE_DEBUG=0`,
`--release -p jolt-prover --features profiling,akita[,zk]`.

**Harness.** `scripts/whiteout_verify_profile.py run --runs 7`: clear and
Whiteout interleaved, the first build alternating by round, `--backend
optimized --format chrome`. Each binary ran through a wrapper that waited
until the 1-minute load was below 2 (at most 20 minutes) and then ran it
under `/usr/bin/time -l`. The script records the load before that wait, so
the loads below are the wrapper's, taken at launch. 41 of 42 runs started
at load 1.74–1.99. One clear muldiv run hit the 20-minute cap and started
at 3.87. Cells give the median [min–max] of 7 runs.

## Headline (Whiteout, idle)

| | muldiv (T = 2^12) | fib 2^16 | fib 2^20 |
|---|---:|---:|---:|
| steady verify, 16 threads (ms) | 80.1 [79.0–81.8] | 88.1 [85.7–90.1] | 105.0 [98.7–110.6] |
| steady verify, 1 thread (ms) | 143.0 [141.6–143.9] | 178.5 [177.5–180.2] | 275.1 [273.0–283.1] |
| first verify, transported (ms) | 119.1 [117.1–121.6] | 138.1 [134.3–139.5] | 167.0 [164.8–181.0] |
| clear Akita verify, 16 / 1 threads (ms) | 5.6 / 6.6 | 5.6 / 6.8 | 8.3 / 14.2 |
| residual products A_a + [B_a\|B′_a], share of 1-thread verify | 28.6% [28.4–28.8] | 37.4% [37.3–37.6] | 52.8% [52.6–52.9] |
| `N·z` wall time (residual + pad), share of 16-thread verify | 10.8% [9.7–12.0] | 13.5% [12.4–15.1] | 18.7% [17.5–21.5] |
| proof bytes, Whiteout / clear | 93,348 / 81,977 | 95,579 / 84,120 | 97,900 / 88,685 |

With 16 threads, the products are summed over workers (CPU time) and exceed
the wall time, so the 16-thread share uses the `whiteout_weight_nz` wall
span. That span is an upper bound: it includes the pad block.

The first transported verify costs 39–62 ms more than a steady one. The
public-state builds under `whiteout_response_lookup` account for 25–37 ms
of that (pad commitment, residual matrices, `A_D`). Whiteout proof bytes
vary by 70–100 B across runs.

## Steady verify by bucket (16 threads / 1 thread, ms)

| bucket | muldiv | fib 2^16 | fib 2^20 |
|---|---:|---:|---:|
| response proof: `N·z` | 8.5 / 51.2 | 11.9 / 77.3 | 19.3 / 155.4 |
| response proof: lifted rows (item 4) | 18.9 / 35.9 | 19.1 / 36.9 | 19.5 / 38.0 |
| pad-layout replay | 12.5 / 13.6 | 13.0 / 15.0 | 15.0 / 19.8 |
| response proof: row batch | 12.6 / 12.1 | 13.4 / 12.9 | 14.3 / 13.5 |
| response proof: eq dual | 4.3 / 5.7 | 6.2 / 8.8 | 8.8 / 14.0 |
| seed, stages 1–7, stage-8 claims | 7.4 / 7.4 | 7.6 / 7.4 | 7.6 / 7.6 |
| relation (auxiliary block) | 6.0 / 6.0 | 6.6 / 6.6 | 7.2 / 7.3 |
| masked prefix | 2.2 / 3.0 | 2.4 / 4.4 | 4.3 / 9.1 |
| response opening E(r) | 3.6 / 4.3 | 3.8 / 4.9 | 3.9 / 5.3 |
| lift | 2.0 / 1.9 | 2.1 / 2.0 | 2.1 / 2.0 |

A steady verify builds no public state: that bucket reads 0.0 ms in every
cell. The Σ check apart from the response proof, the product and fused
sumchecks, and the digit range each take at most 0.1 ms.

16 threads beat 1 thread by 1.8×, 2.0× and 2.6×. `N·z` scales 6–8×, the
lifted rows 1.9×. The row batch and the pad-layout replay do not scale
(0.9–1.3×).

## Prover (ms unless noted)

| | muldiv | fib 2^16 | fib 2^20 |
|---|---:|---:|---:|
| clear `jolt_prover::prove` | 169.0 [159.1–173.6] | 294.5 [278.7–405.9] | 1,016.1 [985.0–1,044.4] |
| Whiteout `jolt_prover::prove`, first use | 3,358.7 [3,324.8–3,398.2] | 5,577.8 [5,495.7–5,695.4] | 5,841.7 [5,748.1–5,947.5] |
| Whiteout prove minus the response plan | 1,243.3 [1,237.9–1,256.4] | 1,778.6 [1,765.3–1,874.2] | 3,266.3 [3,250.6–3,322.2] |
| overhead over clear (plan excluded) | 7.4× | 6.0× | 3.2× |
| `whiteout_prove_tail` | 1,022.7 [1,016.6–1,032.5] | 1,432.3 [1,420.9–1,447.8] | 2,195.1 [2,172.8–2,231.6] |
| — response proof (`whiteout_prove_response`) | 723.1 [709.4–727.9] | 1,011.2 [997.3–1,031.5] | 1,566.9 [1,554.3–1,605.0] |
| — Σ (tail minus response) | 298.9 [295.2–307.3] | 419.0 [416.3–424.8] | 626.2 [618.5–630.0] |
| response plan (`whiteout_plan_response`), first use | 2,115.4 [2,080.8–2,149.6] | 3,789.0 [3,722.8–3,894.0] | 2,591.1 [2,489.8–2,679.3] |
| catalog plan (`whiteout_plan_catalog`), at setup | 434.6 [421.7–448.6] | 479.8 [471.7–490.7] | 831.1 [817.7–849.0] |
| loading the persisted plans (catalog + response) | 11.6 + 1.0 | 17.7 + 1.2 | 21.0 + 1.5 |
| peak RSS, Whiteout / clear (GiB) | 2.17 / 0.32 | 3.44 / 0.29 | 5.28 / 0.61 |
| peak footprint, Whiteout (GiB) | 1.83 | 2.84 | 4.63 |

Each process is fresh, so every Whiteout prove plans its response row
under `commit_pads` (`commit_pads` − `whiteout_plan_response` = 37–52 ms).
The second row subtracts that plan.

## Verdict

- **O6 at d = 128 goes first**, as the rule requires.
- **d = 256, and P9 on top of it: the gate is met only at fib 2^20 on one
  thread.** The residual products take 52.8% [52.6–52.9] there. They take
  28.6% at muldiv and 37.4% at fib 2^16. On 16 threads, all of `N·z` stays
  at or below 19% of the verify. The share grows with T: the 1-thread `A_a`
  product rises from 30.6 to 50.8 to 125.1 ms. Read the gate the way O0
  measured it (1 thread, residual products) at the largest workload. Then
  the rule allows d = 256, but by 2.8 points. Recommendation: land
  d = 128, rerun this benchmark, and start d = 256 (then P9) only if fib
  2^20 still reads ≥ 50% on one thread. P9's rank certification
  (`sigma_binding_width_table`) can proceed in parallel.
- **The 16-thread steady verify is now spread over four items.** In order:
  the lifted rows (19–23%), the pad-layout replay (14–16%), the response
  row batch (14–16%) and `N·z` (11–19%). The replay and the row batch are
  serial. For a ticket targeting steady verify on 16 threads, those three
  come before the residual.

## Anomalies

- The response plan takes longer at fib 2^16 (3.79 s) than at fib 2^20
  (2.59 s). The planner's cost does not grow monotonically with `e`.
- Whiteout's steady verify is 13–16× clear Akita's on 16 threads, and
  19–26× on 1 thread.

## Not verified

- Proof validity is covered only by the harness's own verify asserts. No
  fixture or tamper tests ran.
- The response plan is attributed as nested in `jolt_prover::prove`. This
  rests on span totals (`commit_pads` ≈ the plan + 40 ms), not on a trace.
- One machine (M4 Max, 16 cores), with other users logged in. The idle gate
  held for 41 of 42 runs.
- The residual share after O6 is not projected here. It needs a
  measurement.
