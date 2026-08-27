# Metal wave 7 — lane D2: st0 tier-2 commit pipeline

## Verdict

**Two RETAINED cuts in one commit (`1d55e8e51` on `lane/metal-w7-tier2`,
base a4028227c): async Miller settle + flush 8192→65536.** Same-window
kill-switch ABBA @2^25: **OFF 17.27/17.26 vs ON 16.76/16.92 → −0.42 s
(−2.5%) e2e**; st0 span 5.96→4.09 s (−31%, chrome-matched runs). @2^27
(the one span profile): miller CB mass **5.5→3.50 s (−36%)**, miller_wait
**5.47→0.03 s**, tier-2 lane now has 6.1 s of input-starve slack, device
52% idle — **the tier-2 floor is GONE at every scale**. Byte-diff 20/20
(twice: with and without metal feature), metal suites 405/405, clippy
clean. Kill switches `JOLT_METAL_MILLER_ASYNC=0`,
`JOLT_METAL_MILLER_FLUSH_PAIRS=8192`.

**Premise correction (the important finding): after these cuts st0 @2^27
is DRIVER-bound — and the driver is slower than S0's model.** The wave-6
"banked driver win" mostly doesn't exist at 2^27: the fused extract
measured 10.06 s (top of S0's 8.7–10.3 contention range; trunk pre-fusion
was 8.67) and builds 6.56 (model 4.2–6.0), so the fused driver = 16.62 s
vs the pre-fusion 17.09 — **the fusion's 2^27 net is ~−0.5 s, not
−2.9..−4.4**. My cuts removed every other constraint; st0 = driver + 0.7
= 17.33. The wave-8 door is the driver, specifically `extract_bucket`'s
contention sensitivity (S0's own flagged unknown, now measured).

## Attribution (wave-6 trunk baseline, sha2-chain, CB trace + spans)

@2^25 before (chrome run, st0 5.96 s): the tier-2 lane was the pacing
consumer — recv_wait 0.22 + busy 3.28 (decode 1.23, cpu_absorb 1.11, fold
0.56, reduce_inc 0.38) + **miller_wait 2.33** ≈ 5.9 ≈ the whole window.
Chain: tier-2 blocks in miller_wait → rx_done (depth 2) fills → GPU lane
blocks in tx_done (~2.3 s unspanned) → driver send_wait 2.75. Device
queue: tier-1 2.73 + miller 2.88 CB-s = 5.61 in a 5.88 window, with
1.4 s of CB co-run overlap (miller and G1SegSum DO co-schedule — the
"GPU queue = CB sum" serial model is wrong; `gstart=` added to the jk-cb
trace to measure this).

Miller dispatch-size sweep (in-pipeline, CB trace, co-run): the indexed
fly kernel is occupancy-starved at the old 8192-pair flush.

| flush | dispatches @2^25 | miller CB total | µs/pair |
|---|---:|---:|---:|
| 8192 (trunk) | 105 | 2.87 s | 3.32 |
| 32768 | 27 | 1.91 s | 2.21 |
| 65536 | 14 | 1.73 s | 2.00 |

Rate transfers exactly to 2^27 (measured 2.02 µs/pair at 66k-pair
dispatches; trunk's 5.5 CB-s @ 3.3 µs/pair → 3.50 s). n_one_hot = 14 on
sha2-chain; tier-2 pairs = 1.83 M @2^27 (95% device share = 1.73 M).

## What landed (one commit)

1. **Async Miller settle (`InFlightMiller`).** The tier-2 lane committed
   each Miller batch then slept in `miller_wait` until the CB cleared the
   shared queue — lane-serial wait that backpressured the whole pipeline.
   Dispatches now settle one flush later: commit new CB → cpu_absorb →
   settle previous (wait ≈ 0, it had a full flush interval) → stash new.
   `DetachedPass` + owned `MillerBatch` (dory_reduce precedent); the fold
   reads partials straight from shared storage (`typed_slice`), killing
   the 25 MB/settle readback copy. Byte-identity: merge order per column
   stays dispatch order, GT merges are exact field products —
   partition/order invariant (same algebra the CPU/device split already
   relies on). Device-error recovery unchanged (batch owned in-flight →
   CPU absorb on failed wait; commit-failure returns the batch).
2. **Flush 65536.** Pure occupancy retune, table above. Grouping is
   byte-free by partition invariance. Stream-end drain batch ≤65k pairs
   (~0.13 s exposed CB at 2^27) — measured miller_drain 0.04 s.

## Floor math @2^27 (matched-grade windows; my trace's st5 15.07/st4 8.23/st6b 6.41 ≈ the wave-5 gate trace's 15.08/8.21/6.51)

Before (wave-6 trunk, components from lane R + this lane's CB receipts):
- driver: extract ~10.1 + builds ~6.6 = **~16.6 s** (fused; pre-fusion 17.09)
- tier-2 lane serial chain: busy ~13 (decode ~4.5 dilated, absorb 3.6,
  fold ~2.2, reduce_inc 2.1) + miller_wait ~5.5–6 (≈210 flush-8192
  dispatches) ≈ **~18–19 s demand** → binds above the driver
- GPU queue: 8.8 + 5.5 = 14.3 CB-s, ~12.3 timeline with overlap
- st0 ≈ max(...) ≈ **17.8–18.3** (wave-5 gate measured 17.82 with the
  slower driver + weaker tier-2 dilation)

After (measured, one 2^27 span profile, 64.22 s e2e, RSS 76.28 GiB):
- **st0 17.33 = driver 16.62 + send_wait 0.60 + drain 0.09** — driver-bound
- tier-2 lane: 11.1 busy (decode 4.52, cpu_absorb 3.57, fold 0.87,
  reduce_inc 2.13) + **6.06 s recv_wait slack**; miller_wait 0.03
- device: 11.1 CB-s (G1 7.63 + miller 3.50) in 8.30 busy (2.83 overlap),
  **8.94 s idle (52%)**
- GPU lane recv_wait 7.36 (starved by the driver, as designed)

**Modeled st0 delta @2^27 ≈ −0.5..−1.0 s wall** (17.8–18.3 → 17.33), plus
the structural unlock: tier-2 cannot re-bind until the driver drops below
~11 s, so wave-8 driver cuts land 1:1 on st0. The lane's bar (≥1.0 s) is
met only at the optimistic end — the shortfall is S0's extract eating its
own banked win, not residual tier-2 mass.

## Per-lever verdicts

| lever | verdict | numbers |
|---|---|---|
| async settle | **RETAIN** (default on) | miller_wait 2.33→0.02 @2^25, 5.47-class→0.03 @2^27; ABBA −0.42 s e2e @2^25 (with flush) |
| flush 65536 | **RETAIN** (default) | miller CB −40% @2^25 / −36% @2^27 (2.0 µs/pair) |
| cpu_share 0.05→0.0 | **KEEP 0.05** — regime-split | @2^25 pair: 15.89 (0.05) vs 16.05 (0.0) — device-paced regime pays the +0.09 CB-s. @2^27 the mechanism inverts (3.57 s contended rayon absorb vs +0.18 CB-s on a 52%-idle device): **priced door for the wave gate, one env `JOLT_METAL_MILLER_CPU_FRACTION=0.0` @2^27** |
| GPU-side fold (partials product-tree kernel) | **NOT BUILT — no longer pays** | fold fell 0.56→0.14 @2^25, 2.2→0.87 @2^27 for free (bigger batches, saturated fold grain, less contention); tier-2 lane has 6 s slack. R's 2.2 s was Θ(pairs) host Fq12 muls at ~30-task grain under contention — mechanism answered |
| TG cap on fly_indexed | not retried | kill-listed (in-pipeline inversion, W4/W5) |
| decode/reduce_inc passes | untouched | non-binding (6 s lane slack); decode dilates ~2× @2^27 vs R — bandwidth/contention, not worth chasing while starved |

## Discipline

- Timed 2^27: **0** (one span profile, allowed). Timed 2^25 decision runs:
  ABBA ×4 (kill-switch arms, 50 s cooldowns, FrBind 252.4 µs) +
  fraction pair ×2 (50 s cooldowns). Flush probes were CB-trace
  attribution diagnostics. All cargo under the wave-3 lock; every GPU run
  under the GPU lock.
- Byte oracle: `jolt-prover --features prover-fixtures[,metal]` **20/20
  both**; `metal_commit_matches_optimized` (flush=8 forces deep in-flight
  pipelining + drain, plus all-device/all-CPU arms) PASS; metal suites
  **405/405**; `clippy --all --features host -D warnings` + metal-target
  clippy on jolt-kernels clean; fmt clean.
- No kernels added (KernelId::ALL stays 82). commitment.rs 2347→2394
  lines (+47: in-flight struct + settle split; no new layers).
- `gstart=` field added to the `[jk-cb]` trace line (device-timebase CB
  start) — attribution instrumentation, retained (R-lane precedent).
- Not pushed; `scratch/metal-saturation` untouched. Worktree
  `.worktrees/metal-w7-tier2` (branch `lane/metal-w7-tier2` @
  `1d55e8e51`) ready for merge + cleanup after the wave gate.

## Doors this opens (ranked for wave 8)

1. **st0 driver `extract_bucket` @2^27: 10.06 s** (50.7%→58% of the
   driver) — S0's contention-sensitivity unknown is now the #1 door;
   collect 13.69 s co-runs and dilates it. Builds 6.56 s second.
2. `JOLT_METAL_MILLER_CPU_FRACTION=0.0` @2^27 cert experiment (above).
3. Device has 9 s idle @2^27 — any future host→GPU offload in st0 is
   free device-side now (the old "GPU queue is full" objection is dead).
