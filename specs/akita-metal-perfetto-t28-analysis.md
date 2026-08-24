# Akita Metal T=2^28 Perfetto analysis

Date: 2026-08-23

## Measurement boundary

The primary metric is the `jolt_prover::prove` span. It excludes guest build,
trace generation, and preprocessing. Each comparison uses the optimized CPU
backend and Metal backend with the same workload, padded trace domain
`T = 2^28`, and Chrome/Perfetto tracing enabled. All six traces are complete
JSON streams with balanced spans. The proofs verified.

These are single matched profiling runs, suitable for localization. A release
performance claim should use untraced repeats after the candidate is frozen.
Inclusive child-span totals can overlap across threads; root and stage wall
times below do not double-count.

Machine: Apple M4 Max, 40-core GPU, 128 GiB unified memory.

## Headline results

| Workload | Physical rows / T | CPU | Metal | Speedup | Commit span | PIOP | Eval proof |
|---|---:|---:|---:|---:|---:|---:|---:|
| BTreeMap | 94.5% | 166.55s | 56.34s | 2.96x | 65.77s -> 14.96s (4.40x) | 82.18s -> 34.46s (2.39x) | 16.73s -> 6.89s (2.43x) |
| Fibonacci | 75.0% | 215.18s | 45.72s | 4.71x | 112.32s -> 22.47s (5.00x) | 77.02s -> 16.72s (4.61x) | 23.85s -> 6.49s (3.67x) |
| SHA-2 chain | 50.4% | 213.70s | 42.45s | 5.03x | 107.83s -> 12.59s (8.57x) | 79.12s -> 23.38s (3.38x) | 24.88s -> 6.45s (3.86x) |

The Metal wall-time composition is workload-dependent:

| Workload | Commit | PIOP | Eval proof |
|---|---:|---:|---:|
| BTreeMap | 26.6% | 61.2% | 12.2% |
| Fibonacci | 49.1% | 36.6% | 14.2% |
| SHA-2 chain | 29.7% | 55.1% | 15.2% |

The underlying Akita commit span is just below 5x on the BTreeMap trace and
essentially exactly 5x on Fibonacci. The wider Stage 0 span, which includes
assembly around commit, is 4.52x, 5.08x, and 8.71x respectively.

## PIOP stage wall times

Each cell is `CPU -> Metal (speedup)`.

| Stage | BTreeMap | Fibonacci | SHA-2 chain |
|---|---:|---:|---:|
| Stage 1 | 12.87s -> 4.34s (2.97x) | 12.98s -> 3.89s (3.34x) | 12.77s -> 4.46s (2.87x) |
| Stage 2 | 8.01s -> 13.00s (0.62x) | 4.24s -> 1.08s (3.91x) | 4.82s -> 4.69s (1.03x) |
| Stage 3 | 3.70s -> 0.91s (4.05x) | 3.93s -> 0.81s (4.85x) | 3.94s -> 0.86s (4.58x) |
| Stage 4 | 3.92s -> 5.16s (0.76x) | 4.11s -> 3.74s (1.10x) | 4.29s -> 3.40s (1.26x) |
| Stage 5 | 14.91s -> 2.51s (5.95x) | 15.90s -> 2.26s (7.03x) | 18.63s -> 2.12s (8.78x) |
| Stage 6a | 6.25s -> 0.71s (8.83x) | 4.44s -> 0.48s (9.30x) | 4.35s -> 1.84s (2.36x) |
| Stage 6b | 30.20s -> 7.26s (4.16x) | 28.97s -> 4.05s (7.15x) | 27.41s -> 5.56s (4.93x) |
| Stage 7 | 2.32s -> 0.58s (4.01x) | 2.45s -> 0.40s (6.08x) | 2.91s -> 0.46s (6.33x) |

### BTreeMap: high-activity RAM misses the sparse route

BTreeMap has `RAM log_K = 19` (`K = 524,288`). The trace records:

- `MetalRamCycleFamily::owner_prepare`: 1.45s, but no completed publication;
- `MetalRamReadWrite::route`: `selected = optimized_cpu`,
  `fallback_reason = missing_owner`;
- `MetalRamHammingBooleanity::route`: the same fallback;
- Stage 2 `RamReadWriteChecking::prepare`: 3.26s versus 0.59s CPU;
- Stage 2 `RamReadWriteChecking::prove_round`: 5.60s versus 2.98s CPU;
- Stage 6b `RamRaVirtualization::prove_round`: 2.51s versus 1.94s CPU.

The cause is deterministic in the implementation. `RamAccessTape` retains at
most `2^18` accesses. BTreeMap exceeds that cap, so
`shared_ram_cycle_family_owner` returns `None`; the trace establishes the
lower bound on activity but does not record the exact rejected access count.
This is a representation mismatch, not a lack-of-occupancy problem. The
sparse host owner is appropriate for Fibonacci's low-activity RAM, but a
high-activity T=2^28 trace needs a chunked/dense GPU route.

Stage 4's apparent BTreeMap regression is in the optimized CPU register
read/write kernel. The Metal backend also starts the Stage 5 compatibility
scatter asynchronously at this boundary. The trace covers the final 80ms
address prefetch but not the preceding scatter construction on its worker
thread, so some Stage 5 work and memory contention are shifted into Stage 4.
Instrumenting that construction is required before treating the full 1.23s
delta as a register-kernel regression.

### SHA-2: bytecode K=2^14 uses a narrow hybrid fallback

SHA-2 has `bytecode log_K = 14`; the fused address-major carrier is currently
specialized to `log_K = 13`. The production route now admits the carrier only
for the supported domain and sends only the bytecode address pushforward to
the optimized CPU kernel. The trace records
`fallback_reason = address_domain`; the fallback occupies 1.34s of Stage 6a.

This route does not change the protocol or verifier. The T=2^28 hybrid proof
verified and is already 5.03x end to end. Even a perfect generalized carrier
can save at most the observed 1.34s here, moving the same trace to roughly
41.1s or 5.20x before second-order effects.

## Eval-proof floor

Metal eval proof is nearly workload-independent at 6.45--6.89s, while the CPU
cost changes substantially. Its additive direct-child breakdown is:

| Component | BTreeMap | Fibonacci | SHA-2 chain |
|---|---:|---:|---:|
| Seven `RingRelationProver::new` calls | 3.43s | 3.08s | 3.21s |
| Trace one-hot coefficient packing | 1.17s | 1.09s | 1.03s |
| Stage 1 + Stage 2 sumchecks | 1.34s | 1.51s | 1.35s |
| Remaining ring switch, commitments, and host work | 0.95s | 0.81s | 0.86s |

Within ring-relation construction, the resident Metal decompose/fold path is
2.37--2.72s. Opening telemetry reports about 49 GiB of deferred index state.
The opening command wall time is 5.09--5.54s, but measured GPU-active time is
only 2.44--2.69s. Thus eval proof is not GPU-throughput-saturated: host
preparation, materialization, command gaps, and synchronization occupy roughly
half the command interval and about 60% of the full eval span.

For a literal 5x eval-proof requirement, the current per-workload targets are:

| Workload | Current Metal | CPU / 5 target | Required saving |
|---|---:|---:|---:|
| BTreeMap | 6.89s | 3.35s | 3.55s (51%) |
| Fibonacci | 6.49s | 4.77s | 1.72s (27%) |
| SHA-2 chain | 6.45s | 4.98s | 1.48s (23%) |

Fusing or streaming the deferred index/decompose and coefficient-packing work
is enough in principle for Fibonacci and SHA-2. BTreeMap also needs to reduce
the seven sequential ring-relation constructions and/or batch more of their
sumcheck work; its faster CPU baseline makes the same fixed Metal floor much
harder to beat by 5x.

## Commit characteristics

Commit is the opposite of eval proof. On Fibonacci and SHA-2, command-wall
time and GPU-active time differ by less than 0.3%, so launch latency and idle
gaps are not the primary limit. The SHA-2 trace reports about 975 GB of logical
matrix reads over 10.87s GPU-active time. Further commit gains therefore need
less logical traffic/work, better locality, or more useful CPU/GPU overlap;
simply dispatching more threads will not move the roof.

Metal commit cost tracks the number and distribution of hot one-hot entries,
not padded T alone. The three T=2^28 cases have 2.02B (BTreeMap), 3.26B
(Fibonacci), and 2.23B (SHA-2) hot entries, with 30, 29, and 29 columns.
Fibonacci consequently spends 22.47s in commit despite having fewer physical
trace rows than BTreeMap.

## Memory behavior

Reported process peaks remain around 78--80.5 GiB: 80.11 GiB for Metal
BTreeMap, 78.86/80.46 GiB CPU/Metal for Fibonacci, and 79.68/77.51 GiB for
SHA-2. The T=2^28 bytecode-cycle plan reports about 19--20 GiB currently
resident plus 16.1 GiB planned, versus 115.4 GiB for the unconstrained fully
resident schedule. The hybrid lifetime plan is therefore doing real work: it
keeps the proof below physical memory without a meaningful CPU tail
(`253,696` work units).

The 49 GiB deferred opening index is now the clearest large, short-lived
allocation. Streaming it directly into the opening/decompose pipeline is the
best joint latency-and-memory target.

## Scaling observation

Fresh endpoint traces show the occupancy transition:

| Workload | T=2^20 | T=2^25 | T=2^28 |
|---|---:|---:|---:|
| BTreeMap | 1.09x | 3.52x | 2.96x |
| Fibonacci | 1.08x | 4.49x | 4.71x |
| SHA-2 chain | 1.08x | 4.68x | 5.03x |

Large T removes the fixed dispatch/occupancy handicap, but BTreeMap reverses
direction because its large, high-activity RAM instance falls out of the
sparse route. Higher occupancy cannot compensate for selecting the wrong
representation.

## Numerical path forward

1. **High-activity RAM route.** Record the rejected access count and exact
   owner-rejection reason. Avoid the late sparse attempt once activity exceeds
   the cap. Then add a chunked cycle-major Metal path for the first `log_T`
   rounds and a GPU scatter/reduction into the `K = 2^19` address tail. Reuse
   the Stage 1 RAM-access projection so the path does not rescan the witness.
   This is the largest BTreeMap opportunity.
2. **Eval index/decompose fusion.** Stream opening-index tiles into
   decompose/fold and coefficient packing instead of materializing roughly
   49 GiB, retain the root buffers across the seven ring relations, and batch
   command buffers to remove host waits. A first bar is 1.8s saved; that is
   enough for 5x eval proof on Fibonacci and SHA-2. BTreeMap requires about
   3.55s and likely a second step that batches the seven ring relations under
   a transcript-derived combination challenge.
3. **BTreeMap commit cleanup.** The traced Akita commit needs about 1.8s (12%)
   to reach a literal 5x on this workload. Since the command is saturated,
   target logical matrix traffic and per-column work rather than occupancy.
4. **Generalize the bytecode carrier to log_K=14.** This removes a known 1.34s
   SHA-only CPU island and should take SHA-2 from about 5.03x to at most about
   5.20x. It is bounded and lower priority than high-activity RAM and eval
   proof.
5. **Trace the prefetch gap.** Add a span around compatibility-scatter
   construction and compare overlap versus post-Stage-4 scheduling on
   BTreeMap. Keep the overlap only if Stage 4 + Stage 5 combined wall time
   improves.

At 5x end-to-end, the allowed Metal walls are 33.31s BTreeMap, 43.04s
Fibonacci, and 42.74s SHA-2. SHA-2 clears the bar now; Fibonacci needs 2.68s;
BTreeMap needs 23.03s. Therefore the next campaign should be framed around
the high-activity RAM route and eval-proof floor, not another general
occupancy sweep.

## Trace artifacts

- `benchmark-runs/perfetto_traces/akita_btreemap_28_optimized.json`
- `benchmark-runs/perfetto_traces/akita_btreemap_28_metal.json`
- `benchmark-runs/perfetto_traces/akita_fibonacci_28_optimized.json`
- `benchmark-runs/perfetto_traces/akita_fibonacci_28_metal.json`
- `benchmark-runs/perfetto_traces/akita_sha2_chain_28_optimized.json`
- `benchmark-runs/perfetto_traces/akita_sha2_chain_28_metal.json`

Open any artifact in `https://ui.perfetto.dev/`. These are host-side tracing
spans plus route metadata. GPU-active durations above come from Metal command
timestamps printed by the benchmark; native shader occupancy and cache
counters require a Metal System Trace/Instruments capture.
