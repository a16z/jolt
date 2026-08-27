# Metal saturation: artifact-verified attribution

## Verdict

The closing tree is **not globally launch-bound and not continuously GPU
occupied**. It has three distinct regimes:

1. `st0`/`st8`: device-heavy endpoints. `st0` shares the memory fabric with
   the hoisted record walk; `st8` is Miller/Dory compute with occupancy/ALU as
   the likely local roof.
2. `st3`/`st4`/`st7`: host/device alternation or host preparation. Their exact
   sampled-zero runs reject the closed journal's `NONE >1 s` claim.
3. `st5`/`st6b`: large device streams. `st5` sustains activity; `st6b` mixes
   ordered-queue waits, host preparation, and bandwidth-heavy members.

**Mental model:** the remaining middle-stage wall is chiefly host serialization
plus shared-memory traffic, not missing GPU dispatches. The next winning shapes
remove host readback/allocation boundaries and bytes moved; “port more work” is
not a sufficient strategy. Exact ALU-vs-SLC-vs-DRAM-vs-occupancy attribution is
blocked on hardware counters that the current traces do not contain.

No protocol change was made. This report and all proposed instrumentation are
protocol-neutral.

## Artifact ledger

Audited tree: `88b063db35709bef0e813aea7cdb3980aeddab23`.

| id | artifact | bytes | SHA-256 |
|---|---|---:|---|
| A1 | sibling `.journals/artifacts/baseline-2to25-20260804.log` | 1,994 | `b351e6bf4f439af281278f658131935f5cf7b21fea46c32e837f70f5ef677ea8` |
| A2 | sibling `.journals/artifacts/baseline-2to27-20260804.log` | 1,995 | `e8d232f4db9e432f38402e62221108aaf8546cba5d27d13cbf19ffb347a98b0c` |
| A3 | sibling `.journals/artifacts/monitor-2to27-20260804.json` | 1,619,093 | `3a94bc3f053108b7bba4e0df4e497fe880f8f2712dbaeceffab3f2da8b88164c` |
| A4 | sibling `.journals/artifacts/monitor-2to27-20260804.log` | 2,090 | `cd0d4889358ec1c24f2a0b3a779f3182c30038926c63110bbb797425201fa050` |
| A5 | sibling `.journals/artifacts/cbtrace-monitor-2to25-20260804.json` | 1,151,857 | `c0aefa91ea84a04aeb928471eb3eb23e5c1194dcaa8abec0fa8375a83eaefece` |
| A6 | sibling `.journals/artifacts/cbtrace-monitor-2to25-20260804.log` | 61,489 | `34c19817136ca13269fe68d288b89d8e27e7c9c983d12fb5f4a8ccb2a52a4e02` |
| A7 | sibling `.journals/artifacts/metal-microbench-20260804.log` | 2,000 | `6b737d0b2991f4520f31d0d1a9f7ff7eb9f430f783f483c86443feb32a86f587` |
| A8 | sibling `.journals/artifacts/miller-microbench-20260804.log` | 2,341 | `95c1045d848237cb399ebfb97274b72f43fb41aac255cf4e80c6fb57b00a3fd6` |
| A9 | archived `metal-m5-gputrace-2to25-20260803.json.gz` | 71,292 | `0bff05560ae4b1b1496aaba1c5dfbc777c6f38a3fd1865ed2168e06ce4e76046` |
| A10 | archived `metal-m5-gputrace-2to27-20260803.json.gz` | 125,259 | `956d744efd2d8a7efed97a60410ca15a2624feffa02739865b46c488f99f5b0a` |
| A11 | `/tmp/gpu-util-trace-2to27-final-20260804.json.gz` | 102,645 | `15a45931ff2c1a5a90169739b693422d4e86916cfa19f5708bbca3e7415f49e8` |

The requested fresh non-monitor `benchmark-runs/perfetto_traces/
modular_sha2_chain_{25,27}_metal.json` files were absent from the sibling
worktree at audit time. A1/A2 verify total prover wall and memory, but not the
fresh non-monitor stage vectors. Those vectors must not be treated as
artifact-verified until the trace files are restored or re-captured.

| run | prover wall | peak RSS | peak footprint |
|---|---:|---:|---:|
| A1 non-monitor `2^25` | 19.67 s | 27.42 GiB | 26.78 GiB |
| A2 non-monitor `2^27` | 71.77 s | 76.87 GiB | 75.39 GiB |
| A4 monitor `2^27` | 80.99 s | 76.83 GiB | 75.48 GiB |
| A5/A6 monitor+CB `2^25` | 19.72 s | 25.17 GiB | 24.59 GiB |

The 9.22 s `2^27` monitor/non-monitor difference is an observed cross-run
delta, not isolated monitor overhead; the `2^25` difference is 0.05 s.

## Derivation contract

- Span walls: pair Chrome `B`/`E` records with a per-`(pid, tid)` stack;
  durations are inclusive and therefore not additive across nested spans.
- Counter means: left-hold integration over each stage,
  `sum(value_i * overlap([t_i,t_{i+1}), stage)) / stage_wall`.
- Sampled-zero run: consecutive `gpu_percent == 0` samples, held until the
  next nonzero sample. This is an `ioreg` observation interval, not a hardware
  proof that every intervening cycle was idle.
- The source-default 100 ms sampler produced 350 A3 prove-window samples with
  115.974 ms minimum, 235.089 ms mean, and 1.298991 s maximum inter-sample
  gaps. Missing time is never rewritten as zero.
- CB fields are diagnostic only. `gpu_us` includes ordered-queue age in the
  current runtime contract; `blocked_us` can overlap across batch members.
  Neither sum is stage wall.

## Current `2^27` stage evidence

All wall/counter values below derive from A3. `GPU`/`CPU`/`cores` are
time-weighted left-hold values. `zero` is total held-zero overlap; `max-zero`
is the longest held-zero overlap inside the stage. Small stages are vulnerable
to boundary carry: `st6a` has one in-stage GPU sample and it is zero despite
the displayed 36.0% left-hold mean.

| stage | wall s | GPU % | zero s | max-zero s | cores | dominant inclusive spans | attribution | confidence |
|---|---:|---:|---:|---:|---:|---|---|---|
| st0 | 18.012 | 79.4 | 0.570 | 0.570 | 11.1 | `stream_witnesses` 17.370; background `TraceRecord::collect` 13.337 | GPU commit plus CPU record walk contend on shared memory fabric | high |
| st1 | 4.697 | 77.1 | 0.880 | 0.396 | 2.2 | `Stage1Batch` 2.696; outer prepare 1.994; remainder prepare 1.659; 2.001 outside batch span | low-core host preparation/glue; `ioreg` mean is contaminated by adjacent device work | medium |
| st2 | 2.958 | 48.3 | 1.346 | 0.595 | 3.3 | `Stage2Batch` 2.416; batch rounds 1.010; prepares 0.538/0.501/0.392/0.317 | mixed host preparation and device rounds | high |
| st3 | 2.996 | 16.1 | 2.480 | 2.480 | 6.7 | `InstructionInput` rounds 2.013; `SpartanShift::prepare` 0.657 | host round 0 plus dense device write; not continuously occupied | high |
| st4 | 9.278 | 40.2 | 4.761 | 2.584 | 4.0 | rounds 6.684; register rounds 5.149; register prepare 2.449; RamVal rounds 1.535 | alternating device message/bind with host count scan/allocation; some streaming roof | high |
| st5 | 14.669 | 77.6 | 2.138 | 0.688 | 4.5 | batch 12.756; InstructionReadRaf rounds 9.709; RegistersVal rounds 3.016 | device-dominated scan/reduce; ALU-vs-bandwidth unresolved | medium |
| st6a | 0.245 | 36.0 | 0.013 | 0.013 | 8.1 | Bytecode address prepare 0.225 | small host prepare; below campaign-scale priority | high |
| st6b | 17.491 | 34.5 | 10.600 | 2.234 | 3.3 | batch 15.205; BRRC 5.647; RamRA 2.898; IncCR 2.749; InstrRA 2.620; Inc prepare 1.790; eq 1.076 | bandwidth/queue-heavy device batch plus host prepare; not launch-bound | high for shape, medium for exact roof |
| st7 | 1.893 | 13.5 | 1.615 | 1.615 | 11.8 | `HammingWeightClaimReduction::prepare` 1.887; rounds 0.002 | CPU pushforward/table construction; device rounds irrelevant | high |
| st8 | 8.699 | 86.6 | 0.765 | 0.497 | 1.8 | Dory open 7.900; Miller device spans dominate nested work | device ALU/occupancy candidate, not shared-bandwidth candidate | medium |

Source anatomy agrees with the spans:

- `instruction_input.rs`: round 0 calls host `native_q_evals`; later rounds are
  one fused Metal dispatch/CB/wait and write eight dense tables.
- `registers_read_write.rs`: every cycle round performs a device message,
  host `scanned_offsets()` plus allocation, then a device bind.
- `hamming_weight_claim_reduction.rs`: `build_hamming_weight_tables` remains an
  `O(T)` CPU bundle walk; only the tiny rounds use Metal.

## Closed-ledger correction

A11 and A3 independently refute `zero-GPU windows >1.0 s: NONE`.

| trace | st3 longest | st4 longest | st6b longest | st7 boundary run |
|---|---:|---:|---:|---:|
| prior final A11 | 2.115 s | 2.696 s | 2.403 s | 1.951 s into st8 |
| fresh A3 | 2.480 s | 2.584 s | 2.234 s | 2.112 s into st8 |

The four affected regions are reproducible; the exact prior-final st7-boundary
hold is 1.951 s, not strictly greater than 2 s. The defensible close statement
is: **no multi-second host mass remained unclassified**, not “no sampled-zero
window remained.”

For context, archived opening trace A10 reports stage GPU means of
`91.3, 30.9, 25.4, 16.4, 18.7, 60.2, 0.0, 24.3, 0.0, 84.4%` for
`st0..st8`; the current tree removes major wall, but the low-activity topology
of `st3/st4/st6b/st7` remains.

## What the existing evidence distinguishes

| hypothesis | evidence for/against | verdict |
|---|---|---|
| Fixed launch cost | A6 has 646 CBs/1,073 dispatches. A7 empty-CB round trip is 133.8 us and batched dispatch is 2.58 us. `646 * 133.8 us = 86.4 ms`, 0.44% of the 19.72 s A5 proof. | not a global roof; only tail-round noise |
| Synchronization/host boundaries | A6 stage CB counts are st3 9, st4 51, st6b 70; source shows synchronous waits and st4 host scan/allocation between passes. | material in st4/st6b, but CB sums cannot quantify wall yet |
| Shared DRAM/SLC fabric | A7 device bind falls 357.2→162.4 GB/s (−55%) while concurrent CPU work falls 94.1→52.2 GB/s (−45%). A3 st0 overlaps a 13.337 s record walk with device commit. | first-order st0/st6b mechanism; DRAM vs SLC unresolved |
| ALU roof | A7 chained Montgomery work reaches 11.30 Gmul/s. A8 tower kernels deliver 3.25–4.18 Gmul-equivalent/s. | plausible for Miller-heavy st8; no per-kernel ALU counter |
| Occupancy/register pressure | A8 Miller per-pair cost improves sharply through 4,096–8,192 exposed threads; device Miller is unchanged 82.4→82.1 ms under CPU ALU soak. | local Miller concern, not proof-wide cause |
| Serial/parallel host work | A3 exact spans and low global core equivalents expose st1/st2 glue, st4 boundaries, and st7 prepare; st7 is 99.7% prepare. | proven stage-local cause |

The 86.4 ms figure is an empty-CB-equivalent scale estimate, not a strict
upper bound on encoding, queueing, or synchronization. The seconds-scale
round gaps must be attributed with aligned host/GPU timestamps.

## Counter surface and gaps

### Available now

- MetricsMonitor: process physical memory; global CPU%; derived global active
  cores; active-core count; macOS `thread_count=0`; `ioreg` device and renderer
  utilization. Renderer utilization is zero for essentially all compute work
  and is unusable here.
- CB trace: host-relative commit time, CB `GPUStartTime`→`GPUEndTime` age,
  caller blocked time, dispatch count/mix, logical thread maximum.
- Metal SDK common sets: timestamp; stage-utilization (`totalCycles` plus
  graphics-stage cycles, no compute-cycle field); statistics
  (`computeKernelInvocations`). Device-specific `counterSets` are runtime
  enumerated and have not been queried.
- Local system tools: `powermetrics` advertises GPU power/residency and
  bandwidth channels; `/usr/bin/xctrace` exists.

### Missing or unverified

- SIMD/ALU active cycles, issued instructions, stall reasons, resident
  threadgroups/waves, register-limited occupancy.
- SLC/L2 hits/misses/bytes and DRAM read/write bytes or bandwidth.
- Per-dispatch clock/frequency, power, and thermal residency.
- Absolute host commit/wait and GPU start/end timestamps in one clock domain.
- Process CPU utilization and runnable/blocked state; MetricsMonitor CPU is
  system-global.
- Exact Metal/System Trace templates: the selected developer directory is
  CommandLineTools and no Xcode/Instruments application bundle is present.
  Per the phase ban, `xctrace list templates`, `powermetrics -h`, and device
  counter enumeration were not executed.

## Ranked follow-ups

1. **Counter inventory + clock alignment.** Add an env-gated inventory of
   `MTLDevice.counterSets`, counter names, simultaneous-set failures, and
   dispatch-boundary support. Extend CB trace with monotonic commit/wait
   timestamps and CPU↔GPU timestamp calibration.
2. **st4 remove host round boundaries.** Preserve the legacy proof relation;
   replace message→host count scan/allocate→bind with bounded reusable storage
   and device scan/compaction. This is the memory-viable middle between W2B's
   fast/fat and lean/slow variants.
3. **st6b reduce bytes before more parallelism.** Counter BRRC/RamRA/IncCR/
   InstrRA separately; tile/reuse eq and row materializations through SLC and
   avoid concurrent CPU streams when DRAM is saturated.
4. **st0 schedule the hoisted walk against commit.** Test chunked yielding or
   delayed spawn; do not repeat pool-width/QoS probes already closed negative.
5. **st7 reuse/sparsify the pushforward build.** Do not retry generic device
   scatter (closed negative); target fewer rows/bytes or earlier reuse of the
   already shared instruction rows.

## Exact capture plan

### Inventory; no timed benchmark

1. Run `xctrace list templates`, `powermetrics -h`, and an env-gated Metal
   counter enumerator. Save stdout/stderr, OS/build version, device name, and
   SHA-256.
2. Require dispatch-boundary sampling and at least one counter set covering
   compute activity. If only common timestamp/statistic sets exist, classify
   ALU/occupancy/cache as unavailable and do not manufacture a roof verdict.
3. If more than two mutually exclusive hardware-counter passes are required,
   retain only the two sets that answer (a) ALU/occupancy and (b) cache/DRAM;
   otherwise stop and use System Trace/powermetrics.

### Fresh run A: activity/occupancy set

One `2^24` run of the prebuilt monitor binary, under the existing bench lock:

```text
MONITOR_INTERVAL=0.1 \
JOLT_METAL_CB_TRACE=1 \
JOLT_METAL_COUNTER_TRACE=<activity-set> \
<monitor-binary> --name sha2-chain --scale 24 --format chrome --backend metal
```

Capture Chrome JSON, CB JSONL, counter inventory, command line, environment,
binary SHA-256, `pmset -g batt`, and stage-memory log.

### Fresh run B: memory/cache set

Same binary/arguments, changing only
`JOLT_METAL_COUNTER_TRACE=<memory-set>`. This is the second and final timed
bench for the roof-classification decision.

In parallel with each proof only if inventory confirms availability, run
`powermetrics` at 100 ms in plist mode with GPU-power and bandwidth samplers;
its start/stop timestamps must use the same monotonic-clock calibration. A
System Trace capture replaces, rather than adds to, a timed arm.

### Postprocessing

1. Hash raw files before parsing; reject truncated JSON or unmatched `B/E`.
2. Emit left-hold and raw-sample statistics; flag any counter gap over 500 ms.
3. Join CB sequence to spans using absolute monotonic timestamps, never line
   completion order.
4. Delta start/end hardware counters per dispatch; preserve raw counter names,
   units, error sentinels, logical threads, threadgroup width, and kernel ID.
5. Report stage wall, unioned GPU-active intervals, host runnable/blocked wall,
   queue delay, execution wall, ALU/activity ratio, occupancy, cache bytes,
   DRAM bytes, and bytes/s only where the captured counter defines them.

## Bounded decision matrix

| decision | arm A | arm B | discriminator | cap |
|---|---|---|---|---:|
| Roof map | `2^24` activity/occupancy counters | `2^24` cache/DRAM counters | per-kernel ALU/occupancy vs bytes/cycle | 2 |
| st0 contention | default hoist | same binary `JOLT_RECORD_HOIST=off` | GPU bandwidth/residency and CPU rate move inversely with overlap | 2 |
| st4 boundary removal | trunk | bounded-storage device-scan prototype | host blocked gaps collapse without footprint growth or lower device throughput | 2 |
| st6b cross-member contention | default | same binary BRRC CPU arm via `JOLT_METAL_MIN_TERMS_BYTECODE_READ_RAF_CYCLE=999999999999999` | other members speed up only when BRRC leaves the device queue/fabric | 2 |
| st7 structural candidate | trunk | sparse/reuse prototype | prepare bytes and wall fall; rounds remain unchanged | 2 |

All iteration arms are `2^22..2^24`. A disagreeing pair does not authorize a
third run automatically; document the disagreement and ask the orchestrator.
No full suite runs until an integrated retained change exists.

## Report-ready schema

```text
run:
  run_id, commit, binary_sha256, scale, backend, protocol_config,
  os_build, device, command, env, started_monotonic_ns,
  monitor_enabled, cbtrace_enabled, counter_sets, raw_artifact_sha256[]

stage:
  run_id, stage, start_us, end_us, wall_s,
  gpu_ioreg_tw_pct, gpu_samples, gpu_zero_samples,
  gpu_zero_hold_total_s, gpu_zero_hold_max_s, max_sample_gap_s,
  cpu_global_tw_pct, cores_global_tw,
  rss_open_gib, rss_close_gib, footprint_open_gib, footprint_close_gib

span:
  run_id, stage, span_name, count, inclusive_s, max_s

command_buffer:
  run_id, stage, cb_seq, host_commit_ns, wait_start_ns, wait_end_ns,
  gpu_start_ns, gpu_end_ns, queue_delay_us, execution_us,
  dispatch_count, kernel_mix, logical_threads

dispatch_counter:
  run_id, cb_seq, dispatch_seq, kernel, logical_threads, tg_width,
  counter_set, counter_name, unit, start_value, end_value, delta,
  valid, error

classification:
  run_id, stage, primary_roof, secondary_roof, confidence,
  evidence_ids[], caveats[], next_decision
```

Allowed `primary_roof` values:
`alu`, `dram_bandwidth`, `slc_bandwidth`, `occupancy`, `serial_host`,
`launch_sync`, `mixed`, `unknown`. `dram_bandwidth` and `slc_bandwidth` require
distinct captured counters; otherwise report `mixed` or `unknown`.

## Soundness boundary

Instrumentation, scheduling, storage reuse, tiling, and exact table-layout
changes are protocol-neutral if they preserve every prover message and opening.
Any change to round grouping, polynomial extension, transcript absorption,
challenge timing, claimed relation, or verifier opening semantics needs a
separate soundness argument before implementation and must eventually pass
end-to-end accept, tamper reject, and the full integrated suite.
