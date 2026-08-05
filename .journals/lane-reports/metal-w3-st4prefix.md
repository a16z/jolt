# Metal W3 st4-prefix — exact-CSR GPU prepare + round-pair fusion

## Verdict

- **Prepare: RETAIN.** Exact legacy CSR construction moves to one GPU build
  pass after a four-worker count/scan/metadata pass. It is byte-identical,
  adds no T-sized allocation, and measures −82.1% at `2^22` / −86.1% at
  `2^24` with disjoint Criterion intervals.
- **Prefix loop: NO-GO.** Fusing each bind with the next message cuts the
  seven-round prefix from 13 to 7 command buffers while preserving all 19
  dispatches, but measures only −5.7% at `2^24`; the `2^22` intervals overlap.
  Legacy remains the default. `JOLT_REGRW_FUSED=1` retains the priced probe.

## Existing seven-round schedule

The isolated prefix is the first seven cycle messages: `M0`, then six
challenge-bind/message pairs. The final cycle bind and address phase are
outside this objective.

| point | host work | command buffer | blocking boundary |
|---|---|---|---|
| round 0 | wrap Gruen levels | `M0`: `RegRwMessage*` | wait; read two partial sums + row-pair counts |
| rounds 1–6, before bind | scan preceding counts; allocate exact output CSR + offsets; expand LUT; bind Gruen | — | host CSR boundary on critical path |
| rounds 1–6, bind | — | `Bᵣ`: `RegRwBind*` + `FrBind` | wait; install entry/offset/inc swap |
| rounds 1–6, message | wrap bound Gruen levels | `Mᵣ`: `RegRwMessage*` | wait; read partial sums + next counts |

Legacy total: **13 command buffers, 13 waits, 19 dispatches**. Each paired
round has two waits. The count readback/scan/allocation sits after the prior
message wait and before the bind submission.

The candidate encodes `RegRwBind*`, `FrBind`, an explicit buffer barrier, and
`RegRwMessage*` in one command buffer. Gruen binding moves before submission.
Candidate total: **7 command buffers, 7 waits, 19 dispatches**. It removes the
intermediate wait and submission gap, but cannot move the count scan or exact
CSR allocation: both require the preceding message's data-dependent counts.

## Prepare salvage

The retained prepare path keeps the round representation frozen:

1. Four Rayon chunks count each cycle's 0–3 unique register cells and fill
   `inc` plus operand-index lanes.
2. The host exclusive-scans counts into the existing `u32` CSR offsets.
3. One `jk_reg_rw_build` thread per cycle reads the seven register SoA lanes,
   emits the same sorted `RawRwEntryIdx` cells directly at `offset[t]`, and
   initializes the same `ra`/`wa` LUT indices and zero padding.

The GPU writes the final exact-length mmap-backed CSR. No fixed segments,
sentinels, direct-field coefficients, round buffers, message/bind equations,
or transcript bytes change. `JOLT_REGRW_GPU_PREPARE=0` (also
`JOLT_REGISTERS_PREPARE_SERIAL=1`) restores the serial host builder.

### Memory gate

The GPU output replaces the serial builder's entry `Vec`; offsets, `inc`, and
the three index lanes exist in both arms. There is no additional T-sized
buffer and the final CSR has the identical entry count, so projected prepare
allocation delta at `2^27` is **0 GiB** (gate: ≤ +1 GiB).

## Isolated Criterion decision

One quiet, locked, same-window run. Every timed target iteration runs the
opposite arm immediately beforehand outside its measured duration. Production-
shaped 0/1/2/3-entry cycle rows; setup outside timing; CPU-twin/serial-CSR
oracle before timing.

| objective | size | legacy / serial | candidate | delta | decision |
|---|---:|---:|---:|---:|---|
| prepare | `2^22` | 212.90 ms `[212.53, 214.40]` | 38.017 ms `[37.608, 39.649]` | **−82.1%** | pass |
| prepare | `2^24` | 985.11 ms `[937.10, 1177.2]` | 136.46 ms `[134.46, 144.48]` | **−86.1%** | pass |
| prefix | `2^22` | 312.39 ms `[304.18, 345.20]` | 309.24 ms `[303.64, 331.66]` | −1.0% | reject: overlap |
| prefix | `2^24` | 1.1459 s | 1.0803 s | −5.7% | reject: <15% |

Prepare's conservative production-scale model uses W2B's independently cooled
`2^25` ratio (−64.6%) against its measured 2.449 s serial `2^27` budget:
**≥1.58 s stage gain**. The isolated ratios predict 2.01–2.11 s.

Prefix fusion applied to the campaign's 5.86 s `2^27` prefix gives
`5.86 × 5.7% = 0.34 s`, below both retention bars (15%, 0.5 s). The missing
prize is the still-serial data-dependent CSR scan/allocation, not command-buffer
submission count.

## Correctness and pins

- GPU-built entries, offsets, `inc`, and three index lanes equal the serial
  host builder element-for-element, including padding.
- Fused and legacy seven-message wires equal the CPU twin.
- Full kernel parity against reference covers indexed handoff, direct-field
  rounds, fused schedule, and legacy schedule.
- Prefix pins: fused `7 CB / 19 dispatches`; legacy `13 CB / 19 dispatches`.
- Full cycle pins include the prepare dispatch: `3·log_t + 1` dispatches.

No end-to-end prover run; this lane is isolated `jolt-eval` + targeted kernel
parity only.
