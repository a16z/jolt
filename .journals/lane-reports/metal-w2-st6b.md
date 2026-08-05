# W2-st6b — RA-virtualization dense adoption: defer + fuse + detach

**Status: RETAIN (isolated gate passed).** Branch `scratch/metal-w2-st6b`.
Kernel-isolated lane per Velocity v3 — no end-to-end prove runs here; the
orchestrator's battery + certification own the e2e gate.

## Root cause (fresh Aug-4 traces, no new prove runs)

Stage 6b's wall is ONE round. Per-span analysis of the campaign traces:

| 2^27 trace | st6b | `sumcheck_round[3]` | InstrRaV round-3 `begin_round` span |
|---|---:|---:|---:|
| good mode (Jul-31 canonical) | 13.87 s | 6.75 s | **1.55 s** |
| tail mode (close 2, `st6b 30.963`) | 30.96 s | 23.77 s | **17.19 s** |

Round 3 is the dense adoption: three drivers (Bool 20 polys, InstrRav 16,
RamRav 2) each run a SYNCHRONOUS `jk_ra_materialize` at `T/8` inside
`begin_round` — serialized in the engine's phase 1, before any synchronous
CPU member (BytecodeRRC ~1.0 s, IncCR ~1.6 s at that round) may start. The
2^25 CB audit shows `blocked_us ≈ 2× gpu_us` on every adoption (Bool
67/116 ms, RamRav 7/47, InstrRav 54/107): the excess wait is fresh-page
wiring of the just-allocated ping-pongs (`own_uninit_frs` → fresh
`MmapVec`, ~28.5 GiB across the three drivers at 2^27, at peak footprint).
The round-3 message then RE-READS the 8 GiB (InstrRav) it just wrote.
The 30.9 s tail mode is this block exploding 11×; W1D already excluded
compressor/pool mechanisms for the canonical mode, but the fresh-allocation
volume and the phase-1 host wait are deterministic costs either way.

## What landed

Deferred fused adoption for the RA-virtualization drivers (instruction +
RAM share `RavDriver`; Bool intentionally untouched):

- `LazyFoldedRa` gains a `PendingAdopt` state and a driver-declared
  `lazy_horizon` (default 4 = legacy). At horizon 8 the third bind stays
  lazy (a fourth lazy round runs at width 8 — the gather cost per lazy
  round is constant in T, measured cheaper than each dense-adoption step),
  and the fourth bind only doubles the tables.
- The adoption then rides the NEXT message as ONE detached command buffer:
  new kernel `jk_rav_adopt_round` gathers each poly's `(lo, hi)` pair at
  width 16, writes the dense `cur` at `T/16`, and accumulates the round's
  product-grid lanes in the same pass. No blocking wait in `begin_round`
  (launch only, ~0.5 ms), no separate materialize CB, no re-read of `cur`
  for the message, and the ping-pong allocation halves
  (InstrRav 12 → 6 GiB, RamRav 1.5 → 0.75 GiB at 2^27; write volume
  8 → 4 GiB + removed 8 GiB re-read for InstrRav).
- Every existing fallback contract is preserved fail-closed: a declined or
  failed adopt launch materializes on the CPU from the untouched
  tables/source and the SAME round recomputes host-side (`PendingAdopt`
  normalization mirrors the dense-flight rules; a bind landing directly on
  `PendingAdopt` — tiny-log_T `finish_rounds` — resolves via CPU
  materialize-then-bind).
- Wire bytes unchanged by construction (same values, same messages — the
  gather at width 16 IS the dense-after-4-binds polynomial; exact algebra),
  pinned by the lockstep parity suite.
- Ablation knob: `JOLT_RAV_DEFERRED_ADOPT=0` restores the legacy
  synchronous width-8 adoption (read per driver build).

## Isolated harness + decision (2 agreeing timed benches; 1 discarded)

`metal::st6b_bench` + `st6b_rav_microbench` example: the REAL
`OptimizedInstructionRaVirtualizationKernel` with the slot's device driver,
driven under the engine's two-phase contract at production geometry
(16 committed = 4×4, 8-bit chunks, uniform-random 128-bit lookup indices).
Correctness oracle per arm: wire round polynomials + output claims
byte-equal to the driverless CPU twin (held ✓ both arms, both sizes).

Min over 3 passes per arm, arms interleaved, same binary (runs 1 and 3;
**run 2 discarded** — a sibling lane's `st0-contention` probe was live at
99.6% CPU / load 23.8, inflating both arms 2-3×; per Velocity the third
run broke the disagreement on a quiet box):

| size | arm | total | adopt-round begin | adopt-round span | Σ begin |
|---|---|---:|---:|---:|---:|
| 2^22 | sync | 83.7-108.5 ms | 15.6-24.5 ms | 20.9-32.3 ms | 38.3-41.5 ms |
| 2^22 | deferred | **76.2-104.1 ms** | **0.4-0.6 ms** | **10.1-16.7 ms** | **18.5-19.7 ms** |
| 2^24 | sync | 255.1-272.4 ms | 63.5-64.3 ms | 85.1-86.3 ms | 85.6-98.2 ms |
| 2^24 | deferred | **236.8-248.9 ms** | **0.5-0.7 ms** | **36.4-38.2 ms** | **21.9-35.3 ms** |

Verdict at 2^24: isolated total **−7.2…−8.6%**, adoption-round
`begin_round` (the phase-1 stall every member serializes behind)
**−99%** (64.3 → 0.47 ms), adoption span **−56%**, Σbegin **−64…−74%**.
The isolated total understates production: in the batch the collect
overlaps the ~2.6 s of synchronous CPU members at that round, which the
begin-collapse now exposes.

## Production model (NOT measured here — orchestrator's gate)

At 2^27 good mode the two Rav adoptions cost ~1.55 s (InstrRav span) +
~1.4 s (RamRav member total is 1.5 s, mostly its adopt) with ~2.6 s of
phase-2 CPU available to hide behind; modeled st6b −0.5…−1.5 s. In the
30.9 s tail mode the 17.19 s spike's begin-portion leaves phase 1 entirely
and its fresh-allocation fuel halves (Rav family 13.5 → 6.75 GiB) — the
mode should compress substantially, but tail-mode reproduction is
ambient-dependent and only the wave-close cool runs can confirm.

## Retention matrix (this lane's targeted scope)

- jolt-kernels `--features metal`: **246/246** (1 known-class leaky flag),
  including new `rav_parity_adopt_decline`, `rav_parity_legacy_sync_adopt`,
  and the re-pinned dispatch counts (deferred 4+1+8, legacy 3+1+10,
  decline 4 — exact device-round accounting).
- jolt-kernels non-metal lazy-RA consumers: 21/21.
- clippy `-D warnings` with and without `metal`; fmt clean.
- Byte parity: slot lockstep suites + the harness oracle (the change is
  representation-only; no transcript/protocol surface).

## Follow-up doors (not taken, evidence attached)

1. **Bool driver extension** — the biggest remaining adopter (20 polys,
   15 GiB at 2^27) still runs the legacy synchronous width-8 adoption; the
   `PendingAdopt` machinery is generic, needs only a `jk_bool_adopt_round`
   twin + horizon override. Expected same-shape win, larger volume.
2. **Below-gate tail reclaim** — rounds 8-14 `begin_round` costs 3-7 ms
   each at 2^24 in BOTH arms (dense `take_dense` copy + host binds after
   the gate closes); small but 3 drivers × ~6 rounds.
3. IncCR::prepare / BytecodeRRC host mass is the hostgaps lane's surface,
   untouched here.
