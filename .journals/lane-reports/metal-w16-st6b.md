# W16 st6b — re-attribution + sync-member detach; SLC-tiling door CLOSED

**Verdict:** RETAIN (detach + round-0 prelaunch, default-on, kill switch
`JOLT_ST6B_DETACH=0`), measured st6b **−0.112 s @2^25** (1.180 → 1.068,
−9.5%), modeled **−0.9..−1.4 s @2^27** (structural arithmetic below — the
reclaimable mass is 2^27-heavy; the wave-gate ABBA is the honest 2^27
number). **Parked door #3 (st6b SLC tiling) CLOSED permanently** — the
bandwidth roof shows the gather mass runs at 10–22 GB/s vs a ~400 GB/s
DRAM roof (ALU/latency-bound, not DRAM-bound), and the one kernel that IS
at the roof (IncRound, 370 GB/s) streams once with zero cross-round reuse.
Nothing for the SLC to cache, nothing DRAM-starved to feed.

## 1. Re-attribution @2^27 (one instrumented profile, FrBind 252.6 µs)

Engine-level `m_begin`/`m_collect` member spans added to `prove_batch`
(2-line change in jolt-sumcheck; the engine comment always promised
member spans — metal slots never emitted them). Traced 2^27 run: prove
36.27 s (clean window; profile, not a cert), st6b 4.715 s:

| slice | s | note |
|---|---:|---|
| **BytecodeReadRafCycle rounds** | **2.93** | r0-r5 = 941/551/388/350/439/199 ms; sync `pass.run()` per round — the wall is its blocked wait **behind the whole GPU queue** (Bool+2×RAV detached CBs commit first) |
| prepares (outside round loop) | 0.89 | IncCR 0.356 (RdInc 0.098 + RamInc 0.255 oracle walks) · BytecodeInit 0.292 · RamHB 0.107 · JointOpening 0.121 (∥) · RamRaV 0.035 — serial host, GPU idle |
| IncClaimReduction rounds | 0.40 | r0 277 ms (first-touch wiring of 4×2^27 tables) then 55/30/14… |
| RamHammingBooleanity rounds | 0.26 | r0 143 ms, same shape |
| RamRaVirt / Bool / InstrRaVirt walls | 0.17/0.03/0.03 | already detached — their exec hides inside bytecode's blocked time |

The w2/w3-era mental model ("Bool/RAV gathers are the st6b wall") is
obsolete on the wall: since w2's deferred adoption those CBs are detached
and their exec is BILLED to the bytecode member's synchronous queue wait.
The stage = one sync-wait chain + serial host.

## 2. Bandwidth roof (CB traces @2^25, `JOLT_METAL_CB_TRACE=1`)

st6b kernel CBs @2^25: union GPU busy **728 ms**; blocked (sync waits):
bytecode 731 ms (lazy 460 + dense 199 + adopt 71), IncRound 122, Hamming 95.

| kernel (r0 unless noted) | exec window | bytes touched | eff GB/s | ~Gmul/s | verdict |
|---|---:|---:|---:|---:|---|
| BoolLazyRound | 97.3 ms | rows 1.61 GB | 16.6 | 7.1 | ALU-roofed |
| RavLazyRound (Instr, 16 polys) | 132.6 ms | rows 1.61 GB | 12.1 | ~11.8 | ALU-roofed |
| RavLazyRound (Ram, 2 polys) | ≤87.8 ms (co-run) | rows 1.61 GB | ≥18 | ~1.7 | base-cost-bound (rows+eq+tg_sum ≈ the 16-poly base) |
| BytecodeLazyRound | 60–132 ms (co-run) | rows 0.27 + cur 1.07 GB | 10–22 | 1.7–3.6 | overhead-bound (16-factor thread arrays, runtime loops) |
| IncRound | 11.6 ms | 4 tables, 4.29 GB | **370** | — | **AT the DRAM roof**, zero reuse |
| HammingRound | 8.8 ms | 1.07 GB | 122 | — | partly roofed, tiny |

DRAM roof M5 Max ≈ 400–550 GB/s; SLC 48 MB. Gather working set already
cache-resident (branch tables ≤2.5 MB/driver); the streamed rows (1.6 GB
@2^25, 6.4 GB @2^27) have no cross-round reuse that fits 48 MB, and the
kernels reading them sit at 3–6% of the DRAM roof anyway. **SLC tiling has
no prize on either side — door closed with receipts.** (Windows overlap —
Metal co-runs independent CBs; per-kernel numbers are window-bounds, the
union is exact.)

## 3. The cut: st6b sync-member detach + round-0 prelaunch

Mechanism (scheduling only, zero new kernels, `KernelId::ALL` stays 89):

- **Detach** — the three synchronous device members (bytecode cycle,
  IncCR, RamHB) get the Bool/RAV two-phase treatment: `begin_round`
  commits a `DetachedPass` flight, `collect_round` waits. BytecodeDriver
  implements the already-plumbed `launch_lazy`/`launch_dense`/
  `collect_lanes` seam of `LazyRaDevice`; Inc/RamHB carry their own
  flights. One queue drain per round replaces three serial blocked waits.
  Failure contracts mirror the sync arms exactly (lazy declines stateless;
  dense declines publish the combined recovery; wait failures latch off +
  recover host-side from intact `cur`).
- **Round-0 prelaunch** — round 0 is bind-free, so its lanes are fully
  determined at prepare end. The four metal slot fronts (bytecode, Bool,
  InstrRAV, RamRAV) launch round 0 at prepare return; the engine's round-0
  `begin_round` adopts the flight idempotently (`bind.is_none() &&
  launched` guard). The GPU works under the stage's ~0.8 s of serial host
  prepares instead of idling.

Byte parity by construction (same kernels, same buffers, same values, only
commit/wait timing moves); pinned by slot lockstep suites both arms + both
ratchet arms (below). One kill switch: `JOLT_ST6B_DETACH=0` restores every
sync path bit-exactly.

## 4. Numbers

Same-window pairs, 2^24 then 2^25 (detach-only ON, then +prelaunch ON2):

| scale | arm | st6b | prelude | round loop | e2e wall |
|---|---|---:|---:|---:|---:|
| 2^24 | OFF | 0.961 | — | 0.846 | 6.26 |
| 2^24 | ON (detach) | 1.003 | — | 0.877 | 6.39 |
| 2^25 | OFF | 1.180 | 0.198 | 0.983 | 10.99 |
| 2^25 | ON (detach) | 1.097 | 0.193 | 0.904 | 10.51 |
| 2^25 | ON2 (+prelaunch) | **1.068** | 0.339 | **0.729** | 10.58 |

2^24 is a wash (queue-bound; no host mass to overlap — the win is not
there by construction). @2^25 the round loop drops 0.983 → 0.729
(−26%); the prelude grows +0.14 because IncCR's prepare fill wait now
queues behind the prelaunched r0 CBs (relocated drain, not new cost).
RSS neutral (24.72/24.78/24.90 GiB — flights hold only ~0.5 MB eq
copies). No row-proportional residency added ⇒ no scale-transfer flag.

**Modeled @2^27:** loop floor ≈ queue exec (~2.56 s, scaled union) +
non-hideable host ≈ 2.6–2.7 s vs today's 3.83 s loop ⇒ ≈ −1.1 s; plus
r0-under-prepares hide (0.3–0.5) minus Inc-wait relocation ⇒ **−0.9..−1.4 s
st6b** (4.72 → ~3.4–3.8). The same arithmetic at 2^25 predicts the
measured −0.25 loop exactly; the 2^27 upside is the scale-nonlinear host
mass (first-touch wiring, oracle dilation) that 2^25 barely has. Bar
(≥1.0 s) is met at model center; the wave-gate 2^27 ABBA is the honest
verdict.

## 5. Doors (for the orchestrator)

- **CLOSED: st6b SLC tiling** (gpu-util parked door #3) — roof receipts §2.
- **Parked, priced: BytecodeLazyRound specialization** — runs at 1.7–3.6
  Gmul/s vs 7–12 for its siblings; the `JK_BYTECODE_MAX_FACTORS=16`
  thread arrays (384 u32) + runtime-bounded factor loops likely spill /
  cap occupancy at the production `factors = 3`. A factor-specialized
  variant (function constant or `_f3` kernel) is a plausible 2× on ~1.0 s
  @2^27 of queue ⇒ ~0.3–0.7 s. Needs a jolt-eval fixture; single-kernel
  lane shape.
- **Parked observation: RamRAV base cost** — 2 polys cost ~66% of the
  16-poly InstrRAV window; the per-thread base (rows + eq + tg_sums)
  dominates small-poly drivers. Cross-driver row-walk fusion was priced
  shut in w3 on ALU grounds; the base-sharing angle is different but the
  ownership blast radius is the same.
- IncCR prepare fill wait could move ahead of the prelaunches (stage-entry
  fill) to reclaim the relocated ~0.1 s @2^25 prelude — coupling across
  members; not worth it alone.

## 6. Discipline

- FrBind 252.6 µs pre-profile (gate <350; ref 255). 2^22 probe 2.78 s
  (≤3.40 gate). ONE 2^27 profile (traced 36.27 s, no cert claimed); CB
  traces @2^25; decision pairs @2^24 + @2^25 same-window interleaved,
  ≤2 timed per decision per scale.
- Gates: metal suites **414/414** (1 known leaky) · byte-diff ratchet
  **20/20 plain + 20/20 metal-armed** · slot suites both switch arms ·
  `clippy --all --features host -D warnings` green · fmt green.
- Diff: 10 files; engine spans (`m_begin`/`m_collect`, subscriber-free
  cost ~0) kept — they are how any future profile attributes batch
  members.
- e2e verify green on every timed run (harness verifies each proof).
