# Metal wave 10 — lane S10: st5 deep-cut (re-attribution + RegistersValEvaluation device port)

## Verdict

**RETAIN.** Phase-1 re-attribution overturned the door ranking; the cut
shipped is a **RegistersValEvaluation fused bind+eval device port**
(default-on, kill switch `JOLT_METAL_REGVAL=0`):

- **@2^27 (matched record-class windows, FrBind 249.6/251.8 µs):** st5
  13.93 → 12.96 s (**−0.97 s wall**); component-attributed effect
  **−1.33 s** (RegVal spans 2.53 → 0.04 = −2.49; exposed IRR cycle-CB waits
  +0.84; slot prepare +0.32, ~half since trimmed). Untouched scan spans ran
  +0.19 s in the ON run (window noise absorbing the wall/component gap).
- **@2^25 paired (30 s cooldowns):** st5 3.300 → 3.107 (−0.194 s), wall
  14.19 → 13.88; RegVal spans 0.456 → 0.023.
- **RSS @2^27: peak 74.05 → 72.36 GiB (−1.7 GiB)** — the device tables are
  mmap-backed and free at the round-11 gate handoff, dropping ~9 GiB
  (incl. the 4 GiB inc vector) mid-stage, where the host path's inc/wa
  Vecs stay truncated-not-freed through the stage's IRR-adoption peak.
- Proof bytes **identical ON↔OFF @2^21 AND @2^22** (fib, metal backend;
  sha256 receipts below), all runs verify. Byte-parity default holds — no
  protocol change.

Bar (≥1.0 s modeled @2^27): **met** (−1.33 s component-measured at target
scale; conservative wall pair −0.97 with adverse window noise).

Commits: `c1f9529e4` (port) on `lane/metal-w10-st5` off `c857970ba`.

## Phase 1 — post-S5 st5 re-attribution @2^27

One instrumented profile (sanctioned), FrBind 249.6 µs, traced wall
**50.48 s** (record-class; untraced record 50.56). st5 = 13.93 s, 100%
explained:

| component | s | vs wave-5 anatomy |
|---|---:|---|
| IrrPhaseScan+Suffix CBs ×16 (`IrrScanner::phase_run`) | **9.83** | 11.80 pre-S5 → S5 landed ≈ −2.0 @2^27 |
| RegistersValEvaluation::prove_round ×55 (host) | **2.53** | unchanged (3.0 in a worse window) |
| IRR cycle_init 0.52 + RegVal prepare/oracle 0.61 | 1.13 | |
| InstructionReadRaf::prepare self | **0.24** | **was "1.84 reclaim" — GONE** |
| bind/messages/output_claims | ~0.1 | |

Doors closed with receipts:

1. **(a) scan-CB merge / host-gap overlap: DEAD.** The 16 phase CBs are
   back-to-back — inter-CB gaps 2.5–3.4 ms (~40 ms total); wrap+encode
   overhead 12 ms total (`phase_scan_device` 9.846 vs `phase_run` 9.834).
   99.9% of the scan mass is in-CB GPU execution (S5-tuned kernels; P8-14
   residual stays parked). S5's parked "re-wrap ~0.6 s @2^27 host lever"
   is also dead at 2^27: measured 12 ms.
2. **(c) IRR prepare reclaim 1.84 s: DEAD — already collapsed to 0.24 s
   self.** The wave-5 1.84 was the prepare span TOTAL (included the
   phase-0 scan 0.69 + eq evals 0.23); E8's packed `get_pc` killed the
   residual walk cost.
3. **(b) RegistersValEvaluation 2.53 s host with an idle GPU underneath
   (IRR cycle launch/wait ≈ 0.001 s) → promoted to the top item.**

## Phase 2 — the port (mechanism)

New slot `MetalRegistersValEvaluation` (jolt-kernels
`metal/slots/registers_val_evaluation.rs`) + kernel
`jk_registers_val_round` (`KernelId::ALL` 82 → **83**):

- **One fused dispatch per round** (W2 slot pattern): fold `cur → nxt`
  with the round challenge out of place + accumulate the cubic's
  `t ∈ {0,2,3}` partial sums per threadgroup; host finishes sums and
  assembles the wire poly via the same `from_evals_and_hint` recipe.
  Fr arithmetic exact on both sides ⇒ byte-identical values under any
  summation regrouping.
- **Table strategy:** Inc = no-copy wrap of the host oracle vector as
  ping-pong `cur` (+T/2 mmap `nxt`); Wa served in-kernel from packed rd
  bytes + the 128-entry address-eq table until the first bind densifies it
  into a device `T/2` buffer (the K×T grid never exists on either tier);
  LT stays in the optimized tier's SplitLt form — host binds the ~√T lo
  table per round (µs), kernel computes
  `lt_hi[j>>log_lo] + eq_hi[·]·lt_lo[j&mask]` in place; the split→dense LT
  transition happens below the device gate in production (dense mode still
  supported for the gate-free test path).
- **Async via the engine's begin/collect seam:** rounds commit as a
  detached CB in `begin_round`; the batch engine runs sync members while
  it executes; collected in `collect_round`. CB trace @2^25 shows the
  RegVal CBs partially overlapping IRR's cycle CBs on-device (shared
  GPU windows), GPU totals 113 ms (RegVal) vs 140 ms (IRR cycle).
- **Fallback ladder:** below-gate prepare → optimized kernel from the same
  shared `ValEvaluationParts` (zero duplicated table work); per-round gate
  crossing (round 11 @2^27, groups < 2^16) → copy 2×2^17 Fr out of unified
  memory, drop all device buffers, resume on `ValEvaluationKernel::
  from_bound_state`; any dispatch/wait failure → same handoff with cur
  tables + pre-bind LT intact (LT folds into a flight-owned clone, applied
  only on successful collect), host recomputes the SAME round.
- **Kill switch `JOLT_METAL_REGVAL=0`** (plus the standard
  `JOLT_METAL_MIN_TERMS_REGISTERS_VAL_EVALUATION` gate override).

Why the win is scale-dependent (the 2^24 probe was a wash, −0.05 s): at
small scales the exposed IRR cycle-CB waits + per-CB latency (~2.4 ms
floor) eat the freed host time. The freed host cost is superlinear
(0.456 s @2^25 → 2.53 s @2^27 = ×5.5 for ×4 elements — host allocation +
bandwidth degradation), while the exposed GPU mass scales ~×4
(cycle_wait 0.183 → 0.844 measured). Hence −0.19 @2^25 → −1.33 @2^27.

## Receipts

- Proof sha256 ON == OFF: `0e93560b…d6d1` (2^21), `48ec537f…42c9` (2^22).
- Slot parity tests (device arm forced, `JOLT_METAL_MIN_TERMS=0` +
  `device_probe_count` engagement asserts): structured log_t 3/4/6
  (crosses Wa densify, dense device rounds, split→dense LT on device),
  parked-indices reclaim, kill-switch no-dispatch — 11/11.
- Gate battery: metal suites **411/411** (was 406 + 5 new) · prover-fixtures
  byte-diff **20/20 first pass** · clippy `--all --features host`
  `-D warnings` clean · clippy jolt-kernels `--features metal` clean · fmt.
- 2^25 CB trace (`JOLT_METAL_CB_TRACE=1`, sanctioned): RegVal CB GPU
  113 ms total (r0 28.8, r1 44.5, geometric tail), IrrCycleRound 140 ms —
  the model inputs for the exposed-wait projection that the 2^27 profile
  then confirmed (0.844 s measured vs 0.84 projected).

## Parked / follow-ups

- **Exposed IRR cycle-CB waits 0.84 s @2^27** are now the st5 cycle-phase
  floor — an IrrCycleRound kernel door (factors=9 fused fold+grid), not
  this lane's. Next flagship inside st5 remains the scan CBs (9.8 s,
  S5-tuned, P8-14 parked).
- Slot prepare self-cost ~0.29 s @2^27 (pre-trim): `pack_rd` since
  parallelized (modeled ~−0.15 s, not re-measured at 2^27); residual is
  the 4 GiB no-copy wrap + mmap setup.
- The late-2^27 justification (journal rule): the 2^24 wash made
  scale-transfer the open question; extrapolating ×4 across a demonstrated
  superlinearity was not report-grade, so the second sanctioned profile
  pinned the headline at target scale.

## Discipline

- Timed 2^27 runs: **2 of 2 sanctioned** (phase-1 attribution OFF-arm;
  late ON-arm, justification above). No certification claims — orchestrator
  gates.
- Timed decision pairs: 2^24 ON/OFF (chrome, 30 s cooldown), 2^25 ON/OFF
  (chrome, 30 s cooldowns) — 2 runs per decision. 2^25 CB-trace run:
  attribution. 2^21/2^22 ×4: byte-parity receipts. FrBind health before
  each 2^27 profile: 249.6 / 251.8 µs (<350 gate).
- Diff audited: temporary proof-dump probe in `modular_benchmark`
  reverted; permanent additions = the slot, its parity tests, the shared
  `ValEvaluationParts` refactor, `SplitLt` Clone derive.
- Not pushed; `scratch/metal-saturation` untouched. Worktree
  `.worktrees/metal-w10-st5` (branch `lane/metal-w10-st5`) ready for merge
  + cleanup after the wave gate.
