# Metal wave 15 — lane P15: st5 residual bundle (IrrCycleRound waits + scan tail)

## Verdict

**PARTIAL — bundle modeled −0.55..−0.7 s @2^27, bar (≥1.0 s) missed with
the remaining mass closed as floor.** Commit `375558602` on
`lane/metal-w15-st5res` (base 2b0959e30). Three RETAINed cuts, three doors
closed with receipts:

1. **Cycle ping-pong pre-wire (RETAIN, `JOLT_IRR_CYCLE_PREWIRE=0` kills,
   `=N` sets the trigger phase, default 12):** the cur/nxt pair (32.2 GiB
   @2^27, f=5) is allocated during the address phases and wired by a
   detached one-thread `FrBind` CB **on a side queue** while the phase-scan
   CBs execute. Mechanism receipt: Metal wires every referenced no-copy
   buffer at CB **schedule**; fresh `MAP_ANON` pages wire at ~47-50 GB/s
   (@2^25: init CB blocked 207 ms vs 92.6 GPU; round-0 blocked 110 vs 55.9
   — the overhang is exactly the fresh cur/nxt wire). Side queue is
   REQUIRED: on the shared queue the in-order schedule pipeline made the
   remaining phase CBs pay the wire 1:1 (phase blocked 910→987 ms, st5
   wash) — Metal schedules a queue's CBs in order.
2. **Fused cycle init (RETAIN, `JOLT_IRR_CYCLE_INIT_SPLIT=1` kills):** one
   `jk_irr_cycle_init_fused` dispatch writes combined_val + all ra tables,
   reading each row once — the old shape re-read all rows per output table
   (5 dispatches @f=5) and serialized 5 occupancy ramps. Init CB GPU 92.6
   → 35.5 ms @2^25 (−62%). `KernelId::ALL` 88 → **89**.
3. **Parallel bucket build (RETAIN, unconditional):** prepare's serial
   2^27-row walk into per-table buckets → chunk-parallel collect +
   table-major ordered concat, byte-identical `bucket_flat`/`present`.

## Measured — @2^25 same-window pair (FrBind 250.2 µs, 40 s cooldowns)

| span | ON | OFF | Δ |
|---|---:|---:|---:|
| st5 | 1291.6 ms | 1435.4 ms | **−143.8 (−10.0%)** |
| IrrScanner::cycle_init_run | 43.1 | 134.7 | −68% |
| IrrScanner::cycle_wait (11 CBs) | 143.8 | 184.3 | −22% |
| IrrScanner::phase_run (16 CBs) | 840.6 | 859.0 | no wire theft |
| walls | 11.27 s | 11.46 s | (ambient-shared) |

CB-trace receipts (attribution runs): cycle-stack blocked 404 → 186 ms;
init CB blocked 207 → 47.7; r0 110 → 53.7. Proofs verify on every run;
byte parity by construction (allocation/scheduling only + identical
per-element init math), pinned by the 10/10 scanner parity tests and the
20/20 byte-diff ratchet.

## Modeled @2^27 (post-w14 st5 = 6.66-6.76 s)

- init 0.522 → ~0.19 (fused GPU ×4-scaled + wire off critical path): **−0.33**
- cycle waits 0.854: r0's nxt wire (10.7 GiB, bytes-scaled from the @2^25
  54 ms gap): **−0.15..−0.21**
- prepare bucket walk (serial O(T) removed): **−0.1..−0.2 modeled,
  unisolated** (@2^25 both arms carry it; lands in the wave-gate profile)
- **Total −0.55..−0.7 s.** Transfer haircut applies at the wave gate.

## Re-attribution @2^27 (one sanctioned instrumented profile, record-class window: stages sum 39.7 s, st0 8.59)

st5 = **6.764 s, 100% explained** — the STATUS scan estimate (3.0-3.3) was
low:

| component | s |
|---|---:|
| phase+suffix scan CBs ×16 (`phase_run`; 1 in prepare) | **4.119** |
| IrrCycleRound exposed waits ×13 | 0.854 |
| cycle init CB | 0.522 |
| IRR prepare (eq_table 0.33 + phase-0 scan ~0.26 + buckets/misc) | 0.846 |
| RegVal prepare (oracle_table 0.326) | 0.345 |
| RamRA prepare 0.117 · address/cycle msgs 0.02 · output_claims 0.044 · bind self ~0.06 | 0.24 |

## Doors closed (receipts)

1. **IrrCycleRound kernel cut: DEAD — the kernel is at 75-80% of the Fr
   ALU chain roof** (bind @2^24 f5: 138 M muls / 16.0 ms = 8.7 Gmul/s vs
   11.6 saturated; occupancy proxy maxTotalThreads 1024 = no register
   cliff). Compile-time factor-count probes (f5/f9, registers pinned)
   −2.9..−6.6%; grid-stride S=4/8 probes 0/negative; linear in len
   (×3.99/octave to 2^26, no TLB cliff). Rig: `jolt-eval --bench
   irr_cycle` (parity-checked probes; flag with irr_roof for PR audit).
2. **The 0.854 s "exposed wait" is NOT a pipelining/wait-merge door** —
   composition: IRR exec ~0.45 (roofed) + **RegistersValRound co-run
   ~0.35-0.40** (CB trace: r1+ pairs share identical GPU windows, both
   stretched to the union — R12's co-run additivity, here benign: the two
   slots' work truly overlaps and the wall pays the union) + wire/latency
   ~0.05 (cut by prewire). Encode/dispatch overlap has nothing left to
   hide (cycle_launch 0.4 ms total; Fiat-Shamir serializes rounds).
3. **Global presort: PRICE-OUT — S12's premise is dead on real rows.**
   2^24 sha2-chain dump: **59.47% of rows have a unique full 128-bit
   lookup index** (run p50/p90 = 1, p99 = 2; one 4M-row degenerate index)
   — "scans become pure run-length" is false. Sorted-order uniform-tile
   rates: phase 0 100%, phases 1-8 70-91%, phases 10-15 only 39-43%
   (suffix: phase 8 4.3→91%, phases 12-15 →35-38%). Gross win on S12's
   machinery ratios ≈ −0.7 s; a 128-bit GPU radix sort + rows/u_evals
   private gathers + bucket remap costs ≥0.3 s GPU, +11 GiB transient,
   new-kernel surface, and a Declined→rebuild fallback rework ⇒ **net
   ≈ −0.4 s max at the campaign's largest blast radius. KILL** (supersedes
   the S12 park price of −1.1). Cheaper variant priced too: suffix-only
   bucket_flat sort (order-free consumer, no copies) ≈ −0.3 s for ~0.3-1 s
   host sort or new GPU sort kernels — NO-GO.

## Ranked st5 residual (for the orchestrator)

1. Scan CBs 4.12 s — quiet-floor gap ~1.4 s but presort is dead; needs a
   new mechanism (not this lane's).
2. IRR cycle exec ~0.45 + RegVal co-run ~0.35-0.40 — both at/near kernel
   roofs; RegVal's slot is not this lane's.
3. RegVal prepare 0.345 (`oracle_table` inc-vector build, witness side).
4. IRR prepare residual ~0.5 after buckets (eq_table 0.33 is a parallel
   4.3 GiB fill; phase-0 scan is scan mass).

## Scale-transfer / residency flag

Prewire holds the 32.2 GiB pair from phase ~12 to adoption (~1.2 s longer
@2^27). @2^25 measured peak +1.18 GiB (24.79 vs 23.61). Kill-switch ABBA
at 2^27 with RSS capture before default-on stands per the w13 rule —
orchestrator gate. `JOLT_IRR_CYCLE_PREWIRE=0` restores.

## Gates

- metal suites **413/413** · byte-diff ratchet **20/20 first pass** ·
  `clippy --all --features host -D warnings` clean · clippy jolt-kernels
  metal clean · fmt applied · e2e verify green on all cited runs.
- Pre-existing (also on clean HEAD, not this lane):
  `registers_read_write.rs:1559` unfulfilled-expect fires only under
  metal+bench-utils `--all-targets`.

## Discipline

- 2^27: **1 instrumented profile (the sanctioned one), 0 timed runs.**
- 2^25: two timed span pairs (the first pair exposed the shared-queue wire
  theft — a real decision disagreement; the side-queue fix re-paired) +
  3 CB-trace attribution runs + 1 untimed verify run. 2^24: 1 untimed
  dump run (`JOLT_IRR_DUMP_ROWS`). Kernel iteration via the gpu-locked
  `irr_cycle` bench (fixture scales 2^22-2^26).
- All cargo under the wave lockf; all GPU under the gpu lockf; FrBind
  250.2 µs at session start.
- Diff audited: fixture + probes are bench-utils-gated attribution rig
  (permanent, irr_roof-style); no temporary probes left; production diff =
  fused init kernel + prewire + side-queue pass + parallel buckets +
  `ScannerInputs.ra_count`.
- Not pushed; `scratch/metal-saturation` and sibling worktree untouched.
  Worktree `.worktrees/metal-w15-st5res` ready for merge + cleanup after
  the wave gate.
