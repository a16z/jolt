# Metal M5 saturation campaign — live journal

Opened 2026-08-04 from `feat/metal` / `88b063db3`. Journal style (user
directive): lane entries = verdict + numbers + commit + one-line mechanism.

## STATUS-ADDENDUM (wave 19, 2026-08-27 ~14:00): MULTI-WORKLOAD AXIS
OPEN. R19 baselines — **btreemap @2^27 39.26-39.55 s / RSS 74.5-75.4
GiB · sha3-chain 37.20-39.22 / 69-71 GiB** vs sha2 36.60 median;
@2^25 btree 12.06 · sha3 11.28 · sha2 11.14 (re-anchor). **Metal/CPU
ratios: btree 8.8× < sha2 10.2× < sha3 13.1× — btreemap is the
Metal-relative target.** Excess is host-side stages sha2 never
exercised: btree st2 RamRW +1.86 (ram_K 2^19 vs 2^13, rounds 3.9×) ·
st6a bytecode addr-prepare +0.99 btree/+0.45 sha3 (serial host walk) ·
st7 HW prepare +0.52 btree · sha3 st4 +1.01 (+27% reg-op density).
**All campaign-roofed GPU stages flat across shapes (±0.4) — w5-w18
receipts generalize; scan/presort/machinery kills HOLD on real rows
of all 3 shapes** (one sub-bar inversion: btree suffix fixed-step
−0.06). 16/16 runs verify. Harness bugs: btreemap default targeting
panics ≥2^26 (stale c/op 1550, real 1247→2032; workaround
--target-trace-size), CYCLES_PER_SHA3 overstated (4330 vs ~3345 ⇒
69% fill). Wave-20 doors ranked: (1) st2 RamRW device-port RE-PRICE
on btree 1.0-1.5 — **INVALIDATES the "below bar" kill receipt**
(priced on sha2's trivial ram_K); (2) st6a addr-prepare
devicify/unblock 0.8-1.2; (3) st7 HW prepare parallelize 0.5-0.9
(sha2's 1.28 st7 also unattributed); (4) st4 RegRw sha3 0.5-0.8
(parked w21); (5) st6b btree residuals ~0.5; (6) harness fixes.
C19 cleanup merged (−8.8k LOC, commitment/ split, BENCHES.md);
combined battery green (metal 412/412 · ratchet 20/20 · legacy
444/480/445 · verifier 285); split sanity wall 36.68 @2^27 —
perf-neutral. Trunk ea6cc017d.

### Wave-20 lanes (+ integration)

| lane | task | scope | bar |
|---|---|---|---|
| H20 | 0a71caff (fable-max) | st2 RamRW device-port re-price (btree, shape-aware default) + st6a addr-prepare | ≥0.5 s btree |
| P20 | 7ed6b3e6 (fable-max) | st7 HW prepare (+ sha2 st7 attribution) + st6b btree residuals + harness targeting fixes | ≥0.4 s btree |
| I20 | a08a1a81 (fable-max) | PR #1733 refresh artifact: squash trunk 267d3115b onto origin/main 72dc64516 (12-commit drift incl. #1809 mapper convergence), battery, no push | artifact |

Worktrees: .worktrees/metal-w20-{ramrw,prep} off 267d3115b;
.worktrees/metal-integrate off origin/main 72dc64516. w19 worktree
removed. Gate plan: btree @2^27 ABBA on the wave's switches + one
sha2 @2^27 neutrality wall + 2^25 records ×3 workloads.

## STATUS: ACTIVE — METAL-DORY; waves 5-18 CLOSED, perf phase DONE.
**2^27 record 35.29 s / 3.803 MHz** (all-time best, w16 cool window;
wave-18 clean median **36.60**, best 35.93 second-best-ever; trunk
c8cbe6764, 2026-08-27 ~11:45). **2^25 record 11.21 s.** Wave-18 ABBA
−0.22 s switchable half (pairs +0.05/−0.61/−0.09, 2/3 ON — weakest
signal of the campaign, fragments-wave expected; F18's RLC/prep host
cuts sit in both arms). **st8 span receipt 4.80→4.42 (−0.38, matches
F18 model −0.44..0.52).** Stage vector @2^27 (36.60 traced): st0 7.42
(floor ~6.7) · **st5 5.59** · st4 4.85 · st6b 4.53 · **st8 4.42 (open
4.08; combine_hints 0.297, was 0.546)** · st1 3.25 · st2 2.72 ·
st3 2.18 · st7 1.28. Cumulative vs Aug-24 baseline 63.88/64.56:
**−28.6 s best (−45%), median −27.96 (−43%)**; 2^25 16.72→11.21
(−33%); chain w5..w18 = −4.72 −1.02 −0.50 −2.07 −0.94 −2.73 −4.23
−4.87 0.00 −1.92 −1.35 −1.35 −0.37 −0.22. **PERF DOORS EXHAUSTED:
every stage at hardware roofs or killed with receipts** (w18 closed
the last two parked prizes premise-false/sub-bar; remaining unpriced:
st4/st1 grid-stride t1, likely sub-0.3 fragments). **User decisions (Aug 27
~11:55): (1) after C19 lands → REFRESH PR #1733** (same squash path:
single Metal-backend commit on latest origin/main, full battery,
force-push; PR stays draft — user marks ready). **(2) Campaign
CONTINUES — new axis: MULTI-WORKLOAD coverage** (sha3-chain/btreemap
shapes): profile first; shape-dependent closed doors may reopen
(scan-entropy stats, run lengths, one-hot densities were measured on
sha2-chain rows); workload-specific regressions vs the CPU backends
are fair targets. Same wave/gate discipline + velocity rules.
Measurement rule (3rd offense): the FIRST 2^27 run after any
build/battery session is +1.5-2 s hot — never an ABBA arm. **User-order artifacts (Aug 26): PR #1733 = main+1 @8ba8f1121
(squash approved); mapper PR #1809 open.** Campaign stays on its own
vintage (e4679b5d2) — do NOT rebase scratch/metal-saturation;
integration lives in the PR artifact only.

Mandate (user, Aug 25): **make the Metal Dory implementation faster.**
Fresh orchestrator (predecessor dade2763 lost steer delivery to a daemon
bug). Baseline = the Aug 24 record: 63.88 s best / 64.56 s median @2^27,
16.72 s @2^25, RSS 72.2 GiB @ d2523b09a. Doors from parent: (1) TRS
flat-PC-cache exploitation on Metal, (2) LATTICE Dory receipts — fold-chain
break/streamed folds (challenge pushes), on-GPU public-matrix regen from
seed, (3) st8 Miller-fold/dispatch-merge follow-ons, (4) global
address-major Dory stays DEAD. st0/st5 re-attribution in-mandate.

### Record-trace attribution (2^27 Perfetto, Aug 24 18:09, this trace ~64.4 s)

- st0 16.86 (stream_witnesses 16.21 OPAQUE — contains Dory tier-1/2 GPU
  commit + TraceRecord::collect 11.99 host overlap; prepare_tier2 0.47)
- st5 16.34 (IRR 11.20/311 rounds + RegistersValEval 3.03 + prepare 1.58;
  NO Dory content; **+5.4 s vs wave-3 model ~10.9** — the overshoot lives
  here; trace taken after the N=5 timed runs ⇒ sustained-load skew suspect)
- st4 8.07 · st8 5.82 (Dory open 5.11: miller_fly 3.24/18 · first_msg
  2.03 · second_msg 1.32 · folds 0.86; combine_hints 0.56) · st6b 5.29 ·
  st1 5.03 · st2 2.79 · st3 2.26
- Outside prove wall: Dory setup_prover 48.9 s, setup_verifier 15.5 s
  (55 CPU multi_pairs = the unprimed prepared-point cache, scheme.rs:114
  deliberate). Prove-wall PC-cache exposure measured: 0.035 s (st8 host
  multi_pair x2).

### Wave-5 plan (steps verbatim)

1. **R (re-attribution, st0+st5):** From the record trace + one
   instrumented profile run, explain st0's opaque `stream_witnesses`
   16.21 s (sub-span the tier-1 GPU commit vs `TraceRecord::collect`
   11.99 s host overlap vs tier-2) and the st5 gap (measured 16.34 s vs
   wave-3 model ~10.9 s — verify the wave-3 scan kernels actually engage
   at 2^27 and disentangle sustained-load window skew with a fresh-window
   probe). Deliverable: ranked Dory-target table explaining ≥80% of st0
   mass + st5 gap mechanism, lane report `metal-w5-reattr.md`. Bar:
   attribution only, ≤2 timed runs.
2. **B (st8 Dory open):** Port LATTICE fold-chain break/streamed folds
   (challenge pushes) to the Metal Dory reduce (18 rounds: first_message
   2.03 s + second_message 1.32 s + folds 0.86 s), plus jk_miller_table
   TG-cap-32 handoff (−24% on commit shape) and dispatch-merge follow-ons
   on miller_fly 3.24 s. Bar: ≥0.4 s modeled st8 cut @2^27; soundness
   argument journaled for any protocol-shape change; kill switch on every
   default-on change. Lane report `metal-w5-st8.md`.
3. **T (TRS PC-cache + regen pricing):** Verify the TRS prepared-point
   cache question on the Metal prove path (known: deliberately unprimed
   at jolt-dory scheme.rs:114; record trace shows only 0.035 s host
   multi_pair in st8) — sweep for any other unprepared-pairing or
   repeated-preparation sites reachable during prove; price on-GPU
   public-matrix regen from seed for Metal applicability. Deliverable:
   GO/NO-GO receipts with numbers, `metal-w5-trs.md`. Desk + microbench
   only.
4. **Gate:** lane merges → full battery + same-window 2^25/2^27
   certification vs 63.88 s; report walls + Dory-stage spans + delta via
   message_parent.

skip: kanban card sync — pika-cli binary unavailable on macbook-home
(external machine); parent handles board.

### Wave-5 lanes

| lane | task | scope | bar |
|---|---|---|---|
| R | 34309e15 (fable-xhigh) | st0/st5 re-attribution | DONE — 100%/98% explained |
| S5 | 44896e3a (fable-max) | st5 scan dispatch-context gap (2.9×/2.1×) | DONE — −2.5 s modeled |
| B | 54bb951d (fable-max) | st8: LATTICE folds + miller_table cap32 + merges | DONE — −0.85 s modeled |
| T2b | cadd643a resumed | setup-own the base_affine_cache join arm too | DONE — join 278→1.2 ms |
| T | ccec49a5 (fable-high) | TRS PC-cache verify + regen pricing | DONE — GO −0.46 s + NO-GO |
| T2 | cadd643a (fable-high) | impl: setup-owned prepared-G2 table | −0.4 s st0, RSS-neutral prove |

Lane worktrees: .worktrees/metal-w5-{reattr,st8,trs} on branches
lane/metal-w5-* off 2e1efd307.

### Wave-5 results log (verdict · numbers · commit · mechanism)

- **Lane T receipts: DONE (desk, zero diff).** (1) TRS PC-cache **GO
  ≈−0.46 s @2^27**: pairing arithmetic already fully prepared-exploited in
  prove; the leak is preparation itself — `DoryTier2Prep::new` re-prepares
  2^17 setup G2 points EVERY proof inside st0 (jolt-kernels
  commitment.rs:388, microbench 468-500 ms = the trace span; 87-line
  Miller precompute is the cost). Fix: setup-owned prepared table on
  DoryProverSetup (one object = one URS ⇒ prefix-match sound), +0.95 s
  setup (out of wall), +4.18 GiB eager / 2.09 GiB lazy, value-exact.
  Out-of-wall bonus: same table guts setup_verifier's 15.5 s (55 unprimed
  multi_pairs). (2) regen-from-seed NO-GO → kill list. → T2 impl lane.
- **Lane R re-attribution: DONE (attribution-only, `b44ea7f95` RETAIN).**
  No regression — wave-3 st5 kernels fully engaged @2^27 (16/16 device,
  0 CPU fallbacks). **st5 +5.4 s vs model = model error:** scans cost
  11.80 s CB-wall @2^27; production per-row cost is a scale-flat 2.9×
  (phase) / 2.1× (suffix) over the fixture microbench — dispatch-context
  effect (cold single CB per phase with host gaps vs warm bench loop),
  ~7.7 s headroom at fixture rate. **st0 is HOST-bound @2^27** (regime
  flip vs 2^25): driver = extract 8.67 + build_gpu_job 4.06 +
  build_inc_job 4.26, send_wait 0.08; GPU starved 7.9 s (tier-1 8.8 +
  tier-2 Miller 5.5 CB-s fit underneath); TraceRecord::collect 12.8 s
  dilates the driver ~5 s. More GPU offload won't shrink st0.
  streaming.rs:464 affine cache never fires @2^27. Ranked doors: (1) st5
  dispatch gap ~7.7 s → lane S5; (2) st0 driver two-pass extract→build
  fusion + collect contention → wave 6; (3) tier-2 Miller 5.5 s — B's
  cap-32 door covers jk_miller_table on commit shape. Full tables:
  metal-w5-reattr.md. NOTE: R's profile ran with sibling lanes live —
  host-heavy stages inflated ~5-10%; **wave-gate certification must run
  solo (no live lanes).**
- **Lane B st8: DONE, 2 cuts RETAIN (`4ed633e21`), modeled −0.85 s st8
  @2^27, both bit-identical proof bytes.** (1) r0-D2 MSM+pair shortcut
  (resident loop ignored v2_scalars; CPU-arm compute_d2 identity):
  first_message −38% @2^15/2^16, −0.41..0.48 s. (2) window-table G2
  fixed-base kernel (VMV sweep was a plain 254-bit ladder; shared base →
  host 16-ary table, mixed adds): 123.8→22.1 ms @2^16 (5.6×), −0.39 s;
  KernelId 79→80. Fold-chain challenge-push NO-GO → kill list. cap-32
  miller_table shipped default-on (−23.6% on production shape) but runs
  only in st0 tier-2 fallback — st8 delta 0, and R showed tier-2 fits
  under the st0 driver ⇒ wall effect ~0 today, floors wave-6 driver wins.
  Merge follow-ons closed (no profitable seams; W4's CPU walls were
  contention-skewed). Parked: chunked fold→message pipeline 0.5-0.7 s.
  Kill switches: JOLT_DORY_R0_D2_MSM=0, JOLT_DORY_FIXED_BASE_TABLE=0,
  JOLT_METAL_PAIRING_TG_CAP=0. 405/405 + byte-diff 20/20 + clippy green.
- **T2 gap found by cross-check (orchestrator):** R's st0 table shows
  prepare_tier2 0.47 ∥ base_affine_cache 0.47 in a rayon JOIN — T2's prep
  removal alone leaves the join wall ≈0.47 (other arm), evaporating the
  −0.46 @2^27. T2 resumed: same setup-owned treatment for the G1 affine
  base cache arm.
- **T2 impl: RETAIN, ready to merge.** `13b01608b` — prepared-G2 table
  setup-owned (eager, Arc prefix-borrow); st0 prep join 304→2.4 ms @2^25;
  proofs bit-identical on/off @2^21+2^25; setup +0.26 s @2^25 (out of
  wall); RSS +1.18 GiB @2^25 / expect ~+2.1 GiB @2^27 (table = full 2^17
  g2_vec vs 2^16 transient it replaced — accepted, storm regime was
  ~97 GiB); kill switch JOLT_DORY_SETUP_PREP=0; suites 404/404 + clippy
  green. Wall cut certifies at wave gate (lane walls were ±8 s ambient,
  FrBind 510 µs — span probe is the evidence).
- **S5 interim — mechanism PINNED: data distribution, not clock/
  residency.** Real chunk keys have ~4.2 distinct values per 32-lane tile
  in phases 0-7 (85% per-lane repeat) → w3 collision-only SIMD scatter
  worst-case: all lanes collide, full-length 32-source shuffle-reduce per
  tile. Receipts @2^24: random keys 18.2 ms/phase vs k=2 72.2 ms (4.0×);
  real fib rows reproduce production within 10%. Host-gap axis
  inconsistent; buffer freshness: zero GPU effect (re-wrap ~13 ms wall/CB
  ≈ 0.6 s @2^27 minor host lever). **Wave-6 st0 steer: the tier-1/nocopy
  host-gap pattern is NOT a GPU-side cost door.** Fix in flight: per-lane
  run-length accumulation (~6.7× fewer flushes on plateau phases),
  byte-identical; modeled phase-side ≈3+ s @2^27, suffix next.
- **T2b join-arm completeness: RETAIN (`463475d38`).** G1 affine table
  also setup-owned (scalar_affine_bases feed borrows it); join span
  @2^25 278→1.2 ms, both arms ~0; byte-identical @2^21+2^25; RSS +0.98
  GiB @2^25; one switch JOLT_DORY_SETUP_PREP=0 restores both arms.
  Caveat: R's base_affine_cache 0.47 @2^27 was mostly contention-dilation
  (base solo 2.4 ms) — expectation stays ≈ −0.46 s, not −0.9.
- **S5 st5 scans: RETAIN, merged (`51b18a977`), measured −24.7% scan
  total @2^25 e2e paired (phase CBs 2052→1466 ms, P0-6 −52%; suffix
  −6.4% on fib mix), modeled −2.5 s st5 @2^27 (up to −3.8).** Fixes:
  per-lane run-length accumulators (flush on key change) + suffix group
  detection hoisted behind xor-neighbor entropy probe. Byte-identical
  @2^21+2^22 across new/eager/CPU arms; suites 404/404. Kill switches
  JOLT_IRR_PHASE_SCAN_EAGER=1 / JOLT_IRR_SUFFIX_SCAN_EAGER=1. KernelId
  79→81 (→82 merged with B's). Parked: P8-14 scatter d≈11.6 (~0.7 s, no
  cheap lever); re-wrap 13 ms wall/CB ≈0.6 s @2^27 host lever.

### Wave-6 lanes

| lane | task | scope | bar |
|---|---|---|---|
| S0 | 161a0669 (fable-max) | st0 driver: extract→bucket fusion, collect contention, re-wrap | ≥1.5 s st0 @2^27 |

Lane worktree: .worktrees/metal-w6-st0 (lane/metal-w6-st0 off a2eaededd).
Wave-5 lane worktrees removed, branches deleted (merged).

### Wave-7 lanes

| lane | task | scope | bar |
|---|---|---|---|
| D2 | 4e5fb0b7 (fable-max) | st0 tier-2 pipeline (miller CB + host decode/absorb/fold) | ≥1.0 s st0 @2^27 w/ floor math |
| RSS | 2265664a (fable-high) | eager G2-prep table sized to consumer bound | DONE — merged 3cac2e401 |

Worktrees: .worktrees/metal-w7-{tier2,rss} off a4028227c. Wave-6
worktree removed (merged).

### Wave-7 results log

- **Lane RSS slimming: RETAIN, merged (`124d89578` → trunk 3cac2e401).**
  Consumer bound proven = 2^⌊nv/2⌋ (all access via DoryTier2Prep::new;
  metal/optimized requests ≤ balanced-layout rows; FastTail/st8/verifier
  use raw SRS). Eager table halved: 2.05→1.02 GiB @nv33 ⇒ −2.05 GiB
  deterministic @2^27 (78.78 honest point → ≈76.7 modeled). Byte-identical
  on/off AND vs unsliced hashes @2^21+2^25; prepare span still ~0; suites
  406/406 (+ table-sizing test); oversize degrades to per-pass fallback.

### Wave-8 lanes

| lane | task | scope | bar |
|---|---|---|---|
| E8 | 78d8997f (fable-max) | st0 extract_bucket × collect contention @2^27 | ≥1.5 s st0 w/ transfer argument |
| URS | 4353db92 (fable-high) | recurring committed_* byte-diff gate flake | mechanism receipt + fix, 3× green |

Worktrees: .worktrees/metal-w8-{extract,urs} off dc49d402f. Wave-7
worktrees removed (merged).

### Wave-9 lanes

| lane | task | scope | bar |
|---|---|---|---|
| X9 | 74b6ea5c (fable-max) | tier-1 jk_g1_seg_sum XYZZ headroom (7.63 CB-s; 2.52 vs 11.30 Gmul/s) | ≥0.8 s st0 or NO-GO w/ repriced roof |
| URS | 4353db92 (resumed) | gate-flake mechanism + fix | DONE — merged 3455ef226 |

Worktree: .worktrees/metal-w9-xyzz off 71ee2c57e. Wave-8 worktrees
removed (extract merged; urs still active on its dc49d402f base).

### Wave-18 plan (FRAGMENTS wave — endgame)

Doors are thin (see STATUS); honest bars at ≥0.3 s. Plan: (1) lane F18
— st8 banked residuals: combine_hints 0.546 attribution+cut, preamble
host G1 MSMs→device ~0.10-0.12, untraced preamble 0.257 span+attribute;
T17's kill list binding (no fly→table, no fold pipeline, overlap only
GPU∥host/IO). (2) lane L18 — st6b BytecodeLazyRound
factor-specialization (B16: 1.7-3.6 vs 7-12 Gmul/s siblings, ~0.3-0.7)
+ RamRAV base anomaly; SLC/gather-residual kills binding. (3) Gate:
merge → battery → kill-switch ABBA @2^27 → records → journal. After
the w18 gate the campaign moves to PR-handoff cleanup (flagged debt:
E8 telemetry, X9/T13/irr rigs, w13 miller probe, commitment.rs 2587
lines) unless the parent redirects to a new axis (endgame question
pending, update #4746).

### Wave-18 lanes

| lane | task | scope | bar |
|---|---|---|---|
| F18 | 1055fe6a (fable-max) | st8 fragments: combine_hints + preamble MSMs + untraced | ≥0.3 s |
| L18 | 32a810bc (fable-max) | st6b BytecodeLazyRound specialization + RamRAV anomaly | ≥0.3 s |

Worktrees: .worktrees/metal-w18-{st8frag,lazy} off c334d07de. w17
worktrees removed.

### Wave-19 plan (multi-workload axis opener) + cleanup

Plan: (1) lane R19 — attribution-only on sha3-chain + btreemap: walls
+ stage vectors @2^25 and largest RSS-feasible big scale (probe 2^26
first), per-cycle comparison vs the sha2-chain vector, st5 row stats
(entropy/run lengths) on new shapes, CPU-backend speedup ratios,
ranked wave-20 door list + correctness flags. No optimization. (2)
lane C19 — PR-handoff cleanup (parallel; no GPU). (3) When C19 lands:
gate (battery only — pure refactor, no ABBA), then PR #1733 refresh
per user order (squash on latest origin/main, battery, force-push,
stays draft). (4) When R19 lands: wave-19 gate → spawn wave-20 doors.

| lane | task | scope |
|---|---|---|
| R19 | 8da1856f (fable-max) | sha3/btreemap shape attribution + ranked wave-20 doors |
| C19 | 70ec974e (fable-max) | DONE, merged 9da97dbdf → 236f3ceb1: −8.8k LOC (11 one-off rigs + 5 probe examples + E8 telemetry + g1bat leg deleted, all campaign-only vs origin/main), commitment.rs 2537→commitment/{mod,builder,tier2,tests,bench}, BENCHES.md keepers doc. Lane gates green (metal 412/412 — 4 deliberate test deletions with parity retained; ratchet 20/20). **Trunk battery deferred to the combined wave-19 gate** — running it mid-R19 would pollute R19's timed windows. Deliberate keeps a PR reviewer may flag: REGRW_FUSED opt-in arm, st2 device port (default-off), kill-switch farm (cert story). |

Worktrees: .worktrees/metal-w19-shapes + .worktrees/metal-cleanup off
c4aceaedd/74c3a547a. w18 worktrees removed.

### Wave-18 GATE (2026-08-27 ~11:45 ET, trunk c8cbe6764) — 2^25 RECORD 11.21, perf phase closed

- **Lane F18: GO, merged (`cf3214981` → b91b8dbee).** Three cuts, all
  st8: (1) combine_hints w=2 wNAF signed-digit sweep in
  `jk_g1_combine_rows` (kernel −28.6%, exact −33% add-count model) +
  one-pass parallel chunked normalize (prep 21→5 ms, −0.47 GiB
  transient); (2) preamble VMV G1 MSMs → device via new
  `RoutineHooks.g1_msm` seam (SortedMsm over zero-copy host bases,
  2.7× vs host, engages ≥2^13); (3) untraced preamble 0.257 attributed
  = `RlcSource::fold_rows` serial RLC accumulation → rayon
  column-partitioned (per-column op order preserved ⇒ value-identical;
  no switch needed, byte-diff covers). Kill switches:
  JOLT_METAL_COMBINE_NAF=0 · JOLT_METAL_MIN_TERMS_DORY_HOST_MSM=huge.
  New permanent attribution spans (combine_hints_*,
  dory_host_msm_device, rlc_*). 3 new parity tests (metal 414→416).
  **@2^27 receipt: st8 4.80→4.42, combine_hints 0.546→0.297 (kernel
  0.275), device MSMs 0.076, fold_rows 0.166** — every F18 prediction
  landed. Residuals closed: post-NAF kernel floor ≈0.28-0.38 is ALU;
  buckets/GLV priced sub-bar. Report:
  lane-reports/metal-w18-st8frag.md.
- **Lane L18: double NO-GO with receipts, merged (`2b50e800c` →
  c8cbe6764; production untouched — bench-utils-gated attribution rig
  `jolt-eval --bench bytecode_lazy` + fixtures only).**
  BytecodeLazyRound door (parked 0.3-0.7) DEAD premise-false: isolated
  exec of ALL bytecode device kernels ≈ **155 ms @2^27** — B16 priced
  co-run *windows* (60-132 ms) against 8-16 ms actual exec; the
  "1.7-3.6 Gmul/s" was an op-mix metric artifact (kernel is at the
  compound ALU roof, occupancy 1024/32). Factor-spec f3 −10..12% ≈
  −14 ms — sub-bar 20×. RamRAV anomaly explained: per-poly ram RAV is
  25% CHEAPER than instr RAV; the 66% figure was CB window overlap.
  st6b's queue is Bool+InstrRAV gather ALU (kill-listed) + host/waits
  — no kernel prize. Report: lane-reports/metal-w18-lazy.md.
- **Battery green** (clippy host + host,zk + metal+bench-utils · metal
  **416/416** · ratchet 20/20 · legacy 444/480/445 · verifier 285 ·
  release build).
- **Cert (FrBind 251.3 µs):** kill-switch ABBA @2^27 (OFF =
  COMBINE_NAF=0 + MIN_TERMS=huge), three pairs on disagreement:
  +0.05/−0.61/−0.09 ⇒ **−0.22 s mean switchable** (ON 36.56 vs OFF
  36.78); RLC/prep host cuts live in both arms — the st8 span delta
  −0.38 is the honest wave receipt. Warmup 39.78 discarded. RSS
  71.76 GiB (back in the 71-72 band; w17's 69.24 = outlier).
- **RECORDS: 2^25 11.21 s** (11.21/11.25 — matches F18's lane ABBA
  −0.145). 2^27 clean N=5 35.93/36.58/36.60(traced)/36.79/36.96,
  median 36.60; best 35.93 = second-best untraced ever (all-time 35.29
  stands).
- **Wave-18 stage vector @2^27 (traced 36.60): st8 4.42 (open 4.08)**
  · st0 7.42 · st5 5.59 · st4 4.85 · st6b 4.53 (+6a 0.32) · st1 3.25 ·
  st2 2.72 · st3 2.18 · st7 1.28.
- **Perf phase closed.** Both wave-18 doors were the last parked
  prizes ≥0.3; L18 killed one premise-false, F18 banked the other.
  Next: PR-handoff cleanup track (see STATUS).

### Wave-17 lanes

| lane | task | scope | bar |
|---|---|---|---|
| G17 | 4353db92 (fable-max) | st5 scan-gap repricing + cut (machinery ~1.4) | ≥1.0 s |
| T17 | 9e4c79da (fable-max) | st8: attribution + reduce fly→table + fold pipeline | ≥1.0 s |

Worktrees: .worktrees/metal-w17-{scangap,st8} off a76d08859. w16
worktrees removed.

### Wave-17 GATE (2026-08-27 ~09:15 ET, trunk 5f7b6b674) — median 36.55, 2^25 RECORD 11.35

- **Lane G17: PARTIAL RETAIN, merged (`09e89eeb4` → bfa7a911e).**
  Shipped, all byte-identical (proof hashes equal ON/OFF @2^21+2^22):
  run-offset early-exit segmented scans (one ballot of run starts
  replaces per-step key shuffles; bounded by longest equal-key run;
  JOLT_IRR_SCAN_FIXED_STEPS=1 keeps both w12 fixed bodies as arms) +
  suffix tile pre-gather/run hoisting (sort once per tile, 3 uint4
  shuffles replace per-suffix Fr gathers) + **prepare fold**
  (`jk_irr_eq_outer` builds u_evals ON DEVICE as dispatch 0 of the
  phase-0 CB, committed DETACHED at prepare while RamRA+RegVal prepares
  0.46 s run on host; IRR prepare 195.7→16.7 ms @2^25, phase0_wait
  <5 ms; JOLT_IRR_PREPARE_FOLD=0 restores; Corrupt ⇒ host rebuild
  fallback intact). KernelId 89→**92**. **Scan branch/emission
  machinery repriced to component level and CLOSED as a door:** phase
  machinery 2.8-3.5 ms/CB (~33%), suffix 2.2 (~37%); hardware already
  skips all-lanes-false add steps (fixed−early-exit = 0.15/0.26 ms =
  dead-step shuffle mass only); live steps are the Fr adds the
  reduction requires (18.9% uniform tiles force full depth). Five
  sub-doors killed with receipts (see kill list). In-shape headroom
  ≲0.2 s @2^27. Report: lane-reports/metal-w17-scangap.md.
- **Lane T17: double NO-GO with receipts, merged (`66039fab8` →
  5f7b6b674; tree byte-identical to trunk — report + receipt history
  only).** st8 attribution @2^27 at ZERO GPU cost (parsed the w16 gate
  trace): 4.795 = combine_hints 0.546 + open 4.196 (preamble 0.518
  [0.257 untraced host + 0.157 HOST G1 MSMs + 0.092 fixed-base] + m1
  1.533 + m2 1.288 + folds 0.856). **Door A (reduce-shape fly→table)
  KILLED:** reduce fly already 1.3-1.5 µs/pair; table floors below
  ~16k threads; flatten sits on the critical path; +2.19 GiB transient
  ⇒ ≈0-negative. **Door B (fold→message pipeline) KILLED premise-false:**
  open +0.42 s (11.95 vs 11.49) — the fold wall is device ALU, not
  hideable latency; **GPU∥GPU overlap conserves work — overlap doors
  must pair GPU with host/IO** (journal corollary: no GPU-side slack
  inside open). Banked residuals: preamble G1 MSMs→device ~0.10-0.12 ·
  untraced preamble 0.257 · combine_hints 0.546 split unknown.
  Receipts in lane history (1625610f7/9644219a8). Report:
  lane-reports/metal-w17-st8.md.
- **Battery green** (clippy host + host,zk + kernels/eval
  metal+bench-utils · metal 414/414 · ratchet 20/20 · legacy
  444/480/445 · verifier stack 285 · release build).
- **Cert (FrBind 250.4 µs):** kill-switch ABBA @2^27, OFF =
  `JOLT_IRR_PREPARE_FOLD=0 JOLT_IRR_SCAN_FIXED_STEPS=1`: ON 36.26/36.91
  vs OFF 36.87/37.04 ⇒ **wave-17 effect −0.37 s mean** (pairs
  −0.61/−0.13, both ON-faster; model −0.5..−0.7). Post-battery warmup
  39.03 discarded per standing rule. **RSS 69.24 GiB (74,345,234,432)
  — down 2 GiB vs w14's 71.18;** G17's u_evals schedule-wire covered
  (st5 dropped as modeled — the 4.3 GiB caveat did not bite).
- **RECORDS: 2^27 clean N=5 36.26/36.50/36.55/36.91/37.22(traced),
  median 36.55** (w16 median ~37.2; all-time best 35.29 stands —
  cooler window). **2^25: 11.35 s — RECORD** (11.39/11.35).
- **Wave-17 stage vector @2^27 (traced 37.22): st5 5.90 (G17 −0.45,
  model →5.7 ✓)** · st0 7.49 · st8 4.80 (open 4.20) · st4 4.83 ·
  st6b 4.49 (+6a 0.15) · st1 3.21 · st2 2.74 · st3 2.36 · st7 1.22.

### Wave-16 lanes

| lane | task | scope | bar |
|---|---|---|---|
| D16 | 618bcaa0 (fable-max) | st0 driver unbank (R12 door #3, precondition met) | ≥1.0 s |
| B16 | c202b643 (fable-max) | st6b re-attribution + SLC-tiling verdict | ≥1.0 s |

Worktrees: .worktrees/metal-w16-{driver,st6b} off 8afbfe83f. w15
worktrees removed.

### Wave-16 GATE (2026-08-27 ~06:15 ET, trunk b7de9afea) — 35.29 s / 3.80 MHz

- **Lane D16: RETAIN, merged (`8001d8331` → fc5257e84).** Regime
  verified at ZERO GPU cost (parsed the surviving w15 gate trace):
  driver 7.49 host + 0.81 send_wait = the whole 8.31 stream window; GPU
  6.59 busy with 1.7 slack — R12's precondition held exactly. Shipped:
  builder-lane overlap (builds+send on a dedicated scoped lane, double-
  buffered staging sets, +29.6 MB deterministic; JOLT_METAL_DRIVER_
  OVERLAP=0) + MILLER_CPU_FRACTION_DEFAULT 0.05→0.0 (tier-2 cpu_absorb
  1.94 s moved to device slack; env knob restores). @2^25 pair −0.38
  (11.01 — then-record-class). Doors closed: build-fusion residuals
  (builder lane has 2.4 s slack — CPU cuts move no wall), collect
  reschedule (absorbed post-overlap). Report: lane-reports/
  metal-w16-driver.md.
- **Lane B16: RETAIN + SLC door CLOSED PERMANENTLY, merged
  (`c691c4636` → b7de9afea).** Re-attribution overturned the w3-era
  model: st6b = Bytecode sync rounds 2.93 (billed the whole detached
  Bool/RAV queue) + serial host prepares 0.89 + IncCR/RamHB sync 0.66;
  the parked door's gather walls are 0.03-0.17. **Bandwidth roof:
  gathers move 10-22 GB/s vs 400 GB/s DRAM — ALU/latency-bound, NOT
  DRAM-bound; the only DRAM-roofed kernel (IncRound 370 GB/s) streams
  once with zero reuse ⇒ SLC tiling has nothing to cache. gpu-util
  parked door #3 closed.** Cut: three sync members get the two-phase
  detach + all four prelaunch round 0 under the prepare prelude
  (scheduling only; JOLT_ST6B_DETACH=0 restores sync bit-exactly).
  st6b @2^25 −9.5%, round loop −26%. Bonus: engine m_begin/m_collect
  member spans. Parked: BytecodeLazyRound factor-specialization
  (1.7-3.6 vs siblings' 7-12 Gmul/s; ~0.3-0.7), RamRAV base anomaly.
  Report: lane-reports/metal-w16-st6b.md.
- **Battery green** (clippy host+zk · metal 414/414 · ratchet 20/20 ·
  legacy 444/480/445 · verifier stack 285).
- **Cert (FrBind 251.0 µs, improving early-morning window):** combined
  ABBA (OFF = OVERLAP=0 + FRACTION=0.05 + DETACH=0): clean pairs ON
  37.63/37.12 vs OFF 38.53/38.71 (B1 38.96) ⇒ **wave-16 effect ≈
  −1.35 s** (pairs −0.90/−1.59; first pair dropped: ON 40.59 was the
  post-battery first-run artifact — THIRD occurrence, rule extended:
  never count the first 2^27 run after a build/battery session as an
  ABBA arm, instrumented or not; models −2.2..−3.0 combined ⇒ the two
  lanes' reclaimed GPU-idle partially overlaps).
- **RECORDS: 2^27 best 35.29 s / 3.803 MHz** (clean N=4 35.29/37.12/
  37.21/37.63, median ~37.2; traced 36.02). **2^25: 11.36 s.**
- **Wave-16 stage vector @2^27 (traced 36.02): st5 6.35 (18%, dominant
  again — scan tail 4.12 vs 2.7 floor)** · st0 7.14 (D16 −1.25; GPU
  floor ~6.7 per D16 — st0 near-exhausted) · st8 4.79 (open 4.20) ·
  st4 4.71 · st6b 4.02 (B16 −0.87) · st1 3.23 · st2 2.69 · st3 2.01 ·
  st7 0.91.

### Wave-15 lanes

| lane | task | scope | bar |
|---|---|---|---|
| M15 | 25f361f3 (fable-max) | Miller table tiling (bounded-residency ALU win) | ≥1.0 s |
| P15 | 842e2085 (fable-max) | st5 residual bundle (cycle waits, presort pricing) | ≥1.0 s |

Worktrees: .worktrees/metal-w15-{tiling,st5res} off 2b0959e30. w14
worktrees removed.

### Wave-15 GATE (2026-08-27 ~04:30 ET, trunk acb7cb95d) — 38.81 s / 3.46 MHz, SUB-39

- **Lane M15: RETAIN, merged (`1e5e3d7f2` → 42ba02deb).** Tiled Miller
  table: each flush gathers its ~4681 unique rows into a recycled 78 MB
  tile (≤2 live ⇒ ~160 MB transient, scale-invariant) and dispatches
  jk_miller_table unchanged at tile width. w13's MILLER_TABLE_MAX_ROWS
  gate + whole-table flatten DELETED — tiled is default at every scale.
  Kernel: tiled 1.03 µs/pair at 2^27 geometry vs whole-table 1.16 vs
  fly 1.93. @2^25 A-B −0.51 s, RSS 22.99 vs 24.82. Kill switch
  JOLT_METAL_MILLER_TILING=0 (fly). Doors (b) on-device prep / (c)
  hybrid closed on price. Report: lane-reports/metal-w15-tiling.md.
- **Lane P15: PARTIAL RETAIN, merged (`375558602` → cabc1a6d6).**
  Re-attribution: st5 6.764 = scans 4.12 (STATUS's 3.0-3.3 was low) ·
  cycle waits 0.854 · init 0.522 · IRR prepare 0.846 · RegVal prepare
  0.345. Shipped: side-queue cycle-buffer pre-wire (fresh no-copy wires
  cost ~50 GB/s at CB schedule; shared queue steals 1:1 from phase CBs
  — side queue is clean; JOLT_IRR_CYCLE_PREWIRE=0) + fused cycle init
  (rows read 1× not per-table, init CB GPU −62%; KernelId 88→89;
  JOLT_IRR_CYCLE_INIT_SPLIT=1) + parallel buckets. st5 −10% @2^25.
  Doors CLOSED w/ receipts: cycle-round kernel at 75-80% ALU roof
  (wait-merge has nothing left — the 0.854 wait = roofed exec + RegVal
  co-run); **global presort KILLED** (59.5% unique full indices on real
  2^24 rows, p90 run=1 — S12's run-length premise doesn't extend; ≈−0.4
  max at max blast radius; supersedes the −1.1 park). Report:
  lane-reports/metal-w15-st5res.md.
- Papercut fixed at gate: unfulfilled clippy::panic expect in RegRW
  bench module (both lanes flagged; acb7cb95d).
- **Battery green** (clippy host+zk incl. metal,bench-utils combo ·
  metal 414/414 · ratchet 20/20 first pass · legacy 444/480/445 ·
  verifier stack 285).
- **Cert (FrBind 252.1 µs):** combined kill-switch ABBA (OFF =
  TILING=0 PREWIRE=0 INIT_SPLIT=1): clean pairs ON 39.14/38.81 vs OFF
  40.43/40.25 ⇒ **wave-15 effect −1.3..−1.4 s** (first pair excluded:
  ON 41.85 carried /usr/bin/time + first-run-after-build cold state —
  same artifact as w14's first pair; NEW MEASUREMENT NOTE below). OFF
  arm ultra-stable 40.10-40.43 = wave-14 class confirmed.
- **RECORDS: 2^27 best 38.81 s / 3.459 MHz** (N=5 38.81/38.95/39.14/
  39.23/[41.85 artifact], clean median ~39.05; traced 38.98). **2^25:
  11.85 s — SUB-12.** RSS ON 71.87 GiB (prewire +0.7 visible, benign).
- Measurement note (standing): never count the RSS-instrumented
  first-run-after-build as an ABBA arm — do the /usr/bin/time run
  outside the paired sequence (w14+w15 both showed +1.5-2 s on that arm).
- **Wave-15 stage vector @2^27 (traced 38.98): st0 8.39 (22%, dominant
  — tiling −1.03 measured)** · st5 6.46 · st6b 4.89 · st4 4.81 · st8
  4.73 · st1 3.44 · st2 2.66 · st3 2.14 · st7 1.28. **R12's driver-
  unbank precondition (device drops ~1 s) is NOW MET.**

### Wave-14 lanes

| lane | task | scope | bar |
|---|---|---|---|
| S14 | d4af2738 (fable-max) | st1 attribution + cut (5.07 s, never touched) | ≥1.0 s |
| B14 | d4df7e80 (fable-high) | st4 bundle: batch overlap + arena reuse (S11 doors) | ≥1.2 s |

Worktrees: .worktrees/metal-w14-{st1,st4bundle} off e6a7e3225. w13
worktrees removed.

### Wave-14 GATE (2026-08-27 ~02:15 ET, trunk e89e7da42) — 40.20 s / 3.34 MHz

- **Lane S14: RETAIN, merged (`83a866fd8` → e023d6c21).** st1 anatomy:
  outer_t1 2.06 + azbz 1.67 + claims 0.93 — all device ALU, every
  integer row value paying a Montgomery conversion mul before its
  weight mul. Cut: lazy-form kernel twins jk_outer_{t1,azbz,claims}_lazy
  (integer-domain Az·Bz i256, raw-residue weights, one host R-fix) —
  74/81/52 → 20/26/17 mont-muls/row; kernels −31/−34/−48% @2^24.
  KernelId 85→88; kill switch JOLT_METAL_OUTER_LAZY=0; both arms
  byte-identical e2e (metal-armed ratchet 20/20 in BOTH arms). Stale
  parked door removed: claimed_inputs port landed Aug 4 (9d9362aa6).
  Parked: t1 grid-stride multi-row accumulation (unpriced). t1+azbz
  fusion dead by protocol order (uniskip challenge after t1 message).
  Report: lane-reports/metal-w14-st1.md.
- **Lane B14: RETAIN both doors, merged (`88bb15ea4` → e89e7da42).**
  Overlap: RegRW slot opts into engine begin/collect — detached fused
  bind+message CB per cycle round, RamValCheck CPU rounds underneath
  (JOLT_REGRW_OVERLAP=0 kills). Arena: entry CSRs as byte slabs, retired
  input slab recycled as later output (JOLT_REGRW_ARENA=0 kills). st4
  @2^24 0.69→0.545 (−21.5%), super-additive. +1 parity test pinning the
  detached schedule (metal 413). Report: lane-reports/metal-w14-st4bundle.md.
- **Battery green on merged trunk** (clippy host+zk · metal 413/413 ·
  ratchet 20/20 first pass · legacy 444/480/445 · verifier stack 285).
- **Cert (FrBind 254.4 µs, noisy overnight window):** combined
  kill-switch ABBA A-B-B-A-B-A: ON 43.42/41.77/40.52 (41.90) vs OFF
  42.80/44.60/44.07 (43.82) ⇒ **wave-14 wall effect −1.92 s** (pairs
  +0.62/−2.83/−3.55 — first pair was a cold-start artifact w/ RSS
  capture attached; clean pairs agree). Models summed −3.1..−3.5; the
  recurring ~55% wall-transfer floor.
- **B14 residency flag RESOLVED benign: peak RSS 71.18 GiB ON** (below
  trunk's 72.43 — slab reuse beats fresh-mmap churn at peak; the
  +5-6 GiB st4 transient sits under the global peak).
- **RECORDS: 2^27 best 40.20 s / 3.338 MHz** (traced run; untraced best
  40.52 / 3.312; **N=5 median 41.77**). **2^25: 12.55 s** (12.74
  opener; table arm + w14 cuts).
- **Wave-14 stage vector @2^27 (traced 40.2 s): st0 9.42 (23%,
  dominant, unmoved — device-paced floor)** · st5 6.66 · st4 4.89
  (−0.60 measured) · st8 4.87 (Dory open 4.35) · st6b 4.63 · **st1 3.23
  (−1.84 measured — S14 over-delivered)** · st2 2.78 · st3 2.10 ·
  st7 1.44 · st6a 0.14.

### Wave-13 lanes

| lane | task | scope | bar |
|---|---|---|---|
| M13 | e6054a92 (fable-max) | commit-shape Miller roof repricing + cut (3.5 CB-s) | ≥1.0 s |
| T13 | 138a32bf (fable-max) | tier-1 batched-affine tree (X9 parked door) | ≥1.0 s |

Worktrees: .worktrees/metal-w13-{miller,tier1} off b5a8e3878. w12
worktrees removed.

### Wave-13 GATE (2026-08-26 ~23:55 ET, trunk fd3177b24) — scale-transfer trap caught

- **Lane T13: NO-GO, door closed PERMANENTLY, merged receipts
  (`db8ad73bf` → e69778b57, zero production diff).** Batched-affine tree
  is inversion-amortization-dead at SIMT: measured Fq inversion 388-401
  mul-equiv vs break-even I≤29 (~150× off bar); full tree 2.52× SLOWER,
  single-inversion hybrid 1.10×, 32 KiB TG-staged 5.29× (occupancy was
  the LESSER gate). Batch inversion amortizes per-thread-sequential K
  only; segment shape caps K=64 with 7 inversions/segment. **Tier-1
  in-kernel is EXHAUSTED** (kernel ≥91% of thread-limited roof; XYZZ
  10-mul madd is the inversion-free optimum). Receipts rig flagged for
  PR-handoff audit. Report: lane-reports/metal-w13-tier1.md.
- **Lane M13: kernel-level RETAIN (`cad0feff1` → e6685a16e), but the
  2^27 GATE INVERTED the lane receipts** — shipped behind a measured
  scale gate (`fd3177b24`). Lane: commit Miller fly→jk_miller_table
  (setup-owned prepared coeffs, cap 32) — all Miller kernels sit in the
  same spill-bound band (fly has zero in-kernel headroom); the table
  computes 2.2× less ALU/pair; commit-Miller CB @2^25 −38%, e2e −0.53.
  Gate ABBA @2^27: table 43.96/47.66/44.00 (45.21) vs fly 42.69/42.70/
  42.46 (42.62) ⇒ **table +2.59 s REGRESSION at flagship scale** — the
  2.1 GiB row-scaled table's DRAM/residency traffic stretches
  co-running CBs (R12's additivity, now in the cost direction) past the
  kernel gain in the 72 GiB working set. Fix: default = table ≤2^15
  one_hot_rows, fly above; `JOLT_METAL_MILLER_COMMIT_FLY` forces (1=fly,
  0=table). Confirm: 2^27 default 42.67 (fly class restored) · 2^25
  default 13.09/13.17 (table arm; cooled window — the −0.5 books at the
  next fresh window). Report: lane-reports/metal-w13-miller.md.
- **Net wave-13 @2^27: 0.00 s by design** (flagship arm unchanged);
  @2^25 ≈ −0.5 pending window. RECORDS UNCHANGED: 42.43 / 13.00. RSS
  @2^27 measured 72.43 GiB (M13's +2.2 transient absorbed; fly arm at
  2^27 doesn't build the table at all).
- **Battery green** on e6685a16e (full) + fd3177b24 (clippy host+zk,
  kernels-metal 261/261, ratchet 20/20; 15-line dispatch-only patch).
- FrBind 257.8 µs at cert; ~24 h of sustained GPU load — window
  degraded from the 42.4 class to ~42.7 by session end.

### Wave-12 lanes

| lane | task | scope | bar |
|---|---|---|---|
| S12 | a2e2663d (fable-max) | st5 scan kernels: X9-method roof repricing, then cut | ≥1.0 s |
| R12 | 394d2e9a (fable-high) | st0 post-S0/D2/E8/X9 re-attribution (attribution-only) | deliverable |

Worktrees: .worktrees/metal-w12-{st5scan,st0attr} off 440a7d07c. w11
worktree removed.

### Wave-12 GATE (2026-08-26 ~21:45 ET, trunk d907ec7e2) — 42.43 s / 3.16 MHz

- **Lane S12: GO ~7× over bar, merged (`2689cd009` → d907ec7e2).**
  Scans were 5× above their compute+loads floor. Roof factorization
  @2^24 real sha2 rows: phase CB 49.3 ms = 2.3 loads (memory DEAD at
  ~770 GB/s) + 7.4 field ALU (starvation ≤1.4×) + 7.7 detect/branch +
  **31.9 ms (65%) in scatter3's 32-source Fr shuffle-reduce flush** —
  80% of tile-iterations flush with ~24 colliding lanes on production
  entropy (w5's run-length win was fib-shaped; sha2 d≈8-18 degenerates
  it to ~eager). Suffix 87% emission machinery. Cut (byte-exact by
  fr_add regrouping): flush = bitonic (key,lane) sort + vec4 gather +
  5-step segmented scan; uniform-butterfly path deleted; 512→4096 sgs +
  two-level RAF reduce; suffix sorts once per tile. Phase −79%, suffix
  −78% @2^24; scan CBs @2^25 e2e 1.736→0.535 s (−69%). Kill switches
  JOLT_IRR_PHASE_SCAN_SCATTER=1 / JOLT_IRR_SUFFIX_SCAN_GROUPED=1 /
  JOLT_IRR_PHASE_SCAN_SGS=512 (full trunk restore). KernelId 83→85.
  Proof bytes identical across arms @2^21+2^22 (= w10 hashes). Report:
  lane-reports/metal-w12-st5scan.md.
- **Lane R12: report merged (8317c1a78, zero code diff).** st0 flipped
  regimes: DEVICE-PACED near co-bound (GPU starvation 7.86→1.10 s;
  send_wait 0.08→2.05). **Miller device time is ADDITIVE, not hidden**
  (co-running G1 CBs stretch 0.21→5.14 µs/seg; union ≈ serial sum) —
  w5's "tier-2 fits underneath" framing is dead. X9 landed as wall
  (gpu_run 9.44→7.66). Doors: Miller fly compute 3.5 CB-s (~−1.5 if
  halved); tier-1 pure 4.2-4.4 (X9's parked batched-affine tree ~−1.5);
  driver builds 4.18 wall/22.5 CPU-s unbank once device drops ~1 s.
  Paired device+driver wave models st0 ≈ 7.0-7.5. Report:
  lane-reports/metal-w12-st0attr.md.
- **Battery green on merged trunk** (clippy host+zk · metal 411/411 ·
  byte-diff 20/20 first pass · legacy 444/480/445 · verifier stack 285 ·
  release build).
- **Cert (FrBind 252.6 µs):** kill-switch ABBA A-B-B-A-B-A @2^27:
  ON 43.10/46.66/42.58 (44.11) vs OFF 48.31/48.80/49.84 (48.98) ⇒
  **wave-12 wall effect −4.87 s** (pairs −5.21/−2.14/−7.26; middle pair
  caught a warm ON run; 90 s cooldowns stabilized the tail).
- **RECORDS: 2^27 best 42.43 s / 3.163 MHz, median 42.67 (N=5:
  43.10/46.66/42.58/42.67/42.43)** — the 3 MHz barrier fell; traced run
  42.58. **2^25: 13.00 s** (13.12 opener). RSS not re-measured this
  gate (sort lives in TG memory; no allocation-shape change expected).
- **Wave-12 stage vector @2^27 (traced 42.58 s): st0 9.41 (22%, NEW
  DOMINANT; stream_witnesses 9.30, collect 3.78 overlapped)** · st5
  6.69 (was 12.74, −6.05 measured) · st4 5.49 · st1 5.07 · st8 4.87
  (Dory open 4.31) · st6b 4.60 · st2 2.76 · st3 2.04 · st7 1.47.
- S12 doors closed w/ receipts: grouped-butterfly flush (+20..222%,
  w5's d≥8 boundary confirmed), vec4-packing scatter (nil), RMW batching
  (RMWs free), width (flat), sgs>4096. Parked: global presort ~−1.1 s
  more (big blast radius); IrrCycleRound 0.84 untouched.

### Wave-11 lanes

| lane | task | scope | bar |
|---|---|---|---|
| S11 | 0c677c56 (fable-max) | st4 re-attribution @2^27, then cut the top item | ≥1.0 s |

Worktree: .worktrees/metal-w11-st4 off 3fc420117. w10 worktree removed.

### Wave-11 GATE (2026-08-26 ~19:50 ET, trunk 7ee9ec9e9) — SUB-50 @2^27

- **Lane S11: RETAIN, merged (`611214610` → 7ee9ec9e9).** st4 anatomy
  @2^27 (8.16 traced): **host memset 2.84 s top** — `plan_bind`'s
  PageAlignedVec::from_elem serially zero-filling each round's fresh
  multi-GiB output CSR (~44 GiB/proof) that the GPU overwrites; bind CB
  1.18 · msg CB 0.82 · RamValCheck CPU 1.51 · prepare 1.46. Cut: bind
  outputs → own_mmap(MmapVec::zeroed) (lazy kernel zero + munmap on
  drop — the w3 prepare pattern). @2^24 ABBA st4 −0.120 s (alloc
  0.183→0.001, +31% fault clawback in bind CB); modeled −1.95 @2^27.
  Byte-identical by construction; kill switch `JOLT_REGRW_MMAP_BIND=0`.
  KernelId 83 unchanged. Report: lane-reports/metal-w11-st4.md.
- **Battery green on merged trunk** (clippy host+zk · metal 411/411 ·
  byte-diff 20/20 first pass · legacy 444/480/445 · verifier stack 285 ·
  release build).
- **Cert (FrBind 246.5 µs, evening record-class window):** kill-switch
  ABBA A-B-B-A-B-A @2^27: ON 48.97/50.35/51.96 (50.43) vs OFF
  55.88/53.50/54.59 (54.66) ⇒ **wave-11 wall effect −4.23 s** (pairs
  −6.91/−3.15/−2.63, all negative, magnitude drifts with window warmth;
  well above the −1.95 span model — the OFF arm's 44 GiB serial
  zero-fill costs beyond its span: page-fault/compressor side effects
  the 2^24 model can't see).
- **RECORDS: 2^27 best 48.92 s / 2.744 MHz** (traced run! untraced best
  48.97; N=5 untraced 48.97/50.35/51.96/51.72/50.21, **median 50.35**
  vs old 52.87; old best 50.56). **2^25: 14.70 s** (old 15.21; second
  run after 15.51 opener; lane-paired 13.88 remains the morning-window
  indication). st4 in-vector 8.04→**5.53** (−2.51 measured).
- **Wave-11 stage vector @2^27 (traced 48.9 s):** st5 12.74 (26%,
  dominant) · st0 9.55 (stream_witnesses 9.45; collect 3.74 overlapped) ·
  st4 5.53 · st1 5.10 · st8 4.90 (Dory open 4.35) · st6b 4.56 · st2
  2.67 · st3 2.35 · st7 1.34 · st6a 0.13.
- Doors from S11 (parked, sub-bar): stage-4 batch overlap (RamValCheck
  CPU serializes behind RegRW CB waits) ~0.8 s — this is also the
  honest re-pricing of the w3 fusion NO-GO; mmap fault-clawback arena
  reuse ~0.9 s. W2B "middle ground" representation door SUBSUMED
  (representation-side host cost now ≈0).
- User-order artifacts landed this session: PR #1733 = main+1
  (`8ba8f1121` force-pushed after lane 9565af1a's merge+battery, 415/415
  metal + 21/21 upstream ratchet + zk 14/14); standalone mapper PR
  **a16z/jolt#1809** (d660cf127). ssh-agent lost its key post-restart —
  pushes via gh https credentials.

### Wave-10 lanes

| lane | task | scope | bar |
|---|---|---|---|
| S10 | c10da8f0 (fable-max) | st5 post-S5 re-attribution @2^27, then cut the top item | ≥1.0 s |

Worktree: .worktrees/metal-w10-st5 off c857970ba. w9 worktree removed.

### Wave-10 GATE (2026-08-26 ~11:30 ET certs + 14:45 ET journal, trunk 197029a8d)

- **Lane S10: RETAIN, merged (`c1f9529e4` → 197029a8d).**
  RegistersValEvaluation fused bind+eval device port (new slot +
  `jk_registers_val_round`; KernelId::ALL 82→83), default-on, kill switch
  `JOLT_METAL_REGVAL=0`. Proof bytes identical ON↔OFF @2^21+2^22; metal
  suites 411/411 (5 new parity tests). Lane numbers @2^27 matched
  record-class windows: st5 13.93→12.96 (−0.97 wall; component −1.33:
  RegVal spans 2.53→0.04, exposed IrrCycleRound CB waits +0.84, prepare
  +0.32). **Peak RSS 74.05→72.36 GiB (−1.7)** — mmap tables freed at gate
  handoff. Report: lane-reports/metal-w10-st5.md.
- **Phase-1 re-attribution (ranking overturned with receipts):** st5
  @2^27 = 13.93 s → scan CBs 9.83 (×16, 99.9% in-kernel GPU) ·
  RegVal host 2.53 · cycle_init 0.52 · RegVal prepare 0.30 · IRR prepare
  self **0.24** (the w5 "1.84 s prepare" span total included phase-0
  scan+eq; E8's packed vsr-run had already killed the copy). CB-merge
  door: gaps 2.5-3.4 ms, wrap+encode 12 ms total ⇒ ~40 ms headroom, DEAD.
- **Battery green on merged trunk** (clippy host+zk · metal 411/411 ·
  byte-diff 20/20 first pass · legacy 444/480/445 · verifier stack 285 ·
  release build).
- **Cert (FrBind 253.8 µs; late-morning window):** kill-switch ABBA
  A-B-B-A @2^27: ON 51.50/50.97 (51.24) vs OFF 53.70/54.24 (53.97) ⇒
  **wave-10 wall effect −2.73 s** (pairs agree −2.20/−3.27; exceeds the
  −1.33 component model — window nonlinearity compounds, as w3/w6).
- Record set interrupted at N=3 by the daemon restart: ON walls
  50.97/51.26/51.50 (median 51.26) — **no new 2^27 record** (50.56
  stands) but the 51.x class is now repeatable in ordinary good windows.
  2^25 gate rerun deferred (box occupied by user-ordered integration
  lanes); lane-paired 14.19→13.88 indicates the 15.21 record falls at
  the next quiet window.
- **User orders (14:38 ET, post-restart), executed this session:**
  (1) trunk pushed to `origin/feat/metal` (PR #1733) @ b33353c51;
  (2) integration lane 9565af1a: merge onto origin/main @1e25e9703
  (#1792 field-stack refactor 535 files + #1732 Akita port + #1762 fused
  loads; 105 overlapping files) — literal 260-commit rebase ruled
  infeasible, strategy = one 3-way merge + battery + squashed
  rebase-shaped artifact, orchestrator pushes after review;
  (3) extraction lane ce4abab2: BytecodePCMapper packed-slot fix
  (d75bcb948) → standalone draft PR off origin/main (user note: CUDA
  inherits it on any future rebase). Wave-11 spawn HELD until the box
  is quiet (integration batteries pollute timed iteration).

### Wave-9 GATE (2026-08-26 ~09:45 ET, trunk 096cb7059)

- **Lane X9: RETAIN, merged (`1ffe9388c`).** w3 framing overturned:
  memory DEAD as limiter, ILP dead; the 4.5× gap = thread starvation ×
  simdgroup divergence × TG-256 packing; roof repriced to thread-count-
  limited rate (post-cut kernel at 91% of it). Cuts: length-sorted
  bounds-triple ABI + TG-64 (kernel −27.7%), segment cap 256→128 (w3
  rejection overturned on the post-D2/E8 tier-2 lane, −7.7% in-pipeline).
  Tier-1 family CBs @2^25 **−30.0%** (3.578→2.505 s); e2e @2^25 −0.92 s
  paired. Kill switches JOLT_METAL_G1_{SORT=0,TG_WIDTH=256,
  SEGMENT_LEN=256}. Parked w/ mechanism: batched-affine tree (~6 vs 10
  muls/add, only remaining >20% in-kernel door, gated on TG-memory
  occupancy). Bench rig eea04d67e flagged for PR-handoff audit.
- **Battery green** (clippy ×2 · metal 406/406 · byte-diff 20/20 FIRST
  pass again post-URS-fix · legacy 444/480/445 · misc 285).
- **Cert (FrBind 255.6 µs, probe 2.90 s — but daytime sustained window
  ≈53-55 s class):** kill-switch ABBA ×3 pairs @2^27: OFF 55.30/53.62/
  54.02 (54.31) vs ON 53.21/53.47/53.42 (53.37) ⇒ **wave-9 wall effect
  −0.94 s** (below the −2.0 model — the recurring ~50% wall-transfer
  floor). No new 2^27 record (50.56 stands, w8 freak window). **2^25:
  15.21 s — RECORD** (old 15.62).
- Note: probe ≤3.40 no longer predicts the sustained-2^27 window class
  (2.90 probe → 53 s runs vs 2.96 probe → 50.56 that morning); records
  are opportunistic, paired ABBAs remain the code-effect evidence.

### Wave-9 results log

- **Lane URS: RETAIN, merged (`2954f0ce1` → 3455ef226). Gate-flake
  SOLVED — URS-race hypothesis FALSIFIED with receipts** (all dory_N.urs
  byte-untouched across 4 runs incl. a flaking one; both setup paths
  already lock load-or-generate-save). **Real mechanism: guest-ELF
  uplift race** — every `jolt build` is fingerprint-dirty, cargo
  re-uplifts (unlink→hardlink) the shared muldiv-guest ELF on every
  invocation; 5 parallel tests share the dir; a reader in a sibling's
  unlink window panics ("could not open elf file") at 1-5 s. Explains
  fast-fail, rotating victim, isolated/rerun pass. Fix: exclusive
  advisory lock on <guest_target_dir>.lock across build+read + ELF byte
  caching in Program (legacy host only, no proof-byte change). Proof:
  3× consecutive 20/20 + 406/406 + clippy both modes. [Corrects the
  wave-7 gate's URS-race hypothesis.]

### Wave-8 GATE + ABSOLUTE RECORD (2026-08-26 ~07:00 ET, trunk 3cc862f59)

- **Lane E8: RETAIN, merged (`d75bcb948` + env-gated telemetry
  61cb71792 + microbench 5d62762bb — delete latter two at PR handoff).**
  Attribution first (clean window, CPU-seconds): wave-7's "driver-bound
  17.33 s" was ambient skew — clean-window trunk st0 = 10.54 s. Real cut:
  `BytecodePCMapper::get_pc` Vec<Vec<>> pointer-chase burned ~85 CPU-s
  @2^27 across BOTH row walks (driver from_row 75%, collect bundle_at
  87%); packed per-address vsr-run u64 → one flat load; from_row
  221.6→3.4 ns/row; @2^25 extract_bucket −53%, collect −76% CPU;
  byte-diff 20/20; RSS neutral; untoggleable (representation swap).
  Levers 1/3/4 moot/NO-GO/banked (report). st0 now device-bound: tier-1
  G1SegSum 7.63 CB-s pacing — the parked w3 XYZZ headroom door (achieved
  2.52 vs 11.30 Gmul/s roof) is the next flagship.
- **Battery green** (clippy host+zk · metal 406/406 · byte-diff 20/20
  first pass — no flake this time · legacy 444/480/445 · misc 285).
- **Cert in an unprecedented window class** (FrBind 253 µs, probe 2.96 s;
  wave-7 code measured 54.94 s vs its own 61.15 record): ABBA W7/W8/W8/W7
  = 54.94 · 52.87 · 53.70 · 61.52 — window collapsed on the last leg;
  honest first-pair delta **−2.07 s** (above E8's −0.6..−1.0 st0-only
  model; collect's −76% CPU frees cores beyond st0). Post-cooldown
  **ABSOLUTE RECORD: 50.56 s / 2.655 MHz**, median 52.87 (N=5).
  **2^25: 15.62 s** (old 16.55). **RSS 72.84 GiB** — back to wave-4
  level (the eager-table cost is now fully offset by E8's transient
  savings + slimming).
- URS hygiene lane was stopped mid-flight to clear the machine for this
  cert (its byte-diff suites voided E8's first profile); resumed after.

Wave 8 CLOSED. Wave-9 doors: (1) tier-1 G1SegSum XYZZ headroom
(7.63 CB-s, 2.52 vs 11.30 Gmul/s roof — evidence-based reopen of the w3
parked door, now the pacing item); (2) URS flake fix (lane resumes);
(3) st5 P8-14 0.7 s; (4) st8 chunked fold→message 0.5-0.7 s.

### Wave-7 GATE + ABSOLUTE RECORD (2026-08-26 ~04:00 ET, trunk b0a463ae6)

- **Lane D2: RETAIN, merged (`1d55e8e51`).** Tier-2 floor eliminated:
  (1) async Miller settle (InFlightMiller, settle one flush later) —
  miller_wait 2.33→0.02 s @2^25, ~5.5→0.03 @2^27; (2) flush 8192→65536 —
  fly 3.32→2.00 µs/pair, miller CB @2^27 5.5→3.50 s (−36%). ABBA @2^25
  −0.42 s e2e; st0 span −31%. Kill switches JOLT_METAL_MILLER_ASYNC=0 /
  JOLT_METAL_MILLER_FLUSH_PAIRS=8192. CPU_FRACTION=0.0 priced, default
  0.05 stays. GPU-fold kernel unnecessary (fold 2.2→0.87 s free).
  **Premise half-overturned: st0 @2^27 is now DRIVER-bound** (st0 17.33 =
  driver 16.62; device 52% idle, GPU lane starved 7.4 s). **S0's fused
  extract measures 10.06 s @2^27 (pre-fusion 8.67) — the fusion's 2^27
  net was ~−0.5 s, not −2.9..−4.4; its small-scale ABBA did not
  transfer.** Wave-8 door #1: extract_bucket contention (collect 13.7
  co-running); tier-2 can't re-bind until driver <~11 s; ~9 s device idle
  available for host→GPU repricing (R's "no more GPU offload" claim was
  for the OLD balance — evidence-based reopen permitted).
- **Battery green** (clippy host+zk · metal 406/406 · byte-diff 20/20 on
  rerun · legacy 444/480/445 · misc 285 · build OK). **Recurring gate
  flake (2 of 2 gates):** one committed_* byte-diff test fails fast in
  the parallel full suite, passes isolated + rerun (wave 5:
  advice_committed; wave 7: committed_muldiv_many_chunks). Mechanism
  hypothesis: dory-pcs URS disk-cache cross-process race (urs_lock
  serializes writes, doesn't version reloads; parallel nextest processes
  at different setup sizes). Wave-8 hygiene item.
- **Cert — best window of the campaign (FrBind 254.5 µs, probe 2.96 s):**
  ABBA @2^27: OFF 61.94/61.57 (61.76) vs ON 61.15/61.36 (61.26) ⇒
  **wave-7 effect −0.50 s** (conservative end of D2's model).
  **ABSOLUTE RECORD: 61.15 s / 2.195 MHz**, median 61.44 (N=6, ±0.4 s —
  tightest cluster booked). 2^25: **16.55 s** (old 16.72). RSS @2^27
  **75.08 GiB** measured (−3.7 vs wave-6's 78.78: slimming −2.05 + D2
  flush effects; +2.8 net vs wave-4's 72.24).
- Window-quality lesson quantified: paired-sum across waves 5-7 ≈ −6.2 s
  but absolute record moved −2.73 s — degraded-window paired deltas
  overstate absolute transfer (wave-3 lesson, now with numbers). Only
  same-window records vs records are comparable.

Wave 7 CLOSED. Wave-8 doors: (1) extract_bucket 10.06 s under collect
13.7 co-run (driver-bound st0); (2) URS-race hygiene; (3) st5 P8-14
0.7 s; (4) st8 chunked fold→message 0.5-0.7 s; (5) host→GPU offload
repricing (~9 s device idle).

### Wave-6 GATE + ABSOLUTE RECORD (2026-08-26 ~02:00 ET, trunk 309f8b09a)

- **Lane S0: RETAIN, merged (`b4ad086c0`).** Fused st0 driver
  (extract→bucket single subchunk-parallel pass, 128 units/superchunk vs
  trunk's 2-4 contended blocks @2^27 geometry): driver spans −26.7%
  @2^24 ABBA / −19.2% @2^25; RSS neutral @2^25; byte-diff 20/20; kill
  switch JOLT_METAL_JOB_SLAB_REUSE=0 (restructure itself untoggleable —
  A/B needs trunk binary). Protocol finds journaled: ABBA + 30-60 s
  cooldowns mandatory at small scales (interleaved-no-cooldown pairs
  manufacture ~+60% fake collect dilation); the "0.6 s st0 re-wrap tax"
  was S5's-st5 attribution, st0 wrap is 7-10 ms; `stream_extract` span
  replaced by `extract_bucket` on the metal path.
- **Battery green** (clippy host+zk · metal 405/405 · byte-diff 20/20 ·
  legacy 444/480/445 · verifier/dory/tracer/witness 284 · build OK).
- **Cert (FrBind 253.7 µs, probe 3.22 s):** two-binary ABBA @2^27 with
  60 s cooldowns: W5 67.81/65.69 (mean 66.75) vs W6 67.33/64.13 (mean
  65.73) ⇒ **wave-6 wall effect −1.02 s in-window** — real but floors
  harder than S0's −2.5..−3.3 model (GPU queue + tier-2 lane deeper than
  modeled; driver cut banks against tier-2 shrinkage).
- **ABSOLUTE RECORD: 62.92 s / 2.133 MHz** (W6 N=7: 67.33 cold · 64.13 ·
  64.33 · 64.09 · **62.92** · 63.07 · 63.85 + 63.09 RSS run; stable-window
  N=5 median 63.85; four runs beat the old 63.88 best). Window tonight
  was ~3-4% off Aug-24 grade early, improving through the session —
  record-grade model for wave-6 code ≈ 60-61 s remains unproven.
- **RSS honest point: 78.78 GiB peak (+6.5 vs wave-4's 72.24).**
  Attribution: T2 eager tables at 2^27 scale (G2Prepared 2^17 ≈ 2-4 GiB;
  lane-T eager pricing was +4.18) + S0 slabs @2^27 geometry. Margin to
  the ~97 GiB storm regime ≈ 18 GiB — accepted. **Cheap reclaim door:
  size the eager G2 prep to the max_rows prefix actually consumed (2^16,
  half the table) ≈ −2 GiB at zero wall cost.**
- Tech debt for PR handoff: commitment.rs now 2347 lines (past the
  1000-line soft flag; split when the campaign trunk goes to PR).

Wave 6 CLOSED. Remaining doors: tier-2 Miller commit-shape (~5 s CB under
st0 + unlocks S0's banked driver headroom) · eager-prep slimming (−2 GiB)
· st5 P8-14 residual 0.7 s · st8 chunked fold→message pipeline 0.5-0.7 s.

### Wave-5 GATE (2026-08-25 ~22:30 ET, merged trunk 00ce20da2)

Merges: reattr b44ea7f95 · trs 13b01608b+463475d38 · st8 4ed633e21 ·
st5 51b18a977; one conflict (KernelId::ALL 80 vs 81 → **82**).

**Battery green:** clippy host + host,zk `-D warnings` · metal suites
405/405 · prover-fixtures byte-diff 20/20 (one flake on first pass,
passed isolated + full rerun; suspect dory-pcs URS disk-cache
cross-process race — pre-existing footgun, papercuts CLI unavailable on
this box) · muldiv 3/3+3/3 · prover-legacy 444/480/445 · verifier/dory/
tracer/witness 284 · release build OK. skip: prover-fixtures,zk — no
`zk` feature on jolt-prover at this trunk vintage.

**Certification (FrBind 255.9 µs healthy; 2^22 probe 3.15 s ≤3.40):**
window degraded under sustained 2^27 load (wave-4 code measured 72.12 s
tonight vs its own 63.88 record) — absolute walls not record-comparable;
paired same-window kill-switch A/B on one binary is the evidence
(campaign rule): **OFF 72.12 → ON 67.40 = −4.72 s / −6.5%**, consistent
with ON-mean 67.3 (N=4) vs OFF 72.1. Exceeds the −3.8 s conservative
model. Record-window model: 63.88 × (67.40/72.12) ≈ **59.7 s
(~2.25 MHz)** — labeled cross-window model; absolute record attempt
deferred to a fresh window (watchdog re-arm = parent action; probe gate
≤3.40 s unchanged). 2^25 pair in-window: 19.02 ON vs 19.17 OFF —
noise-level at this degradation; fresh-window cert will decide.

**Stage vector (traced run 64.61 s wall, ≈ record-trace conditions):**
st0 17.82 (+0.96 vs record trace — ambient host dilation; prepare_tier2
now 0.00 ✓) · st1 5.12 · st2 2.74 · st3 2.30 · st4 8.21 · **st5 15.08
(−1.26)** · st6a 0.19 · st6b 6.51 (+1.22 ambient) · st7 1.61 · **st8
4.95 (−0.87 — matches B's model exactly)**. Dory spans: open 4.47
(miller_fly 2.89, first_msg 1.63, second_msg 1.37), combine 0.45,
setup_prover 51.2 (out of wall, +2.3 from eager tables).

Wave 5 CLOSED. Wave-6 doors ranked: (1) st0 driver host path (extract
8.7 + build 8.3 + collect contention ~5 s; tier-2 lane floors at
14.3 CB-s demand); (2) S5 P8-14 residual 0.7 s + re-wrap 0.6 s;
(3) B's chunked fold→message pipeline 0.5-0.7 s (st8).

## STATUS: PAUSED (user, 2026-08-05) — superseded by wave 5 above

- Waves 1–4 CLOSED; all retained work merged on `scratch/metal-saturation`
  and pushed to PR **a16z/jolt#1733** head `feat/metal` @ `61e5be763`
  (2026-08-05; push gates green: metal release build + kernels/dory/eval
  metal suite 404/404).
- **Absolute record BOOKED 2026-08-24** (post-reboot, record-grade window:
  probe 3.27 s @2^22 ≤ 3.40 s gate): **best 63.88 s / median 64.56 s @2^27**
  (N=5: 63.88 · 64.19 · 64.56 · 65.92 · 66.44; tight ±1.3 s, no bimodality;
  every run e2e-verified). Padded **2.101 MHz** best / 2.079 MHz median;
  peak RSS 72.2 GiB. Beats the 69.63 s wave-2 record by −5.75 s (−8.3%).
  The ~56-58 s window-equivalent model was optimistic ~7 s — the paired-A/B
  −17.5% did not transfer fully to absolute walls even in a clean window.
- 2^25 same window: **16.72 s** (vs 19.01 s wave-2), RSS 25.4 GiB.

## Current state (flagship ledger)

| point | 2^27 | 2^25 | commit |
|---|---:|---:|---|
| campaign baseline | 71.77 s / 1.870 MHz / RSS 76.87 GiB | 19.67 s / 27.42 GiB | 88b063db3 |
| wave-2 gate (best absolute) | 69.63 s / 1.928 MHz / 76.77 GiB | 19.01 s / 25.16 GiB | 3830f4da8 |
| wave-3 trunk, paired A/B | −17.5% vs wave-2 code same-window (78.46 vs 95.11) · RSS −4.7 GiB | 20.05 s in-window | 95511fa07+ |
| **wave-4 trunk — ABSOLUTE RECORD (2026-08-24)** | **63.88 s / 2.101 MHz** (median 64.56 s, N=5) / RSS 72.24 GiB | 16.72 s / 25.44 GiB | d2523b09a |
| wave-5 trunk (METAL-DORY), paired A/B | **−4.72 s / −6.5%** vs wave-4 code same-window (67.40 vs 72.12; ON mean 67.3 N=4 vs OFF 72.1) · record-window model ≈59.7 s | 19.02 in-window (pair noise-level) | 00ce20da2 |
| **wave-6 trunk — ABSOLUTE RECORD (2026-08-26)** | **62.92 s / 2.133 MHz** best; stable-window median 63.85 (N=5: 64.33 · 64.09 · 62.92 · 63.07 · 63.85; 3 runs < old 63.88); ABBA vs wave-5 −1.02 s in-window; **RSS 78.78 GiB (+6.5 vs wave-4 — eager Dory tables + driver slabs)** | — | 309f8b09a |
| **wave-7 trunk — ABSOLUTE RECORD (2026-08-26 04:00)** | **61.15 s / 2.195 MHz** best; median 61.44 (N=6: 61.15 · 61.36 · 61.48 · 61.90 · 61.65 · 61.39, ±0.4 s); ABBA vs wave-6 −0.50 s; **RSS 75.08 GiB** (−3.7 vs 78.78 via slimming; +2.8 net vs wave-4) | **16.55 s** / 27.39 GiB | b0a463ae6 |
| **wave-8 trunk — ABSOLUTE RECORD (2026-08-26 07:00)** | **50.56 s / 2.655 MHz** best; median 52.87 (N=5: 52.87 · 53.70 · 50.56 · 51.37 · 53.54); first-pair ABBA vs wave-7 −2.07 s (position-balanced distorted by mid-sequence window collapse, W7-b 61.52); **RSS 72.84 GiB (≈ wave-4's 72.24)** | **15.62 s** | 3cc862f59 |
| wave-9 trunk | kill-switch ABBA −0.94 s (OFF 54.31 vs ON 53.37, N=3 each, daytime window; 2^27 best remains 50.56 from the w8 freak window) | **15.21 s — 2^25 RECORD** | 096cb7059 |
| wave-10 trunk | kill-switch ABBA **−2.73 s** (ON 51.50/50.97 vs OFF 53.70/54.24, A-B-B-A, pairs agree); ON walls 50.97-51.50 routine in a late-morning window; record N=5 interrupted at N=3 by daemon restart; **RSS 72.36 GiB (−1.7 via S10 mmap handoff)** | lane-paired 13.88 (gate rerun pending quiet window) | 197029a8d |
| **wave-11 trunk — ABSOLUTE RECORD, SUB-50 (2026-08-26 evening)** | **48.92 s / 2.744 MHz** best (traced run; untraced 48.97; N=5 median **50.35**); kill-switch ABBA **−4.23 s** (ON 50.43 vs OFF 54.66, 3 pairs all negative −6.91/−3.15/−2.63) | **14.70 s — RECORD** (opener 15.51) | 7ee9ec9e9 |
| **wave-12 trunk — ABSOLUTE RECORD, 3 MHz (2026-08-26 ~21:45)** | **42.43 s / 3.163 MHz** best, median **42.67** (N=5: 43.10/46.66/42.58/42.67/42.43; traced 42.58); kill-switch ABBA **−4.87 s** (ON 44.11 vs OFF 48.98, pairs −5.21/−2.14/−7.26) | **13.00 s — RECORD** (opener 13.12) | d907ec7e2 |
| wave-13 trunk | **0.00 s @2^27 by design** — T13 batched-affine NO-GO (permanent); M13 Miller table +2.59 s @2^27 REGRESSION caught at gate, shipped scale-gated (table ≤2^15 rows, fly above); RSS 72.43 GiB | table arm ≈ −0.5 pending fresh window (13.09/13.17 cooled) | fd3177b24 |
| **wave-14 trunk — ABSOLUTE RECORD (2026-08-27 ~02:15)** | **40.20 s / 3.338 MHz** traced best (untraced 40.52 / 3.312; N=5 median **41.77**); combined kill-switch ABBA **−1.92 s** (ON 41.90 vs OFF 43.82; clean pairs −2.83/−3.55); **RSS 71.18 GiB** (arena beats fresh-mmap churn at peak) | **12.55 s — RECORD** (12.74 opener) | e89e7da42 |
| **wave-15 trunk — ABSOLUTE RECORD, SUB-39 (2026-08-27 ~04:30)** | **38.81 s / 3.459 MHz** best (clean N=4 38.81/38.95/39.14/39.23; traced 38.98); combined ABBA **−1.3..−1.4 s** clean pairs (OFF arm 40.10-40.43 ultra-stable); RSS 71.87 GiB | **11.85 s — SUB-12 RECORD** | acb7cb95d |
| **wave-16 trunk — ABSOLUTE RECORD (2026-08-27 ~06:15)** | **35.29 s / 3.803 MHz** best (clean N=4 35.29/37.12/37.21/37.63, median ~37.2; traced 36.02); combined ABBA **−1.35 s** clean pairs (−0.90/−1.59; post-build first-run pair dropped) | **11.36 s — RECORD** | b7de9afea |
| wave-17 trunk | kill-switch ABBA **−0.37 s** (ON 36.26/36.91 vs OFF 36.87/37.04, pairs −0.61/−0.13); clean N=5 median **36.55** — best sustained median (best 36.26; all-time 35.29 stands); **RSS 69.24 GiB (−1.9 vs w14)**; T17 st8 doors double NO-GO at zero GPU cost | **11.35 s — RECORD** | 5f7b6b674 |
| wave-18 trunk — PERF PHASE CLOSED | switchable ABBA **−0.22 s** (3 pairs +0.05/−0.61/−0.09; RLC/prep cuts in both arms); **st8 span 4.80→4.42** (combine_hints −46%, MSMs on device, RLC parallel); N=5 median **36.60**, best 35.93 2nd-ever; RSS 71.76 GiB; L18 doors dead premise-false at zero 2^27 cost | **11.21 s — RECORD** | c8cbe6764 |
| wave-19 trunk — NEW AXIS BASELINES | C19 cleanup −8.8k LOC + R19 attribution (no perf change; split sanity 36.68 in-band). **btree @2^27 39.26-39.55 / ratio 8.8× (TARGET) · sha3 37.20-39.22 / 13.1×** vs sha2 36.60 / 10.2×; excess = host st2/st6a/st7; GPU roofs generalize | btree 12.06 · sha3 11.28 · sha2 11.14 | ea6cc017d |

Wave-2 record stage vector @2^27: st0 17.98 · st1 5.45 · st2 2.98 ·
st3 2.30 · st4 9.52 · st5 14.81 · st6a 0.19 · st6b 7.02 · st7 1.33 ·
st8 7.99 (both fresh runs carried ~+6 s st0 ambient penalty).

## Standing rules

- Byte parity lifted; gate = e2e accept + tamper reject + full battery +
  written soundness argument for protocol changes. Naive shared-challenge
  two-round fusion BANNED (diagonal kernel `Δ=γX(X−Y)`, d≥2).
- Velocity v3: iterate 2^22–2^24, ≤2 timed runs per decision (3rd on
  disagreement), full battery + 2^25/2^27 certification once per wave gate.
- Timed A/B must be same-window interleaved (ambient device power/clock
  moves whole distributions ±5..25 s). Only same-window pairs are evidence.
- Cooldown + FrBind health check (<350 µs; healthy ref 255 µs @2^20)
  before certification runs; record-grade window probe: 2^22 e2e ≤3.40 s.
- Single-kernel discipline: harness one kernel via jolt-eval, optimize to
  max, then next. No e2e for kernel iteration.
- Kernel-adding merges: re-count `KernelId::ALL` (currently `[Self; 83]`
  after S10's jk_registers_val_round).
- Kernel microbenches must use production-distribution inputs (real trace
  rows), not random keys — the w3 st5 model error (+5.4 s) was random-key
  fixtures hiding the collision-scatter worst case (w5 lane S5).
- Host-heavy wall spans are window-scaled (observed up to ~1.7×: st0
  17.33 degraded vs 10.54 clean on identical code, w7→w8). CPU-seconds
  are the transferable iteration metric for host-path work; wall spans
  only rank within one window (w8 lane E8).
- Scale-transfer rule (w13): any cut whose resident data grows with rows
  (setup-owned tables, caches) must be kill-switch ABBA'd at 2^27 BEFORE
  default-on — 2^24/2^25 receipts do not transfer (Miller table: −0.53 s
  @2^25 but +2.59 s @2^27; the 72 GiB working set prices residency that
  small scales ride free).
- Gate certs and 2^27 span profiles require ALL sibling lanes stopped —
  including CPU-only lanes (a byte-diff suite voided E8's first profile).
- All cargo under `/usr/bin/lockf -k /tmp/jolt-metal-wave3-cargo.lock`;
  `gpu_lock()` for timed GPU. No pushing without parent's word.
- Gate battery: clippy host+zk `-D warnings` · muldiv host+zk · prover-
  legacy default/zk/akita · verifier · dory · tracer · witness · metal
  suites (`cargo nextest run -p jolt-kernels -p jolt-dory -p jolt-eval
  --features jolt-kernels/metal,jolt-eval/metal`). Build:
  `cargo build --release -p jolt-prover --example modular_benchmark
  --features prover-fixtures,metal`.

## Kill list (permanent, with mechanism)

- tier-1 batched-affine tree (X9's parked door): inversion-amortization-
  dead at SIMT — Fq inversion 388-401 mul-equiv vs break-even I≤29;
  full tree 2.52× slower, all variants ≥1.10×; occupancy was the lesser
  gate. Tier-1 in-kernel exhausted (≥91% of thread-limited roof). (w13)
- Miller prepared-coeff table as the 2^27 commit default: row-scaled
  2.1 GiB residency stretches co-running CBs +2.59 s at flagship scale;
  it remains the default ≤2^15 rows behind the w13 scale gate.
- st5 scan flush variants (w12 S12, all with receipts): grouped-butterfly
  flush (+20..222% — w5's run-length win holds only for d<8 fib-shaped
  entropy), vec4-packing scatter (nil), RMW batching (device RMWs are
  free), width changes (flat), sgs>4096 (no gain). The landed answer is
  bitonic sort + segmented scan.
- st5 scan branch/emission machinery (w17 G17, component ladder on real
  rows): the branches ARE the algorithm at production entropy — hardware
  already skips all-lanes-false add steps; live steps are required Fr
  adds (d≈8-18 keys/tile, 18.9% uniform tiles force full depth). Five
  sub-doors killed: phase-specialized function constants (dispatch-
  uniform branches are 0.09-0.21 ms/CB, rest is data-dependent),
  two-pass detect→emit (per-row descriptors = 1.67 GB round trip > the
  whole 2.8-3.5 ms machinery), 2-slot held-state LRU (flush rate 0.80
  UNCHANGED — simd_any couples 32 lanes), tail-RMW prefetch (+1.0-1.35
  ms register pressure; w12's "RMWs free" needed the scatter's latency
  cover), uniform-flush butterfly shortcut (neutral — saved sort pays
  the extra votes). In-shape headroom ≲0.2 s @2^27.
- st8 reduce-shape fly→table (w17 T17): reduce fly is already
  1.3-1.5 µs/pair (denser than commit fly); per-round tables floor below
  ~16k threads; flatten sits on the critical path; +2.19 GiB transient
  ⇒ ≈0-negative. Commit-shape tiling (M15) does NOT transfer — G2 side
  changes per round.
- st8 fold→message pipeline (w17 T17, measured): open +0.42 s — the
  fold wall is device ALU, not hideable latency; **GPU∥GPU overlap
  conserves work; overlap doors must pair GPU with host/IO.** No
  GPU-side slack inside open.
- st6b BytecodeLazyRound factor-specialization (w18 L18): premise-false
  — isolated exec of ALL bytecode kernels ≈155 ms @2^27; B16's 0.3-0.7
  prize was co-run-window mispricing (60-132 ms windows vs 8-16 ms
  exec); "1.7-3.6 Gmul/s" = op-mix artifact, kernel at compound ALU
  roof. f3 spec −14 ms sub-bar 20×; simd-shuffle +21% / sgbar tree +31%
  (plain barriered tg_sum optimal on AGX). RamRAV anomaly = window
  overlap (per-poly ram RAV 25% cheaper than instr).
- st8 combine_hints post-NAF residual (w18 F18): kernel floor
  ≈0.28-0.38 @2^27 is Fr ALU at ~6.6 Gmul/s; buckets/GLV priced
  sub-bar. wNAF w=2 is the landed answer.
- w5 "tier-2 fits underneath tier-1" framing: DEAD — Miller device time
  is additive under co-run (G1 CBs stretch 0.21→5.14 µs/seg, union ≈
  serial sum). Device-side st0 cuts pay wall directly. (w12 R12)
- st5 scan-CB merge/overlap ("dispatch context" framing): CBs are
  back-to-back (2.5-3.4 ms gaps ×16, wrap+encode 12 ms) — total headroom
  ~40 ms; 99.9% of the 9.83 s is in-kernel GPU time. (w10 S10)
- st5 IRR prepare as a door: self-time is 0.24 s — the w5 "1.84 s" was a
  span-total artifact (included phase-0 scan+eq); E8's packed vsr-run
  already removed the reclaim copy. (w10 S10)
- Global address-major Dory flip: >68× measured @2^22 + sharding inversion.
- st4 cycle-prefix radix-4 on unchanged CSR: measured prefix is
  cycle-domain (wrong variables).
- st4 address-first restructuring, BOTH arms (Gate-1): address phase
  6.30 s binary / 7.10 s radix-4 @2^24 = 42× over the 0.15 s kill line;
  radix-4 loses to binary. State algebra sound, arithmetic dominant.
- st0 walk↔commit scheduling fixes: bimodality is ambient device
  power/clock state (reproduced solo); full fix matrix dead/fail-unsafe.
- W2B round-loop rewrites: +52 GiB footprint or in-place rounds +46.8%.
  (Its device PREPARE build was salvaged in wave 3: −86.1%.)
- Generic radix-4/round-pairing outside st4: slots already fuse bind+eval;
  packed challenges are rank-2 weights, illegal in Dory opening points.
- `malloc_zone_pressure_relief` on freed huge regions: no-op.
- Dory reduce challenge-push/fold-defer (LATTICE fold-chain break, st8):
  NO-GO — every fold state is read directly by the next message's
  pairings; pushing weights through C±/D pairings expands bilinearly
  (+~1.4 s at r0 alone); folds are kernel compute, not sync (67+39 ms/rnd
  @2^16 vs 7 ms host normalize). (w5 lane B, metal-w5-st8.md)
- On-GPU public-matrix regen from seed (LATTICE port): PRICED NO-GO —
  Metal Dory's public base data is 32 MiB total, built in ~9 ms/proof
  (0.06% of st0), zero-copy cache-resident; regen ≥3 orders costlier than
  a 64 B cached read; URS is OsRng-persisted, not seed-derived (regen
  would need a seeded-URS protocol change). (w5 lane T, metal-w5-trs.md)
- Typed-Dory quaternary packing (st6b/st7): PRICED NO-GO — ~0.08 s @2^27
  vs 18-25 lane-days blast radius. Oracle soundness GO stands if geometry
  ever changes.
- st6b gather residual kernels: measured NO-GO — width-1/2/4 lazy gathers
  are the mass; row-batching and SIMD-reduction prototypes both overlapped
  baseline. Attribution harness retained.
- st8 fly persistent-state restructure: PRICED SHUT — spill cliff below
  one Fq6 (fly peak ≈430 u32 live); split ladder +12.7..18.2%.
- st0 TG cap: NO-SHIP — in-pipeline inversion (+2% wall).
- st4 round-loop fusion: NO-GO on top of GPU CSR prepare (−5.7%;
  JOLT_REGRW_FUSED=1 opt-in probe kept).
- st2 RAM-RW device port: below bar (env-gated, default-off).
  **RECEIPT INVALIDATED w19 for btree-class shapes** — priced on sha2's
  trivial ram_K 2^13; btree runs ram_K 2^19 with 3.9× rounds and st2
  +1.86 s. Reopened as wave-20 door #1 (shape-aware re-price).

## Parked doors

- ~~Absolute record run~~ — DONE 2026-08-24 (63.88 s best / 64.56 s median
  @2^27; see STATUS). Residual door: the ~7 s model-vs-measured gap in a
  certified-clean window (probe 3.27 s, tight N=5) — the wave-3 paired
  −17.5% overstated absolute transfer; re-attribute st0/st5 walls if the
  campaign resumes.
- st0 bg12 E-cluster commit + starvation guard: fail-unsafe without guard;
  needs explicit mandate + 2^27 cert.
- Radix-4 packed round oracle-SOUND (3dbb9c10e48a Q1/Q2/Q4) if a PCS-clean
  cheap-state site appears; Val temporal convention pinned in
  metal-w2-r4gate1.md.
- st1 packing: legal but round loop already fused — prize small.
- st8 jk_miller_table −24% at TG cap 32 on commit shape (fly-lane handoff;
  superseded by M15 tiling as the commit default).
- ~~st8 banked residuals (w17 T17)~~ — CONSUMED w18 F18: all three cut
  (st8 4.80→4.42); post-NAF floor kill-listed.
- ~~st6b BytecodeLazyRound + RamRAV (w16 B16)~~ — CLOSED w18 L18
  premise-false; see kill list.
- ~~st5 IrrCycleRound exposed CB waits 0.84 s @2^27~~ — priced/closed
  w15 P15: cycle waits 0.854 sit at the cycle-exec ALU roof; prewire +
  fused init shipped, wait-merge dead.
- st4 batch overlap ~0.8 s (RamValCheck CPU serializes behind RegRW CB
  waits; = the honest re-pricing of the w3 fusion NO-GO) + mmap
  fault-clawback arena reuse ~0.9 s — each sub-bar alone, ~1.7 bundled.
  (w11 S11)
- st0 XYZZ headroom: achieved 2.52 vs 11.30 Gmul/s roof.
- Predecessor campaign's parked doors: see `gpu-util.md` §Parked doors.

## Wave index (verdict one-liners; narrative in archive/)

- **Wave 1** (kernel attribution + first ports): see
  `archive/metal-saturation-waves1-2.md`; canonical attribution report
  `metal-m5-saturation-report.html`.
- **Wave 2** (st6b deferred+fused adoption, st7, st3): record 69.63 s
  @2^27 (−2.14 s vs baseline; st6b 16.3→7.0). Battery green @3830f4da8.
- **Wave 3** (single-kernel: st5 scans, st4 CSR prepare, st6b IncCR
  prepare, st8 parallel fold, st0 XYZZ): certified paired A/B **−16.65 s /
  −17.5%**, RSS −4.7 GiB; battery 353/353 metal. NO-GOs journaled with
  mechanism. `archive/metal-saturation-waves3-4.md`.
- **Wave 4** (st8 dispatch-merge bundle default-on; fly + typed-Dory + st0
  cap doors priced shut): battery 404/404; isolated −0.61 s measured.
  Same archive file.
- Lane detail: `lane-reports/metal-w{2,3,4}-*.md`, `briefs/`.

## Predecessor

GPU-utilization campaign (CLOSED 2026-08-04, mandate met): `gpu-util.md`
(compact) + `archive/gpu-util-campaign-full.md`. Its negative-results
index and parked doors remain binding context for this campaign.
