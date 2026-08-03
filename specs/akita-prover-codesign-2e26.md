# GOAL-MODE HARNESS: Akita packed prover at 2^26

Execute this spec autonomously in goal mode. Work the queue top-down, one
item at a time, following the loop protocol exactly. Do not ask for
permission on queue items marked ENG; STOP and write a design note for items
marked PROTOCOL before implementing them.

## Goal

On `perf/optimize-akita-prover` (jolt) + `perf/onehot-commit-sweep`
(`~/akita`, wired via path deps in jolt's root Cargo.toml):

- **PRIMARY: sha2-chain 2^26 prove ≤ 65 s** (baseline 91.93 s), measured by
  the harness below under measurement discipline.
- STRETCH: ≤ 60 s.
- GUARDRAILS (hard): all gates green after every landed change; verify
  ≤ 350 ms; proof size within the existing 1% catalog slack; peak RSS
  ≤ 95 GB (improving it is item 6, regressing it is a revert); security
  target and norm analysis are FROZEN — nothing that touches
  `SisSecurityPolicy`, the q128 tables, or norm-bound formulas.

Done when: primary goal met AND every queue item is landed or written into
the dead-end ledger with a measurement — or all items are exhausted.

## Closing comparison (2026-07-25, back-to-back cooled, same-night)

| scheme | prove | verify | peak RSS |
|---|---|---|---|
| akita packed (this branch, Q2A+Q6) | **93.55 s** | 192 ms | **87.35 GB** |
| dory (same branch, no akita feature) | 109.46 s | 96 ms | (not sampled) |

Same-night ratio **1.17×** (drift hits both legs equally; in the 91.93-era
reference conditions the campaign's ratio was ~1.24×: ~90.3 vs 112-114).
Dory prover preprocessing that run: 31.0 s (not counted in prove).

## State (2026-07-25, campaign closed — queue exhausted, goal not met)

333 s campaign start → **~90 s** in 91.93-reference conditions (best
same-night A/B'd reading 90.27; the evening's monotone ambient drift
+~1 s/h put late readings at 94). RSS **94.9 → 87.3 GB**. Goal ≤65 s NOT
met: it had banked on Q1 (−8-18), Q3 (−3-9) and Q4 (−4-6), all three of
which died under measurement (kernel already at machine rate; premise
false; candidates circular/sub-bar — see the queue entries).
MACHINE-DRIFT NOTE: the 91.93 record does not reproduce late-night — the
same-code baseline re-ran at 96.31 (B1); accept decisions used same-night
A/B pairs + traced spans, never the stale record.
Landed this campaign: A1 kernel chunking; J2+J3 decode dedupe; P1
rank-aware catalogs (root n_a 7→6); A5 fused multi-poly sweep (commit
96→~44 s); **Q2A** RaIndices sharing into 6b inits (jolt 9e957dbfc, span
−2.3 s, A/B 96.31→90.27); **Q6** block-cache drop post-commit (akita
eee5ad1f + jolt 3706e1290, RSS −7.6 GB, perf-neutral within gate).
Attribution (95.2 s traced, drift-era): commit 44.1 | fold/opening ~15.6
| stage1 8.5 | stage6b 6.0 | stages 3/4/5/6a/7 ~17.1.
Open threads, ranked: Q6 remainder to ≤60 GB (A-free + stage-3/4
transients); Q3 note awaiting user verdict (recommendation: close; msb
worth ~net −1 s only); Q1b resurrection diff (−1.2 s, sub-bar); Q2B
stage-7 hamming decode reuse (RSS-gated); Q5 uniskip SIMD (~−1-1.5 s,
shared-code risk).

Facts that bound the work: PIOP already runs over the 128-bit field
(`AkitaPackedScheme::Field = AkitaFp128`, in-field challenges); commit work
= P·T·n_a·D with n_a·D security-pinned; K=256 geometry-optimal at 2^26.

Dead-end ledger (measured; do not revisit): A4 lazy u128 accumulators (NEON
wide wins 1.9×); A6 fused in-register widen (100→131 s); merge-tile tuning
beyond (64 blocks, 32 cols) — bench matrix flat; L1 tiles under block
splitting (252 s; fixed by cap-triggered self-reduction, akita 015669b9);
Q1 P-core-affine pool / thread-count tuning (16t already optimal: 1.15 s vs
12t 1.31 s at bench shape; e2e-vs-bench 61 vs 50 ns is sustained-clock);
Q1b merge-sweep partition rebalance (real but −1.2 s attributable < bar —
idle was pipeline-absorbed; see Q1b entry for the resurrection diff);
Q3 committed-column virtualization (see specs/committed-column-
virtualization.md — RamRa sparse ⇒ 0.24 s, BytecodeRa circular, msb net
−1 s, K reshapes = Q7 wall).

## Harness (exact commands)

- Headline (NO tracing), from /Users/mgeorghiades/jolt:
  `PERF_LOG_T=26 /usr/bin/time -l cargo nextest run --release -p
  jolt-prover-legacy --features akita -E 'test(sha2_chain_akita_perf)'
  --run-ignored all --no-capture --cargo-quiet`
- Iteration scale: same with `PERF_LOG_T=22` (seconds-scale signal).
- Attribution: add `PERF_TRACE=1` → writes
  `benchmark-runs/perfetto_traces/sha2-2exp{N}-akita.json`; analyze with
  `python3 scripts/perf/analyze_trace.py <trace.json>` (span ranking) and
  `python3 scripts/perf/occupancy.py <trace.json> <span-name>` (busy-thread
  histogram over a span window).
- Kernel bench (akita repo): `cargo test -p akita-prover --release
  --features parallel --lib merge_sweep_bench -- --ignored --nocapture`.
- Gates (all must pass before a change lands):
  1. `cargo nextest run -p jolt-prover-legacy muldiv --cargo-quiet
     --features host` and `--features host,zk`
  2. `cd ~/akita && cargo test -p akita-prover --release --features
     parallel --lib`
  3. `cargo nextest run -p jolt-akita --cargo-quiet` (mandatory when
     catalogs/planner change)
  4. clippy both modes, checking `pipestatus[1]` not `$?`.

## Measurement discipline (violations produced four false readings this campaign)

- Before EVERY benchmark: `ps aux | grep jolt_prover_legacy-` must be empty;
  never chain a benchmark behind sleeps in a wrapper that can overlap a
  relaunch.
- Cooldown ≥120 s after any build or prior run; prebuild with `--no-run`
  first so the measurement is run-only.
- A single e2e number never accepts or rejects a change: use the 2^22
  iteration scale plus kernel benches to iterate, and confirm at 2^26 with
  a cooled run; if a 2^26 reading contradicts the model by >15%, re-run once
  cooled before concluding. Traced runs arbitrate disputes (trace overhead
  is now ≈2%).
- Accept a change iff: median improvement ≥1.5 s at 2^26 (or ≥0.15 s at
  2^22 for stage-local items with a clear traced-span delta) AND gates
  green AND RSS not regressed >2 GB. Otherwise `git revert`, append to the
  dead-end ledger with the numbers, move on.
- Transcript identity required except catalog regenerations (which then
  require gate 3 and a fresh drift/coverage pass).

## Work queue (descending payoff; per-item protocol inline)

### Q1 [CLOSED 2026-07-24 — kernel already at machine rate; no software-recoverable gap]

Probe (`merge_sweep_thread_probe`, akita d17635fc, bench shape = e2e shape):
16t 1.15 s / 50.4 ns×t · 12t 1.31 s / 43.1 · 10t 1.54 s / 42.1 ·
8t 1.82 s / 39.8. Verdicts: (a) REFUTED as a fix — E-cores net-positive
(16t beats 12t by 14%); implied split P≈43 ns, E≈102 ns, and blended 16t
throughput already equals perfect balance (macOS migration load-balances
the static partition; finer chunking buys nothing). (b) confirmed as the
in-situ residual: e2e kernel spans total 717 CPU-s / 11.7 G accums =
61.3 ns vs 50.4 fresh-bench — sustained-clock/thermal, not software.
(c) moot — 1-coeff-entry bench already shows 50.4. The 26-28 ns model was
optimistic: true P-core burst rate ≈40 ns/accum (8t).
AMENDED by the Q3 audit (2026-07-24): the "no serial gaps" occupancy read
was too coarse. Fine-grained per-thread analysis of the merge window shows
the static block partition (column_sweep.rs:404) hands one thread the
nearly-empty RamRa range (RamRa ≈7-8% dense) — 15 busy threads for 74.7%
of the window, 3.26 s of wall lost. That imbalance IS most of the
61-vs-50 ns e2e-vs-bench gap. Recoverable with no protocol change → Q1b.

### Q2 [DONE 2026-07-24 — −2.3 s span-attributed; A/B 96.31→90.27 s] (jolt 9e957dbfc)

Init-vs-rounds split measured (2^26 trace): Bytecode cycle init 2.21 s
(get_pc linear scan ×bytecode_d passes) | InstructionRa init 0.38 s |
RamRaVirtual init 0.05 s. Landed: `BooleanityCycleInput.ra_indices` →
`Arc<Vec<RaIndices>>`, all three 6b initializers gather their chunk
columns from it via `gather_index_columns` (L2-blocked single pass — a
block of RaIndices stays cache-resident while all d columns sweep it; the
naive d-interleaved write loop was 10× slower). Traced 2^26: inits
2.64 → 0.16 s (+0.16 gathers), stage 6b inclusive 8.03 → 6.02 s. RSS and
verify neutral; transcript-identical. NOTE: the caveat in the audit was
wrong in a good way — LookupsRa chunks ARE committed-width, so chunks are
shared directly (no base-index sharing needed).
REMAINING (deferred, not counted done-work): stage-7
HammingWeightClaimReduction::initialize re-runs the full decode pass via
compute_all_G (1.62 s) at a different r_cycle. Reusing ra_indices there
extends the 3.6 GB vec's lifetime past 6b → RSS-gated; decide after Q6
attribution, or via a flat PC array (u32×T = 256 MB) at witness time that
would also cheapen witness/stage-1/6a decode sites. Also 6b bytecode-init
now exposes ~0.04 s floor; nothing further here.

### Q3 [PROTOCOL] [NOTE WRITTEN 2026-07-24 — awaiting approval; recommendation: CLOSE]

Design note: `specs/committed-column-virtualization.md`. Measured verdicts
(three claim-graph audits + commit-cost audit): the premise's 1.55 s/column
uniformity is wrong — RamRa is ~7-8% dense (worth 0.24 s, not 3 s);
BytecodeRa virtualization is soundness-circular (the committed one-hot form
IS the PC range-binding via the RAF/Int identity); the msb elimination nets
only ~−1 s after its replacement sumcheck; per-family K reshapes hit the Q7
geometry wall (ring footprint ∝ K). Real finding: a 3.26 s kernel
load-imbalance artifact → Q1b [ENG], no approval needed. Do not implement
Q3 unless the goal is unmet after Q1b/Q4/Q5/Q6 (then msb, net ~−1 s).

### Q1b [CLOSED 2026-07-24 — real but sub-bar: −1.2 s attributable; reverted per discipline]

Two implementations measured (both byte-identical, suite-green): (v1)
global 64-block tile grid with rayon stealing — dense tiles are ~10.7
CPU-s, so ±half-tile quantization re-created seconds of tail (per-thread
merge busy 33.0/34.2/42.3 min/med/max); (v2) entry-weighted contiguous
ranges ×2-per-worker, tile-capped (sub-tile ranges cost +18% A-widening at
the uniform bench shape) — balance fixed (min 6.7→38.0 CPU-s, mergeCPU
642→623), and the cooled uniform kernel bench improved 1.14→1.07 s. BUT
commit wall moved only 44.09→42.92 traced (−1.17 s): most of the "3.26 s
idle" was absorbed by pipeline overlap, and the residual max-thread 42.3
vs med 38.9 is scheduler/E-core-blended. e2e pairs contradictory within
±2 s ambient (N1 90.27 → N2 91.94 untraced; traced pair −4.1 with −1.9
ambient drift in untouched spans). Below both accept clauses → reverted.
Resurrection candidate if the campaign ends <1.5 s short: the v2 diff is
fully specified above (weighted bounds + tile-cap + 2x-worker ranges).

### Q4 [CLOSED 2026-07-24 — premise false: the spans are nested, not sequential]

Trace nesting (B/E parent chains, 2^26): `onehot_accumulate` (5.85 s) runs
INSIDE `fold_grind_sample > OneHotPoly::decompose_fold_batched` — the
grind's single accepted probe (nonce 0, plain preset) IS the fold walk
(`fold_probe_witness_kernel` → `build_point_decompose_fold_witness` →
`OneHotPoly::decompose_fold_batched` → `onehot_accumulate`,
ring_relation.rs:165→209, fold_grind.rs:383). The two span totals in the
analyzer ranking (7.5 s x8 grind / 5.9 s x1 accumulate) are parent and
child: root-level grind ≈ 6.4 s (mostly the accumulate), the 7 recursive
levels ≈ 1.1 s combined. Nothing to fuse; no rerolls to hide at this
preset. Residual fold-side floor: the accumulate itself (~51 ns/entry over
~1.8 G entries — position-sorted streaming, already parallel) — no bounded
idea ≥ the accept bar found.

### Q5 [CLOSED 2026-07-24 — audited smalls are phantoms/sub-bar; uniskip already tuned]

Span-sized against the 2^26 trace: `ShiftSumcheckProver::compute_message`
= **0.00 s** across 26 rounds (the "serial loop" domain is tiny — the item
was a phantom; shift's real cost is initialize 1.35 + ingest 1.23);
bytecode address-phase products already parallel over a K/2 = 2^12 domain
(≤0.2 s); the bind-frees are the only real small, bounded by
`MultilinearPolynomial::bind_parallel` excl 2.21 s total ⇒ −0.3 to −0.8 s
expected — implemented (`drop_in_background_thread` in
CompactPolynomial::bind/bind_parallel + SmallScalar: Send + 'static) and
REVERTED as sub-bar; diff trivially reproducible. Stage1's remaining 8 s:
uniskip extended-evals loop verified already deep-tuned (decode-carry,
S192 fmadd accumulation, one montgomery reduce per x_out — outer.rs:
200-272); the "S64/S128 SIMD" residual means vectorizing the R1CSEval
integer products — shared Spartan surface, muldiv-gated, upside ~1-1.5 s
of the 3.3 s span, not reliably ≥ bar. No Q5 component clears the accept
criteria alone or bundled (~-1 to -2 s combined best case, high shared-
code risk on the only large piece).

### Q6 [LANDED first fix 2026-07-24 — RSS 94.9→87.3 GB; target ≤60 GB not reached, remainder scoped]

Attribution done (0.5 s RSS sampler aligned to trace phases; the packed
path has NO allocative instrumentation — pivoted): pre-prove standing
52.7 GB (trace 4.1 → witness/setup/blocks build to 52.7 by t=15 s);
commit window flat 67-69; **peak 87.6 GB sampled in stages 3-4**
(stage3 ramps 70.9→81.6, stage4 spikes); 6b releases to 69.6; the fold
DECLINES 69.6→60.5. time -l peak 94.8 (sampler misses the spike top).
LANDED: drop the one-hot block cache (~15 GB) right after the commit
absorb — `OneHotPoly::clear_block_cache` (akita eee5ad1f) +
`AkitaProverHint::drop_one_hot_block_caches` + packed.rs call (jolt
3706e1290); blocks_for transparently rebuilds inside the stage-8 fold
(~1 s). Sampled peak 87.6→78.2, time -l 94.8→87.3/87.2 (two cooled
runs); perf 92.96→94.39/94.06 vs adjacent pre-fix run = +1.1-1.4 s,
inside the ±2% gate (ambient drifted ~+1 s/h all evening). REMAINING to
≤60 GB (scoped, not implemented): (a) expanded A 12.9 GB — freed-after-
sweep needs a setup-lifecycle API (`CpuPreparedSetup.expanded` is an
Arc shared for potential re-proves; regen costs ~2.1 s
derive_public_matrix_flat); (b) the stage-3/4 transient (+10-19 GB over
standing, jolt-side val/LT/wa materializations) is now the peak owner;
(c) pre-prove standing 52.7 GB (witness indices + setup + trace) bounds
everything below ~53 without deeper surgery.

### Q7 [DEFERRED — do not work at 2^26]

K=2^16 (favorable only ≥2^28: P·n_a 153 vs 203 at 2^30; needs nv=42
catalogs + seed-streamed A); rank-hold candidate dumps at nv≥35 (P3);
anything in the dead-end ledger.

## Loop protocol

1. Take the highest unfinished queue item. Mark it in this file
   (`[IN PROGRESS <date>]`).
2. Run its First step / probe. If the probe kills the hypothesis, record the
   numbers under the item, mark `[CLOSED — <one-line verdict>]`, next item.
3. Implement smallest-diff; equality/parity tests where the item touches
   kernels (mirror `merge_sweep_matches_bucketed_core_across_polys`).
4. Gates → iteration-scale measure → cooled 2^26 confirm per discipline.
5. Accept ⇒ commit (`perf(scope): what — measured X→Y s`) in the owning
   repo, update the State/attribution numbers here, mark item `[DONE — Δ]`.
   Reject ⇒ revert, ledger entry, next.
6. After every landed item: if 2^26 prove ≤ 65 s, run the closing sequence:
   back-to-back cooled akita AND dory 2^26 runs (dory:
   `-E 'test(sha2_chain_dory_perf)'`, no `--features akita`), record the
   final table here, and stop.

Never push or force-push. Commits stay local on both repos. If context runs
long, this file plus the memory entry `akita-perf-branch-state` are the
resume points — keep both current as part of step 5.

## Appendix: algorithm/arithmetic audit (2026-07-24, agent-verified file:line)

Verdict: per-element inner arithmetic uniformly tuned — every T-sized fold
accumulates unreduced (`Folded128Product` via `mul_to_product_accum`,
`par_fold_out_in_unreduced`) and reduces once per lane/round; no dense
materialization of one-hot data (RaPolynomial/SharedRaPolynomials index
form); AkitaFp128 mul/add/sub are hand asm (schoolbook 2x2 + Solinas fold,
~6 mul-ops; flag-fused canonicalize). Confirmed-tuned compute_message list:
booleanity.rs:894/293, read_raf_checking.rs:797, hamming_booleanity.rs:164,
ram/ra_virtual.rs:257, instruction_lookups/ra_virtual.rs:273,
spartan/instruction_input.rs:314, registers/val_evaluation.rs:207,
instruction_lookups/read_raf_checking.rs:866. Structural waste found only
in initialize() index builders (now Q2) and the Q5 smalls. Stage map:
stage3 = Shift + InstructionInput + RegistersClaimReduction; stage5 =
InstructionReadRaf + RamRaClaimReduction + ValEvaluation; 6a = bytecode
address read-raf + booleanity address; 6b = bytecode cycle read-raf +
booleanity cycle + hamming + RamRaVirtual + LookupsRa + claim reductions.
Not measured: init-vs-rounds split per 6b sub-prover (bounds Q2's payoff).
