# GOAL-MODE HARNESS: Akita packed prover peak RSS at 2^26

Execute autonomously in goal mode. Work the queue top-down, one item at a
time. All items are ENG (no protocol changes anywhere in this spec — every
fix is lifetime, laziness, deduplication, or streaming of data the prover
already computes).

## Goal

On `perf/optimize-akita-prover` (jolt) + `perf/onehot-commit-sweep`
(`~/akita`, path deps) — both freshly synced onto their origin/main
(2026-07-26: jolt rebased onto main post-#1676/#1677/#1701; akita merged
through #329 with the schedules-crate adaptation, jolt b61f3bc1a).

- **PRIMARY: peak RSS ≤ 48 GiB (51.5 GB `time -l`)** at 2^26 sha2-chain,
  from the post-Q6 87.3 GB.
- STRETCH: ≤ 35 GB — below dory's 36.6.
- GUARDRAILS (hard): prove time within ±2% of the post-sync baseline on
  adjacent cooled pairs (a >2% regression is a revert, whatever the RSS
  win); all gates green; verify/proof size unchanged; transcript identity
  (nothing here may change what is proven or absorbed); security/norm
  surfaces untouched.

Done when: primary met AND every item landed or dead-end-ledgered with a
measurement — or the queue is exhausted.

## The measured model (2026-07-25, all numbers verified against sampled
## timelines; conversion at 2^26: 1 B/cycle = 64 MiB, GiB = B/c ÷ 16)

RSS(T) ≈ Fixed + B/c(peak phase)·T. The fixed term is three residencies of
the ONE public matrix A (n_a=6 × ppb=2^21 ring-element columns):

| slab | size | built | last real use |
|---|---|---|---|
| A field form (16 B/coeff) | 12.0 GiB | setup | commit sweep end (t≈62 s) |
| A negacyclic CRT-NTT (5 primes × i32) | 15.0 GiB | setup (`prepare_setup` envelope slot, akita cpu.rs `register_setup_contract_ntt_slot_on_prepared`) | stage-8 fold kernels (t≈93-108 s) |
| A cyclic CRT-NTT (`NttCacheMode::BothTransforms`, ntt_cache.rs) | 15.0 GiB | setup | **unverified — audit is M1** |

Per-cycle terms at the (post-Q6) stage-3/4 peak window: trace `Vec<Cycle>`
96 (struct is 96 B, size-asserted in tracer) + witness one-hot index
columns 58 + fused-inc 34 + stage field transients ~180 + index-form
transients 94 (`ra_indices` 54 + 6b gathered columns 40) ≈ **460 B/c**.
Model vs measured: pre-prove standing 52.7 GB = trace 6.4 + A-field 12.9 +
A-NTT 32.2 + misc (sampler: 52.7 exact); stage-4 peak 43 GiB + 42.5 ≈ 85.5
vs 87.6 sampled / 94.8 `time -l` pre-Q6.

Dory for reference: ~2 GiB fixed (SRS) + ~520 B/c → 36.6 GB. Akita's
per-cycle slope is already BELOW dory's; the entire deficit is the 43 GiB
of redundant A copies plus the block cache. Duplication map: the per-cycle
chunk values exist in FOUR layouts during 6b (witness columns 58, ra_indices
54, gathered columns 40, fused-inc one-hot 18) — one SoA copy (the witness
columns, already held by the commit hint as `Vec<Option<u8>>` per column,
exactly the `RaPolynomial::new` input format) can serve every consumer.

Landed already: Q6 block-cache drop post-commit (blocks rebuild inside the
stage-8 fold from retained indices; akita `clear_block_cache` +
`drop_one_hot_block_caches`) — 94.9→87.3 GB.

## Harness

- Headline: `PERF_LOG_T=26 /usr/bin/time -l cargo nextest run --release -p
  jolt-prover-legacy --features akita -E 'test(sha2_chain_akita_perf)'
  --run-ignored all --no-capture --cargo-quiet`.
- RSS timeline (phase attribution): wrap the run with the 0.5 s sampler
  (`ps -o rss=` on the test pid → scratchpad timeline), align against a
  `PERF_TRACE=1` trace's phase windows (the two-file alignment script from
  the 2026-07-25 analysis).
- Iteration scale: `PERF_LOG_T=24` — RSS slabs scale linearly in T except
  the fixed A forms, so 2^24 separates fixed-vs-scaling cheaply.
- Gates: muldiv host + host,zk; akita suite (release, parallel); packed
  e2e (`muldiv_e2e_akita | advice_e2e_akita`); jolt-akita suite when
  catalogs/planner/config surfaces move; clippy host + host,zk; fmt.
- Measurement discipline as in the 2^26 perf spec: exclusivity check,
  prebuild + ≥120 s cooldown, adjacent cooled pairs for the perf-neutrality
  gate (ambient drifts ~±1 s/h — never compare against stale absolutes).
- QUIET-MACHINE GATE (added 2026-07-26 after contaminated M0): the prover
  grep is NOT enough — concurrent Claude/agent sessions build sibling
  worktrees (rustc at 600%+ invalidated a full M0 pair and made the kernel
  bench read 2.7× slow). Before every measured run: no foreign
  `rustc|cargo|nextest` processes AND 1-min load average < 4, polled until
  stable for 60 s. A reading taken while the gate was violated is void.
- Accept an item iff: `time -l` peak improves ≥3 GB at 2^26 with the
  sampler agreeing on the window it came from AND prove within ±2% on the
  adjacent pair AND gates green. Else revert + ledger.

## Work queue (descending expected GB, dependencies noted)

### M0 [ENG] Post-sync re-baseline — prerequisite, not an optimization

The upstream merge (#320/#323/#328/#329 + regenerated catalogs) may have
moved both time and RSS; the hot 2^22 smoke read +1 s over pre-sync cooled
numbers. One cooled 2^26 headline with sampler + trace. Record prove, peak,
phase profile as THE baseline for every accept decision below. Also log
`CpuPreparedSetup::shared_ntt_cache_bytes()` once (accessor exists,
cpu.rs:146) to confirm the 32.2 GB NTT figure on the synced stack.

VERDICT (2026-07-26, quiet-gated traced run M0A): **prove 118.78 s, peak
87.71 GB `time -l` / 80.36 GiB sampled** — peak inside prove_stage4 (RSS
model confirmed on the synced stack). Phase profile (sampler GiB): commit
plateau 68.5 (18.9→78.6 s), stage1 72.6, stage3/4 peak 80.4, fold window
73.5 (112.9→134.6 s). First pair (118.25/123.03, 2026-07-26 pre-dawn) was
void — foreign session compiling; quiet-gate now enforced. The post-sync
prove regression is REAL (~118.8 traced vs ~96 pre-sync untraced record;
traced overhead ≈5 s) — cause not yet isolated (#328 larger-d fold cost /
catalog schedule drift candidates); tracked separately, does not gate RSS
items (they gate on the ±2% adjacent pair). The NTT slot build measures
only **1.9 s** at this envelope (prepare_ntt_cache span), not 8-10 s.

### M1 [ENG] Cyclic-transform usage audit — decides ±15 GB of the plan

Trace which stage-8 kernels pull `neg` vs `cyc` from the prepared NTT cache
(akita: consumers of `PreparedNttCache` / `CyclotomicCrtNtt` in the fold /
commit_w_level / relation paths). The ring is negacyclic (X^64+1); if `cyc`
serves no path reachable from the packed prove at these shapes, M3 is a
mode change; if it is used, M3 becomes per-transform laziness instead.
VERDICT (2026-07-26, code audit): `cyc` IS used — the ring-switch relation
path consumes it (`CyclicRowsComputeBackend::cyclic_digit_rows` →
`mat_vec_mul_ntt_single_i8_cyclic`, cpu.rs:648-670, and
`fused_split_eq_quotients_prover_bounds`, kernels/linear/fused_quotients.rs
— both via `with_shared_ntt`), interleaved per fold level with `neg` uses
(`digit_rows` → `mat_vec_mul_ntt_single_i8`). So M3-as-deletion is dead;
both transforms are stage-8-only though, so M2 still moves all 30 GiB out
of the stage windows. The refined M3 question: the envelope slot is sized
to the FULL root matrix (12.58M rings) while stage-8 products may only
touch per-level prefixes — the width fields added to the two kernels'
spans (single_cyclic.rs) capture the actual max prefix in the M0 trace;
if max-used ≪ envelope, size the lazy slot from the schedule instead.

### M2 [ENG] Lazy NTT envelope slot — expect −7 to −15 GB peak

Defer the envelope NTT slot build from `prepare_setup` (the
`register_setup_contract_ntt_slot_on_prepared` call) to first use — the
lazy path (`ensure_ntt_slot`) already exists and is single-flight. Moves
30 GiB (or 15 post-M3) out from under the stage-3/4 peak; the fold window
becomes the new peak, so the realized peak delta depends on M3/M5. Build
cost (~2-4 s NTT conversion) moves inside the fold window — it is real
prove time, so the ±2% gate arbitrates; if it breaches, overlap the build
with stage 7 (spawn on the backend pool) before giving up.
Watch: any pre-fold path that touches the slot (M1's audit lists consumers)
turns this into a no-op — verify with the sampler that the stage-window
plateau actually drops.

INTERIM (2026-07-26, M2B adjacent to M0A): lazy-registration alone is NOT
enough — the one-hot commit's own terminal root product (inside
`OneHotPoly::commit_inner_group`, single `mat_vec_mul_ntt_single_i8` call
~0.1 s at commit end) touches the slot, so it builds at t≈64.5 (1.86 s)
and the stage-3/4 peak is untouched: prove 108.22 s (**−10.6 s!**), peak
87.95 GB (+0.2). The commit itself ran 47 of its 49 s at a ~38-41 GiB
plateau (−30 vs baseline) and got 10.6 s faster — consistent with macOS
compressor pressure on the idle slab being lifted, and with M2 skipping
4 of 6 eager `prepare_ntt_cache` builds (unused setups never build).
COMPLETION (M2C, this commit): Q6's lifecycle applied to the slot —
`CpuPreparedSetup::drop_built_ntt_slots()` (rebuild-on-next-use is safe
post-M2) called from packed.rs right after the commit absorb via
`AkitaProverSetup::drop_ntt_slots()`; the fold's first use rebuilds both
transforms in ~1.9 s, paid from M2B's −10.6 s margin.

VERDICT (2026-07-26, M2C adjacent to M0A, quiet-gated): **ACCEPT**.
Peak **87.71 → 81.27 GB `time -l` (−6.4)** / 80.4 → 74.6 GiB sampled;
prove **118.78 → 99.42 s (−16.3%)** — traced 99.4 ≈ the pre-sync untraced
record, i.e. M2 recovered most of the post-sync regression (the eager
slab was throttling the commit via compressor pressure, not just RSS).
Stage windows collapsed as designed: stage1 72.6→42.7, stage3/4 peak
80.4→47.7, stage7 56.0→32.3, commit 59.7→43.6 s. New peak = fold window
(74.6 sampled: blocks rebuild + NTT rebuild + fold transients stack
there; fold 21.7→22.2 s absorbs the rebuild). Next targets follow from
the new profile: fold window (M5/M7/M8) is the peak; stage plateau ~42
still carries A-field 12 GiB (M4) + index layouts (M5). NOTE for M4:
`build_ntt_slot_for_key` reads `expanded.shared_matrix()` — the fold's
NTT rebuild now needs A's field form at t≈91, so M4 must either free
after that rebuild or re-derive from seed (2.1 s) inside the fold.

### M3 [ENG] Drop the unused NTT transform — expect −15 GB (fold window), conditional on M1

If M1 says `cyc` is unreachable: build the envelope slot
`NegacyclicOnly`-equivalent (mode exists for exact-negacyclic; a
both-transforms-minus-cyc variant is a small enum addition in
akita-types/ntt_cache.rs). If `cyc` IS used: split the slot so each
transform materializes on first use and the unused one never does.

MEASURED (2026-07-26, M3D adjacent to M2C): extent-aware
`with_shared_ntt(extent, f)` implemented — consumers declare rows×width,
slot builds at the rounded max request, smallest built cover reused.
Commit-terminal build shrank 30→5 GiB (commit plateau 68.9→38.7 GiB
sampled, commit 43.6→40.5 s; prove 99.42→97.13 s) **but the fold rebuilt
the FULL 12.58M-ring envelope** — the root-level ring-switch relation's
extent is n_d·e_hat.len() = 6·2^21 = the whole matrix (protocol geometry,
frozen). Peak 81.27→81.15 GB = below the 3 GB bar standalone, so M3 is
NOT committed alone — it rides in the M7a composite below. Deletion
variant is dead for good: the fused relation kernel consumes cyc at
every level (fused_quotients.rs:233).

### M7a [ENG] Stream the root relation's transforms — expect −25 GB (fold window)

The root relation reads each A element exactly once per prove, so its
matrix-scale NTT cache is pure standing memory. Streamed one-shot fused
kernel (`fused_split_eq_quotients_one_shot_streamed`): per column tile,
transform needed elements from A's field form in-loop
(`from_ring_cyclic_with_params` — new cyc-only constructor — for D/B,
`from_ring_pair_with_params` for the A quotient; identical values to the
cache fill, so results are bit-identical — equality test in cpu.rs).
Dispatch: relation/quotient extents > 2^21 rings stream from the field
view; smaller extents (deeper levels, digit rows) share the ≤5 GiB
cached slot. Non-one-shot shapes fall back to the cached path. Fold
window should drop ~30 GiB cache → ~5 GiB slot; streaming cost ≈ the
deleted 1.9 s envelope build, net time ~0. NOTE: this pins A's FIELD
form as a fold-window dependency — M4 must free it only after the last
relation streams (or re-derive from seed).

VERDICT (2026-07-26, M7H adjacent-night to M0A, quiet-gated): **ACCEPT**
(akita 268a2e47, composite with M3's extent machinery). **Peak 87.71 →
55.10 GB `time -l` (−32.6)** / 80.4 → 51.3 GiB sampled; **prove 118.78 →
95.61 s (−19.5%)**, fold duration unchanged (21.7 s — streaming cost
fully absorbed by the deleted builds). It took FOUR eliminations to get
the envelope out of the fold: (1) M2 lazy registration, (2) commit_w /
commit_terminal_w base-dim warm skip, (3) prove.rs fold-entry role +
terminal warm skip (`ensure_fold_level_role_ntt`), (4) the quotient-only
z-relation (wl=0 tl=0 zl=2^21 z_cap=240) needs CRT CHUNKING — one-shot
streaming bails — so a chunked streamed z-quotient kernel mirrors the
cached bracketing. `AKITA_NTT_BUILD_BACKTRACE=1` dumps per-build stacks
(this is how each was found; the harness tracing::info lines land only
in PERF_TRACE chrome traces, not stdout). New profile (sampled): commit
43.7 / stages ≤47.3 / fold 47.9 base + 3.4 transient spike (root-level
stage2_sumcheck, t≈107) = 51.3 peak. Remaining slabs everywhere:
A-field 12, blocks 13.6 (fold), slots ≤5.6 (fold), trace 6.4, indices.
M7b (next, surgical): drop built slots after the root level's products —
deeper levels rebuild ≤0.6 GiB in ~0.04 s — should take the fold under
48 sampled = PRIMARY. M4 (seed-streaming: `entry_rng(idx)` is
per-element seekable, so random access is native) then removes A-field
from all post-commit windows → commit window becomes the peak (~47.5
time-l) and M8 opens the path to STRETCH.

### M4 [ENG] Free A's field form after the commit sweep — expect −12 GB (stage windows)

Last field-form use is the root commit sweep (t≈62 s); the object lives in
`CpuPreparedSetup.expanded: Arc<AkitaExpandedSetup>`, shared for potential
re-proves. Needs a lifecycle API: either (a) an explicit
`release_expanded_matrix()` on the prepared setup called by packed.rs after
the commit absorb, with lazy re-derivation from seed on next use
(`derive_public_matrix_flat` ≈ 2.1 s — only re-proves pay), or (b) if the
NTT conversion (M2) is the sole post-commit reader, convert-then-free
inside the same call. Mind the Arc: dropping must actually release (no
second strong ref held by the setup contract or NTT keying).

DESIGN UPDATE (2026-07-26, post-M7): **M4 is now THE decisive item** —
after M2/M3/M7 the three windows sit within 4 GiB of each other (commit
43.7 / stage-3-4 spike 51.0 / fold 49.3 sampled) and A-field 12 GiB is
in ALL post-commit windows; nothing else reaches primary (M6 fully fixed
still leaves fold ~49.3 → ~53 `time -l`). Post-M7 constraints:
- The STREAMED relation kernels read `expanded.shared_matrix` flat at
  FULL width per fold level (once each) — so post-commit consumers are
  (a) slot rebuilds at prefix ≤2^21 and (b) streamed relation full-width
  reads. (b) must switch to SEED-derivation: `derive_public_matrix_flat`
  uses `LabeledMatrixXof::entry_rng(idx)` — per-ring-element seekable,
  so the streamed kernels derive element `i·width+j` on the fly (XOF +
  from_ring transform per element; the relation already pays the
  transform, XOF adds F::random ×64 per element — measure, margin is
  ~25 s vs the ±2% gate).
- Then the release: `shared_matrix` is immutably shared
  (`Arc<AkitaExpandedSetup>`, ~70 refs across 7 crates incl. verifier
  stage3). The clean route is `RwLock<Option<FlatMatrix>>` (or
  OnceCell-style regenerate-on-read) INSIDE AkitaExpandedSetup with a
  guard-returning accessor + seed re-derivation for re-proves; the
  jolt-side "swap prepared setup for a truncated-matrix twin" variant
  dies on envelope/role-dim validation (the seed pins max_setup_len).
  Sized: one focused file (akita-types setup.rs, 28 refs) + guard
  plumbing at ~15 consumer sites. Do it FIRST next session, fresh.
- Prediction once landed: stage windows −12, fold −12 → peak = commit
  window 43.7 sampled ≈ ~47.5 `time -l` → **PRIMARY met with margin**;
  M8 (commit tile blocks, −10) then chases STRETCH 35.

M6 allocative decomposition (2^24 flamegraphs, ×4 for 2^26):
RegistersReadWriteCheckingProver 6.3 GiB total at 2^26 — entries vec
(`RegistersCycleMajorEntry<Fp128, LookupTableIndex>`, gamma-folded F
values) 5.2 GiB = 82.5 B/cycle, RdInc regen (CompactPolynomial i128)
1.0 GiB, RamValCheck 1.0 GiB; spike +11 over base = these + ~2.7 build
transients. Fix shape: compact entry storage (u64 register values,
lift-on-first-bind like CompactPolynomial) — halves the matrix; only
worth doing AFTER M4 (window then sits at ~39 anyway).

### M5 [ENG] Index-layout dedup — expect −6 to −7 GB (stage windows), perf-neutral or better

One SoA copy of the per-cycle chunk values serves all consumers:
- 6b RA initializers take `Arc` clones of the hint's witness columns
  directly (same values by construction — packed.rs `assemble_one_hot_trace`
  computes them with the same chunk functions; the Q2 gather becomes a
  no-op for the RA families).
- booleanity/`SharedRaPolynomials` reads the same columns instead of the
  AoS `ra_indices` (drop `compute_all_G_and_ra_indices`'s index output or
  keep it as views).
- the fused-inc one-hot columns for booleanity/hamming Arc-share the
  witness inc columns instead of rebuilding.
Ordering constraint: the hint owns the columns from commit time — the Q6
cache-drop only cleared blocks, the index columns stay; verify the packed
flow keeps the hint alive through 6b (it does — stage 8 opens from it).
Transcript-identical; muldiv + packed e2e + a 2^22 traced span check.

### M6 [ENG] Stage-3/4 transient sequencing — expect −3 to −5 GB (peak window)

The sampled stage3 ramp (+11 GB) persists into stage4's spike: stage-3
materializations (Registers RW ra/val polys, Shift/InstructionInput) are
not fully released before RamValCheck materializes wa/LT/inc/val. Audit
drop points between the stage-3 batch end and stage-4 initialize; add
explicit drops/`drop_in_background_thread` for bound-out polys. Sampler
arbitrates (the window max is the metric).

STATUS (2026-07-26, post-M7b): with the fold at 49.3, this window IS the
global peak (51.0-51.1 sampled across M7I/M7J, ~55.0-55.8 `time -l`).
FALSIFIED sub-hypothesis: replacing stage 3's
`drop_in_background_thread(instances)` with a synchronous drop changed
NOTHING (M7J: 51.00 same spike; reverted) — no Drop impl re-defers, so
the spike is INTERNAL to stage-4's initializes over a ~40 base:
`RegistersReadWriteCheckingProver::initialize` (RdInc witness regen +
`ReadWriteMatrixCycleMajor::new` — per-cycle F entries) +
`RamValCheckSumcheckProver::initialize` (prover.rs:1141-1152), settling
at 45.0 and freeing to 34.5 at stage-5 entry. Next: allocative
decomposition at 2^24 (in flight) to size matrix vs val-check vs
witness-regen temporaries; the fix is inside those initializers
(lazy/streamed entry values or staged build-drop), not at the stage
boundary. M7b landed as 09d01635 (fold 51.3→49.3 sampled, peak-neutral
standalone — enables this item to move the global peak).

### M7 [STRETCH] Seed-streamed A-NTT — the ≤35 GB unlock

Replace the resident NTT envelope with chunked generate→convert→multiply→
discard inside the fold kernels (the matrix is seed-derived; ~2 GiB working
set). Real kernel engineering with a perf-regression risk (conversion
re-paid per use — M1's audit counts uses). Also the prerequisite Q7 names
for K=2^16 at 2^28, so it pays twice. Only attempt after M2-M6 land and
the fold window is confirmed as the remaining peak.

### M8 [STRETCH] Lazy per-tile block build in the commit sweep — expect −10 GB (commit window)

The merge sweep touches blocks tile-at-a-time; build each 64-block tile's
sorted entry lists on demand from the index columns and discard after the
tile pass (~4 GiB in flight instead of 13.6 resident). Only matters once
M2/M4 make the commit window the peak. The FOLD still needs all blocks
concurrently (position-major accumulate) — out of scope here (fold-layout
redesign is the ledgered non-goal).

### Deferred / non-goals

Trace compaction to projection arrays (~−3 GB, touches every stage);
fold-layout redesign to kill the block cache in the fold window (~−13 GB
below the M7 floor); anything protocol-visible. The ~28-30 GiB bedrock
(trace 6 + indices 3.6 + fold blocks 13.6 + stage transients) is accepted.

## Projection

M0-M6 landed: peak = max(stage window ≈ 30-33, fold window ≈ 42-46 with
blocks+NTT, commit window ≈ 39) → **~43-46 GB** (primary met if the fold
window behaves; M3 outcome is the swing). +M7: fold ≈ 30 → **~30-33 GB**
(stretch, below dory). Dory crossover even without M7: akita's slope
(~460 B/c) is below dory's (~520), so ≥2^27 akita wins regardless.
