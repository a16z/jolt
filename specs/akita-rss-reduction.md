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

### M1 [ENG] Cyclic-transform usage audit — decides ±15 GB of the plan

Trace which stage-8 kernels pull `neg` vs `cyc` from the prepared NTT cache
(akita: consumers of `PreparedNttCache` / `CyclotomicCrtNtt` in the fold /
commit_w_level / relation paths). The ring is negacyclic (X^64+1); if `cyc`
serves no path reachable from the packed prove at these shapes, M3 is a
mode change; if it is used, M3 becomes per-transform laziness instead.
Deliverable: a list of (kernel, transform used) with file:line, plus a
one-line verdict in this spec.

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

### M3 [ENG] Drop the unused NTT transform — expect −15 GB (fold window), conditional on M1

If M1 says `cyc` is unreachable: build the envelope slot
`NegacyclicOnly`-equivalent (mode exists for exact-negacyclic; a
both-transforms-minus-cyc variant is a small enum addition in
akita-types/ntt_cache.rs). If `cyc` IS used: split the slot so each
transform materializes on first use and the unused one never does.

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
