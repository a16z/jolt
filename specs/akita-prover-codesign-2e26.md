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

## State (2026-07-24)

333 s at campaign start → 91.93 s now (dory same-branch ≈ 112-114 s).
Landed: A1 kernel chunking; J2+J3 decode dedupe (shared Spartan, sped dory
too); P1 rank-aware catalogs (root n_a 7→6); A5 fused multi-poly sweep with
self-reducing accumulators + bench-tuned tiles (commit 96→~45 s).
Attribution at 98.8 s traced: commit 45-48 | fold/opening 14.8 (grind-fold
6.2 + onehot_accumulate 5.9) | stage1 8.0 | stage6b lattice 8.0 | stages
3/4/5/6a/7 ~17 | RSS 94.9 GB.

Facts that bound the work: PIOP already runs over the 128-bit field
(`AkitaPackedScheme::Field = AkitaFp128`, in-field challenges); commit work
= P·T·n_a·D with n_a·D security-pinned; K=256 geometry-optimal at 2^26.

Dead-end ledger (measured; do not revisit): A4 lazy u128 accumulators (NEON
wide wins 1.9×); A6 fused in-register widen (100→131 s); merge-tile tuning
beyond (64 blocks, 32 cols) — bench matrix flat; L1 tiles under block
splitting (252 s; fixed by cap-triggered self-reduction, akita 015669b9).

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

### Q1 [ENG] Commit kernel gap: 61 ns/accum measured vs 26-44 ns modeled — expect −8 to −18 s

Rationale: 11.7 G ring-accums / 45 s / 16 threads = 61 ns; the L1-traffic
model (6 KB/accum at ~80 B/cycle) says 26-28 ns on a P-core. Hypotheses, in
test order: (a) E-core drag (rayon uses 12P+4E; if E-cores run ~3× slower
per accum the blended rate matches measurement) — test by running the sweep
with a 12-thread P-core-affine pool (or `RAYON_NUM_THREADS=12` +
QoS-pinning) in the kernel bench first, then e2e; (b) sustained clocks /
L2 residency at the 64-block tile — bounded by the flat bench matrix, do
not re-tune tiles; (c) per-entry decode overhead — bounded ≤15%.
First step: merge_sweep_bench under varying thread counts (16/12/8) — the
per-thread ns curve immediately separates (a) from (b)/(c).
Accept: e2e −≥3 s. If (a) confirmed but the fix helps <3 s (E-cores still
net-positive), record the measured split and close the item.

### Q2 [ENG] Stage-6b index-builder hoisting + cross-stage index sharing — expect −2 to −5 s

Audit-grounded (see appendix). Three 6b `initialize()`s rebuild d chunk
columns with one full `trace.par_iter()` PER CHUNK, recomputing the
chunk-independent base each pass: `LookupsRa`
(instruction_lookups/ra_virtual.rs:218 — `to_lookup_index` evaluated
instruction_d=16 × T; 15 redundant full-trace passes), `RamRaVirtual`
(ram/ra_virtual.rs:220 — `remap_address` ram_d×T), `BytecodeReadRafCycle`
(bytecode/read_raf_checking.rs:700 — `get_pc_for_cycle` bytecode_d×T).
Fix: one pass computing the base, extracting all d chunks (the pattern
already exists in `RaIndices::from_cycle`, shared_ra_polys.rs:145).
Second leg: `compute_all_G_and_ra_indices` (shared_ra_polys.rs:229) already
materializes all families' indices in 6a — thread them to 6b instead of
rebuilding (`get_pc_for_cycle` currently computed over T in ≥3 places);
caveat: LookupsRa uses the virtual chunk width, so share the base
`lookup_index`, not the chunks.
CLOSED-alternative recorded: the f128 arithmetic audit found NO
reduce-per-op in any T-sized hot loop (all `Folded128Product` /
`par_fold_out_in_unreduced`; mul is 35-instr aarch64 asm ≈ 6 mul-ops) — the
old "delayed-reduction coverage" hypothesis is dead. Remaining arithmetic
crumbs are in Q5.

### Q3 [PROTOCOL] Committed-column virtualization — expect −3 to −9 s commit + PIOP share

Commit costs 1.55 s per committed column (45 s / 29). Inventory at K256:
16 instruction chunks + 8 increment chunks + 1 increment MSB + ~4
bytecode/RAM chunks. One-hot is already the cheapest encoding, so the win is
NOT re-encoding — it is not committing columns the PIOP can derive from
other commitments via existing claims (candidates: bytecode/RAM address
chunks, increment MSB).
Protocol gate: produce a design note (new file in specs/) enumerating, per
candidate column, the replacement derivation sumcheck and its cost bound,
soundness argument sketch, and the affected `input_claim_constraint` /
BlindFold surfaces (CLAUDE.md invariant). STOP after the note — user
approval required before implementation.

### Q4 [ENG] Fold-pass fusion — expect −4 to −6 s

`fold_grind_sample` (6.2 s, single accepted fold — no rerolls to hide) and
`onehot_accumulate` (5.9 s) walk identical per-block entry lists with
challenge weights back-to-back (`decompose_fold.rs` imports `accumulate`).
Fuse into one walk; byte-equality test against the unfused pair (same shape
as the A5 equality tests). Watch: the grind may in principle reroll — the
fused path must preserve probe-order semantics exactly (transcript
identity).
Accept: e2e −≥2 s, akita suite green.

### Q5 [ENG] Stage1 residual + audited small items — expect −2 to −4 s

Stage1 (8.0 s): uniskip extended evals 3.1 (S64/S128 integer products,
SIMD candidates), materialise 3.1, claimed-inputs 1.8; shared code, muldiv
gates mandatory. Audited smalls to bundle: `CompactPolynomial::bind_parallel`
LowToHigh frees the old buffer synchronously each round — use
`drop_in_background_thread` (compact_polynomial.rs:256, 3,141 calls);
shift Phase2 serial `for j in 0..half_n` loop + reduce-per-op products
(spartan/shift.rs:866); bytecode address-phase reduced products
(bytecode/read_raf_checking.rs:472). Each bounded (√T/K domains).

### Q6 [ENG] RSS attribution and reduction — target ≤60 GB; possible 1-3 s side win

94.9 GB vs dory 36.6; ~30 GB unattributed (peak varied 76-95 GB with phase
overlap). Known slabs: expanded A 12.6 GB; block cache 15.6 GB (droppable
post-commit or rebuildable from u8 indices at K≥D); fold buffers ∝ ppb.
First step: allocative build at 2^24 (`RUST_LOG=debug --features
allocative`), attribute the peak, then lifetime fixes (drop/rebuild block
cache around the fold, free A after last sweep). Perf-neutrality gate ±2%.

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
