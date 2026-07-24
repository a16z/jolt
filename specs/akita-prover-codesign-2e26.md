# Akita prover optimization — state and ranked opportunities at 2^26

Branch pair: `perf/optimize-akita-prover` (jolt) + `perf/onehot-commit-sweep`
(`~/akita`, path deps). Machine: M4 Max, 12P+4E cores, 128 GB. Workload:
sha2-chain, T = 2^26, K = 256, 29 committed one-hot columns, root n_a = 6.

## Campaign so far (terse)

333 s → **91.93 s** prove (3.62×); dory same-branch ≈ 112-114 s → akita is
~1.22× faster with transparent setup and ~2× faster verify. All landed
changes transcript-identical except catalog regenerations; gates green
throughout (muldiv host+zk, 224 akita unit tests, 38 jolt-akita, drift).

| Change | Effect |
|---|---|
| A1 sub-block chunking (akita 30db45c8) | commit off the fallback path: 333→154 s |
| J2+J3 decode dedupe (jolt db2d67889, df5860864) | stage1 24→8 s; shared Spartan code, sped dory up too |
| P1 rank-aware catalogs (akita e019b7c6, jolt 7d43d3411) | root n_a 7→6 for +112 B proof; 154→136.7 s |
| A5 fused multi-poly sweep + self-reducing accumulators + tuned tiles (akita 75d97724, 015669b9, 8a1fa829) | commit 96→~45 s; 136.7→91.9 s |

Dead ends (measured; do not revisit without new evidence): A4 lazy u128
accumulators (NEON wide wins 1.9×); A6 fused in-register widen (re-widens per
use, 100→131 s); merge-tile tuning beyond (64 blocks, 32 cols) — full bench
matrix flat within 5-30%; L1 tiles under block splitting (252 s, fixed by
self-reduction).

Facts that bound everything below: the PIOP already runs over the 128-bit
field (`AkitaPackedScheme::Field = AkitaFp128`, in-field challenges,
`MONTGOMERY_R_SQUARE = 1`); commit work = P·T·n_a·D coefficient-adds with
n_a·D security-pinned (~5.5 bits of width per +1 rank in the q128 table);
K=256 is geometry-optimal at 2^26 (bigger K inflates K·T address space →
rank/eval-ladder pushback).

## Measured attribution (98.8 s traced ≈ 92 s untraced)

commit 45-48 s | fold/opening 14.8 s (grind-fold 6.2 + accumulate 5.9 +
ring-relation ~2) | stage1 8.0 | stage6b lattice 8.0 | stages 3/4/5/6a/7
~17 | RSS 94.9 GB.

## Opportunities, descending by expected payoff at 2^26

### 1. Commit kernel gap: measured 61 ns/accum vs 26-44 ns modeled — eng, **−8 to −18 s**, medium confidence

Data: 11.7 G ring-accums / 45 s / 16 threads = 61 ns each. Model: per
accumulate the kernel moves ~6 KB through L1 (2 KB widened-A read from the
staged chunk, 2+2 KB accumulator RMW); P-core L1 sustains ~80 B/cycle →
~26-28 ns on a P-core. Three candidate explanations for the 2.2× gap, each
checkable and each with a distinct fix:
(a) **E-core drag**: rayon runs 16 threads on 12P+4E; if E-cores accumulate
at ~3× P-core cost, the aggregate lands near the measured number. Fix: pin
the sweep pool to P-cores or weight-split the block ranges. Expected −8-12 s
if confirmed.
(b) Sustained-clock/L2 effects at the 64-block tile (128 KB accums slightly
over L1). Fix: none cheap (bench matrix already flat) — this bounds the
downside.
(c) Entry-decode and cursor overhead per accumulate. Bounded by the bench's
own flatness at ≤ ~15%.
First step: run the sweep under CPU counters (Instruments) split per core
type — one afternoon, converts this item from medium to certain either way.

### 2. f128 delayed-reduction coverage in PIOP hot loops — eng, **−5 to −10 s**, medium-high confidence

Data: on comparable work the f128 PIOP measures 1.7× faster than the Fr
legacy PIOP (33 s vs 55.4 s after removing the ~9 s of lattice-only stages);
pseudo-Mersenne 2-limb multiplication is 3.5-4× cheaper than 4-limb
Montgomery CIOS at the op level, and element size halves memory traffic, so
2-2.5× end-to-end is the arithmetic potential. The shortfall concentrates
where the Fr stack has years of unreduced-accumulator/NEON tuning that
`AkitaFp128` paths may not: stage5 (4.5 s), stage6b lattice booleanity
(8.0 s), stage6a (3.3 s), `bind_parallel` (3.5 s over 3,141 calls).
`Folded128Product` exists — the audit is whether every hot `compute_message`
and bind path actually uses it rather than reduce-per-op.
First step: cycle-count one stage6b inner loop; compare against mul-count ×
6-8 ops. Fix pattern is mechanical once found (same shape as the existing
BN254 delayed-reduction code).

### 3. Committed-column virtualization — protocol, **−3 to −9 s commit + PIOP share**, needs a feasibility pass

Data: commit cost is exactly 45 s × (columns/29) ≈ **1.55 s per column**;
fold and booleanity also scale with P. Inventory at K256: 16 instruction
chunks + 8 increment chunks + 1 increment MSB + ~4 bytecode/RAM chunks.
Analysis so far: one-hot is already the cheapest *encoding* (1 accumulate
per chunk-cycle; dense small-scalar digits cost ~3×), so the win must come
from **not committing** columns whose values the PIOP can derive from other
commitments via existing claims — the bytecode/RAM address chunks (addresses
are functions of committed PC/RAM state already constrained elsewhere) and
possibly the increment MSB. Realistic range 2-6 columns.
First step: column-by-column dependency audit against the stage-3/4/6
claims: for each, either name the derivation sumcheck that replaces it (and
bound its cost — it must be ≪1.55 s equivalent) or strike it. This is the
item that decides whether 2^26 lands under ~65 s.

### 4. Fold-pass fusion — eng, **−4 to −6 s**, high confidence

Data: `fold_grind_sample` (6.2 s — a single accepted fold, no rerolls this
run) and `onehot_accumulate` (5.9 s) are back-to-back passes that both walk
the identical per-block entry lists applying challenge weights
(`decompose_fold.rs` already imports `accumulate`). Fusing them into one
entry walk halves the traversal and entry-decode cost; the arithmetic
differs per pass but the walk dominates. Same co-design shape as A5, kernel-
local, byte-equality testable.

### 5. Stage1 residual — eng, **−2 to −3 s**, high confidence per piece

Post-J3 stage1 is 8.0 s: uniskip extended evals 3.1 (now dominated by the
S64/S128 integer products, SIMD-izable), linear-stage materialise 3.1,
claimed-inputs 1.8. Bounded but cheap; benefits dory equally (shared code).

### 6. RSS to ≤60 GB (M2) — memory, enables 2^27; possible 1-3 s side win

94.9 GB at 2^26 vs 36.6 GB dory; ~30 GB is unattributed (readings varied
76-95 GB with phase overlap, suggesting peak-coexistence rather than one
slab). Known: expanded A 12.6 GB, block cache 15.6 GB (droppable or
u8-recomputable after commit — entries are recomputable from indices at
K≥D), fold buffers ∝ ppb. First step: allocative run at 2^24; then lifetime
fixes. Gates 2^27+ (projected ~380 GB at 2^28 otherwise).

### 7. Not at this scale (recorded to keep the ladder honest)

- **K=2^16**: at 2^26 roughly commit −12% / PIOP −25% but rank →~9, A must
  be seed-streamed, catalogs need nv=42; net ≈ neutral here. Becomes
  favorable ~2^28-2^30 (P·n_a 153 vs 203 at t=30). Revisit with a planner
  candidate-dump at nv 42 before believing my ±1 rank model.
- **Rank-hold at nv≥35 (P3)**: n_a creeps 6→7→8 with T (+5.5 bits/rank law);
  wider-slack candidate dumps at (36,·)/(38,·) may find n_a=6 corners.
  Matters from 2^28 up.
- Security-dimension or norm changes: out of scope by decision.

## Composed outlook at 2^26

92 s − (1: ~12) − (2: ~7) − (3: ~6) − (4: ~5) − (5: ~2) ≈ **~60 s ≈ 1.85×
dory**, without protocol-security changes. Items 1-2 alone (pure
engineering, no protocol review needed) reach ~75 s ≈ 1.5×.

## Protocol (unchanged)

Iterate 2^22, confirm 2^26 cooled and process-exclusive (`ps aux | grep
jolt_prover_legacy-` before every launch); same-session A/B or min-of-N —
single hot e2e readings misled four times this campaign. Gates per change:
muldiv host + host,zk; akita `--lib` suite; jolt-akita suite when catalogs
change; clippy both modes via `pipestatus`. Harness:
`PERF_LOG_T=26 [PERF_TRACE=1] cargo nextest run --release -p
jolt-prover-legacy --features akita -E 'test(sha2_chain_akita_perf)'
--run-ignored all --no-capture`. Commit per accepted iteration; failed
attempts get reverted and recorded under dead ends.
