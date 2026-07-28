# Packed one-hot protocol — evidence and decision

Bars at 2^26 (adjacent same-night pair vs prove 101-103s): ACCEPT prove
≤80s with commit ≤34s AND batched_prove ≤18s traced; PARTIAL 80<prove≤95;
FAIL <−8s → revert.

STATUS: phase 1 implementation and soundness validation complete; production
benchmark pending (2026-07-28).

## Adopt-and-extend checkpoint

PATH A is selected. Quang's seven Jolt commits through `a740e209c` were
cherry-picked with their original authorship and boundaries. The two Akita
kernel prerequisites (`a96b26fa`, `0d38c126`) were cherry-picked onto the
memory-optimized Akita branch; `050d93bb` was not duplicated because this
branch already contains the equivalent rank-slack policy.

The one integration conflict was intentional: Quang's branch assumes a
permanently materialized setup matrix, while this branch releases that matrix
after commit. The streamed trace kernel now takes an owned covering snapshot
for its full-width read, after which the existing NTT/setup release remains in
force. The reconciled checkpoint passes:

- all 72 `akita-algebra` tests and focused Clippy;
- all 42 `jolt-akita` tests;
- `muldiv_e2e_akita`, forced-K256, and committed-program packed e2e;
- both packed advice e2e tests;
- affected Jolt library Clippy targets.

## Sparse-root experiment

A root-only experiment on Quang's packed source tested the representation
change before any protocol edits:

- public default lane zero was omitted for instruction, bytecode, RAM, and
  increment columns;
- increments used centered radix-256 digits plus a signed carry;
- the physical object remained one packed polynomial.

At the production T=2^26 K256 root geometry, an S-C-S-C bracket produced:

| root | sample 1 | sample 2 | mean |
|---|---:|---:|---:|
| full one-hot control | 68.441s | 69.545s | 68.993s |
| sparse/default-zero candidate | 35.259s | 36.445s | 35.852s |

Root contributions fell 1,815,318,886 → 898,701,558 (−50.49%); root wall
fell 48.04%. Mean RSS also fell 36.46GB → 34.77GB. Raw log:
`/private/tmp/jolt_sparse_root_20260728.log`.

This falsifies the earlier 93–97s projection, whose model assumed every
semantic row still incurred a root contribution. The central whole-prove
expectation is now about 83s, with an 80–90s planning interval until protocol
overhead and the opening-side effect are measured.

## Phase 1 protocol decision: implicit public zero

Let `a` be the K-ary lane, `t` the cycle, and `A(t)` the activation:

- `A(t)=1` for instruction, bytecode, and increment columns;
- `A(t)` is the existing RAM-access indicator for RAM columns.

Existing stages continue to use the semantic polynomial `P(a,t)`, which is
strict one-hot when active and empty otherwise. The single Akita commitment
instead binds

```text
S(a,t) = P(a,t) for a != 0
S(0,t) = 0
H(t)   = sum_a S(a,t)
P(a,t) = S(a,t) + eq(a,0) * (A(t) - H(t)).
```

For any address point `r`,

```text
P(r,t) = eq(r,0) * A(t)
       + sum_a S(a,t) * (eq(r,a) - eq(r,0)).
```

This is the load-bearing identity. Stage 6 still proves Booleanity and
virtualization for `P`. Stage 7 changes its address reduction so every
upstream `P(r,t)` claim is represented by the public baseline
`eq(r,0) * A(t)` plus the recentered `S` sum. The Stage-7 outputs are
therefore openings of `S` and can be passed directly to the existing
single-polynomial selector reduction in Stage 8.

Consequences:

- no second commitment or auxiliary row-weight polynomial;
- no extra Fiat–Shamir challenge, proof field, or sumcheck round;
- the existing semantic exact-one property is retained: `P` has row sum
  `A` by construction and its cells are covered by the existing Booleanity
  relation;
- lane zero is fixed by the protocol, never selected adaptively from witness
  frequencies;
- RAM inactivity is preserved by its existing `A(t)` claim.

For a Stage-7 RA member with Booleanity point `b`, virtualization point `v`,
and batching powers `(g_h, g_b, g_v)`, the output identity becomes

```text
baseline =
    eq(rho,0) * A *
    (g_h + g_b*eq(b,0) + g_v*eq(v,0))

sparse =
    S(rho) *
    (g_b*(eq(b,rho)-eq(b,0))
     + g_v*(eq(v,rho)-eq(v,0))).
```

Their sum equals the unchanged input expression
`g_h*A + g_b*P(b) + g_v*P(v)`. Increment members use the same Booleanity
recentering; their decode weight is zero at lane zero, so the decode term
needs no baseline correction.

## Balanced increment representation

For radix `B=K` and `n=64/log2(B)`, represent the signed fused delta as

```text
delta = sum_{i=0}^{n-1} d_i * B^i + carry * 2^64
d_i in [-B/2, B/2-1]
carry in {-1,0,1}.
```

Each signed value is encoded in its lane modulo `B`; lane zero is the public
implicit default. The decode polynomial is the signed lane value

```text
signed_id(a) = a                  if a < B/2
             = a - B              otherwise.
```

The construction applies unchanged at K16 and K256. The existing Rust wire
identifiers `UnsignedIncChunk` and `UnsignedIncMsb` are retained during this
experiment to avoid an unrelated proof-model rename; their protocol meaning
changes to balanced digit and signed carry and is bound by a new layout
digest version.

## Claim-to-code map

| Protocol obligation | Prover/source | Symbolic/verifier | Final binding |
|---|---|---|---|
| Commit `S`, omitting lane zero | `zkvm/packed.rs::JoltOneHotTraceRows` | canonical layout metadata in `jolt-claims/.../lattice/strategy.rs` | `jolt-akita/src/trace_onehot.rs` |
| Keep upstream semantic `P` | existing RA indices and full increment lane columns in Stage 6 | existing Booleanity and RA/read-RAF relations | Stage-7 recentering |
| Recenter Booleanity/virtualization claims | `claim_reductions/hamming_weight.rs` | `lattice/relations/hamming_weight.rs`; `jolt-verifier/stages/stage7/hamming_weight_claim_reduction.rs` | Stage-7 output claims now denote `S` |
| RAM activation baseline | existing `RamHammingWeight` claim | promoted to an Akita-only Stage-7 derived public from the already transcript-bound Stage-6 clear claim | same Stage-7 identity |
| Balanced digit/carry witness | `zkvm/packed_witness.rs`; `zkvm/packed.rs::fused_inc_columns` | lattice geometry and Hamming relation | Stage-7 signed decode |
| Commit/open transcript binding | unchanged semantic-evals-before-selector order | `jolt-verifier/stages/stage8/packed.rs` | layout digest v5 |

## Ambiguity register

- Scope: this representation is Akita-only. `akita` and `zk` are mutually
  exclusive Cargo features; the Dory/BlindFold protocol remains unchanged
  and must stay green as a non-regression gate.
- Default choice: fixed lane zero. An adaptive/modal default would require
  new public metadata and transcript binding and could leak trace
  distribution; it is out of scope.
- K regimes: centered base `K`, not a K256 special case. Forced-K256 and
  natural K16 tests are both required before performance runs.
- Padding: instruction/bytecode/increment semantic rows remain active
  one-hot `P` rows; a zero value is represented by an empty committed `S`
  row. RAM rows remain inactive when the existing selector is zero.
- Arithmetic range: fused deltas satisfy `|delta| < 2^64`; centered digits
  and the signed carry reconstruct over integers far below the fp128
  modulus. Boundary vectors must test both extrema and radix ties.
- Wire naming: the old unsigned/MSB enum names are temporarily retained.
  Promotion can include a follow-up rename, but performance validation must
  not be coupled to that mechanical blast radius.

## Phase 1 falsifiers and gates

Before T=2^26:

1. Concrete semantic tests must show the recentered identity for every
   column family at non-Boolean points and balanced reconstruction at
   boundaries.
2. K16, forced-K256, committed-program, advice, and verifier tamper suites
   must pass.
3. A staged benchmark must show contribution count tracking nonzero lanes
   and a material root reduction. If Stage-7 overhead consumes most of the
   root win, stop before the production run.
4. The production result is judged by the original ACCEPT/PARTIAL/FAIL bars;
   the root-only experiment is not promotion evidence.

Validation completed before the production run:

- concrete recentering and balanced-decomposition tests pass at K16 and K256;
- natural-K16, forced-K256, committed-program, and advice end-to-end proofs pass;
- all 191 Akita claims tests and all 84 fixture-enabled verifier tests pass,
  including every clear-claim and commitment-wire tamper;
- the standard and ZK Dory mul/div gates pass unchanged;
- affected Akita, standard, and ZK Clippy targets pass with warnings denied.

## Phase 0a — planner probe: the n_a mechanism is FALSIFIED

Probe: `crates/jolt-akita/tests/schedule_probe.rs` (ignored test; resolves
schedules through the production configs, DP fallback on catalog miss).
Results under `JoltD64OneHotK256`/`K16` at 10‰ rank slack:

| shape | ppb | live blocks | root n_a | root fold bucket |
|---|---|---|---|---|
| (34 vars, 29 polys) current | 2^21 | 128/claim (3712 block-claims) | **6** | 2^20−1 |
| (39 vars, 1 poly) packed K256 | 2^21 | 4096 | **6** | 2^20−1 |
| (36 vars, 1 poly) packed K16 | 2^18 | 4096 | **6** | 2^20−1 |

Why packing cannot move n_a: claims 29→1, but the packed domain grows to
capacity·K·T so live blocks rise 3712→4096 — the t*-driving product
claims×blocks is UNCHANGED (+10% from the 3/32 capacity padding), so
δ_fold/bucket stay put, and at bucket 2^20−1 the q128 D=64 SIS row gives
rank 6 for any width in [65609, 4178804] — both ppb 2^20 (quang's schedule)
and 2^21 sit in the same band. Rank 5 needs width ≤65608 (ppb ≤2^16), which
multiplies blocks 32× and pushes the bucket UP. The geometry is
self-balancing; n_a=6 is a hard floor here (consistent with the 2026-07-24
campaign's Q1/Q3 verdicts).

Bucket-down routes checked and closed analytically:
- blocks↓ via ppb↑: rank-6 width cap is 4178804 < 2^22 — ppb is maxed.
- δ_fold 4→3 via a tighter fold-l∞ grind cap: the grind currently spends
  ~1.5s beyond the mandatory root fold (see below), i.e. the cap already
  sits at the challenge-fold distribution's natural tail; an 8× tighter cap
  has vanishing acceptance probability.
- K-regime change: cols(K)≈263/log2(K)+1 shrinks entries, but domain
  cols·K·T grows ~K/log2K → blocks → bucket → rank. At K=2^12:
  ~21 cols but rank ≥7 ⇒ 21·T·7 ≥ 29·T·6·0.84 — no win; at K=16: rank
  would need ≤5 to beat 29·6, but bucket stays 2^20 (probe) ⇒ 57·T·6 is
  2× worse. K=256 is near-optimal for accumulate count.

## Baseline window decomposition (tonight's traced 2^26 A side, prove 103.3s)

- commit 42.9s: `onehot_merge_sweep` 622.9 thread-s (÷16 ≈ 39s wall — the
  accumulate wall), everything else ≪1s.
- batched_prove 25.6s: `ring_relation_fold_grind` 11.65s ⊇ root
  `decompose_fold_batched` 10.09s (the grind's overhead beyond the mandatory
  root fold ≈ 1.5s); ring-switch/quotient/finalize ≈ 12 thread-s across 8
  levels; per-claim work that packing removes: `OneHotPoly::build_blocks`
  3.67 thread-s (x29) + root-level per-claim eval/tensor/transcript ≈ 2-5s
  wall total.

## Feasibility projection for the packed protocol at 2^26

n_a unchanged ⇒ commit floor ≈ 29·T·6 accums at the measured sustained wall
(65-74 ns/accum ×16t) ≈ 37-40s. Packing buys: emission/materialization ≈ −3s
(measured on the fused-sweep dead end, same mechanism), batched_prove
per-claim collapse ≈ −2..−5s, witness-gen columns never built ≈ −1..−2s.
Projected total: prove ≈ 93-97s — BELOW the PARTIAL floor at worst, far from
ACCEPT ≤80s. The two invariants (accumulate wall × n_a=6; per-level fold
machinery) are untouched by packing.

Empirical arbitration pending: PR #1706's stack AS-PINNED (his jolt head +
quangvdao/akita@050d93bb git dep, no reconciliation — valid at 2^24 where
the missing M2-M8 RSS work doesn't bind) vs our stack, quiet adjacent 2^24
pair. If his measured delta ≈ the projection, infeasibility of the ACCEPT
bar is confirmed empirically + analytically; if much larger, the window
model is missing something and phase 0 continues.

## Fork and merge findings (Path A cost)

- quangvdao/akita@050d93bb diverges from our 3710a42c at merge-base
  1b17ad53: his side = upstream #332 + #333 + one planner commit
  (050d93bb); OUR M2-M8/T2 RSS work is entirely absent from his rev
  (65 files, −3795 vs ours) — mandatory reconciliation for any 2^26 use.
- #332/#333 cherry-pick cleanly onto 3710a42c (done on akita
  `perf/packed-onehot`). 050d93bb conflicts in 3 files because OUR tree
  already ships the same rank-slack planner feature (identity-bound slack
  tags, slack-candidate sweep) — his commit is an earlier-base parallel
  version; probe shows our planner already rank-selects. Not picked.

## Stage-8 soundness read (design level: PASS)

Verifier stage8/packed.rs binds `one_hot_trace_point` then
`one_hot_trace_evals` into the transcript BEFORE sampling the selector
challenge, and recomputes the packed evaluation Σ eq(selector,i)·eval_i
itself from the bound leaves — the prover cannot adapt semantic evaluations
to the selector. Remaining diligence (phase 2 if we proceed): explicit
tamper/negative test for the binding order; padding-slot zero semantics are
a completeness concern only.

## Phase 0b — 2^24 as-pinned pair (K16 regime at this size)

Note: at 2^24 both stacks select the K=16 chunk regime (57 semantic columns
native / 64-slot packed), so this pair probes packed-vs-native in the K16
regime; the K256 question needs 2^26 (or a forced-K256 run).

- First attempt (2026-07-27 22:29): wrapper timeout killed the pipeline
  mid-quiet-gate (ambient load ~5 for ~2.5h: code42 agent, Spotlight over
  fresh build artifacts, video decode). The orphaned script still ran side
  A hot (a ~33-min rebuild absorbed into the timed window immediately
  before the test): OURS at 2^24 K16 = prove 64.88s, peak 25.7GB —
  DIRECTIONAL ONLY.
- Clean rerun (detached, file-logged, gate load<6 with ambient documented,
  order B-then-A, both sides prebuilt): PENDING.

## Decision

PATH A — adopt and extend Quang's packed single-polynomial implementation.
The root-only sparse experiment is the decisive evidence that the original
all-rows-active projection missed a roughly 2× root lever. The protocol
change above is now the shortest credible route to the acceptance band while
preserving this branch's memory work.
