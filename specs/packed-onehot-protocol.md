# Packed one-hot protocol — evidence and decision

Bars at 2^26 (adjacent same-night pair vs prove 101-103s): ACCEPT prove
≤80s with commit ≤34s AND batched_prove ≤18s traced; PARTIAL 80<prove≤95;
FAIL <−8s → revert.

STATUS: runtime ACCEPT at T=2^26; implementation, soundness, and local
revalidation complete (2026-07-28).

## Adopt-and-extend checkpoint

PATH A is selected. Quang's seven Jolt commits through `a740e209c` were
cherry-picked with their original authorship and boundaries. The two Akita
kernel prerequisites (`a96b26fa`, `0d38c126`) were cherry-picked onto the
memory-optimized Akita branch; `050d93bb` was not duplicated because this
branch already contains the equivalent rank-slack policy. The Jolt-owned
catalogs were subsequently regenerated because their checked-in entries
predated that policy.

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

## Planner correction

The first production sparse-root run logged `n_a=7` even though the current
10‰ rank-slack planner selected `n_a=6` for the same `(39 vars, 1 poly)`
K256 layout. The checked-in Jolt catalog was old. A direct planner probe
showed:

| policy | root n_a | positions/block | blocks | proof payload |
|---|---:|---:|---:|---:|
| zero slack | 7 | 2^20 | 8192 | 99,292 bytes |
| 10‰ rank slack | 6 | 2^21 | 4096 | 99,404 bytes |

The narrower root costs only 112 payload bytes (+0.113%). Regenerating both
Jolt catalogs changed the K256 production entry to the second row. The slow
catalog-drift oracle, catalog coverage tests, K16/K256 e2e cases, verifier
fixtures, and tamper tests all passed afterward.

The regenerated K256 setup envelope is 12,582,912 ring elements
(12,884,901,888 bytes in field form). That size exposed a second interaction:
releasing the field-form setup immediately after commit made Stage 8 derive
setup data again and made in-process verification regenerate the full matrix.

## Production measurements

Workload: `sha2-chain`, 17,785 iterations, padded trace T=2^26, K=256,
release build, `PERF_TRACE=1`. The original bars were:

- ACCEPT: prove <=80s, commit <=34s, and `batched_prove` <=18s;
- PARTIAL: 80s < prove <=95s;
- FAIL: less than 8s improvement.

| variant | commit | batched prove | total prove | verify | peak RSS |
|---|---:|---:|---:|---:|---:|
| pre-protocol baseline | 42.90s | 25.60s | 103.30s | — | — |
| implicit-zero, catalog n_a=7, release matrix | 38.62s | 14.89s | 87.08s | 0.130s | 43.07GB |
| implicit-zero, regenerated n_a=6, release matrix | 32.08s | 21.72s | 85.97s | 2.54s | 32.87GB |
| n_a=6, retain field matrix, valid run 1 | 31.16s | 14.06s | 76.29s | 0.185s | <=43.61GB (bound) |
| n_a=6, retain field matrix, valid run 2 | 30.19s | 13.66s | 74.13s | 0.182s | <=43.61GB (bound) |

The two valid retained-matrix runs average 75.21s prove, 30.68s commit, and
13.86s `batched_prove`. Both clear every ACCEPT bar. Against the 103.30s
trace, the mean reduction is 28.09s (27.2%).

One additional retained-matrix run produced 106.56s prove
(`commit=30.91s`, `batched_prove=22.91s`). It did not follow the declared
120-second cooldown. Its trace shows the commit at normal speed followed by
broad 1.4–2x inflation across unrelated CPU-bound stages. It is preserved as
an excluded run rather than folded into the accepted pair.

The sandboxed full run could not emit `/usr/bin/time -l`'s peak-RSS field.
The exact setup size gives a hard bound: adding the entire 10.74GB released
suffix to the measured 32.87GB release-matrix peak yields at most 43.61GB.
This ignores the temporary derivation buffers that retention removes, so it
is conservative. A matched T=2^25 A/B measured the trade directly:

| T=2^25 K256 | batched prove | total prove | verify | peak RSS |
|---|---:|---:|---:|---:|
| release field matrix | 13.98s | 44.45s | 1.26s | 25.05GB |
| retain field matrix | 8.26s | 40.10s | 0.104s | 28.18GB |

## Why the optimizations stack

The implicit-zero protocol removes public zero-lane contributions from the
single committed polynomial. The regenerated rank-aware schedule then lowers
the root rank from seven to six and halves the live block count, which cuts
commit from 38.62s to roughly 31s.

That schedule also enlarges the setup extent. Releasing its field form after
commit traded the commit gain for setup derivation in `batched_prove`; it was
a memory/runtime policy conflict, not a protocol failure. The final lifecycle
drops the transformed NTT slots after commit but retains the field matrix
through Stage 8. The trace records 10.74GB and 11.41GB NTT releases while
avoiding setup re-derivation. This restores `batched_prove` to roughly 14s
without adding a commitment, opening, proof field, or sumcheck round.

## Fork and merge findings (Path A cost)

- quangvdao/akita@050d93bb diverges from our 3710a42c at merge-base
  1b17ad53: his side = upstream #332 + #333 + one planner commit
  (050d93bb); OUR M2-M8/T2 RSS work is entirely absent from his rev
  (65 files, −3795 vs ours) — mandatory reconciliation for any 2^26 use.
- #332/#333 cherry-pick cleanly onto 3710a42c (done on akita
  `perf/packed-onehot`). 050d93bb conflicts in 3 files because OUR tree
  already ships the same rank-slack planner feature (identity-bound slack
  tags, slack-candidate sweep). His commit is an earlier-base parallel
  version, so it was not duplicated. Regenerating the Jolt-owned catalogs was
  the missing operational step.

## Stage-8 soundness read (design level: PASS)

Verifier `stage8/packed.rs` binds `one_hot_trace_point` and
`one_hot_trace_evals` before sampling the selector challenge. It recomputes
the packed evaluation `sum_i eq(selector,i) * eval_i` from those bound
leaves, so the prover cannot adapt semantic evaluations to the selector.
All fixture-enabled verifier tests pass, including clear-claim and
commitment-wire tampering.

## Validation verdict

Highest evidence stage: **revalidated** for this local engineering target.
Two production-size runs that followed the cooldown contract clear the
predeclared ACCEPT bars, and the mechanism transfers to a matched T=2^25
release/retain A/B. It has not been replicated on another machine.

Correctness and compatibility gates:

- natural K16, forced K256, committed-program, and both advice e2e paths;
- all Akita claims tests and all fixture-enabled verifier tests;
- standard and ZK Dory mul/div non-regression tests;
- catalog regeneration drift and coverage checks;
- affected Akita, standard, and ZK Clippy targets with warnings denied.

## Decision

ACCEPT PATH A. Keep Quang's packed single-polynomial kernels, the implicit
public-zero protocol, the regenerated rank-aware catalogs, and the
post-commit NTT-only release. The result reduces the T=2^26 prover from
103.30s to a revalidated 74.13–76.29s while retaining a conservative
43.61GB peak-RSS bound.
