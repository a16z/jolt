# Whiteout planning time, part two (O2b)

Parent: `specs/whiteout-optimization-campaign.md` (O2), after
`specs/whiteout-o2-planner.md`. The change is in Akita (`wo/akita-o2b`,
crates `akita-planner` and `akita-params`). jolt changes only its `[pin]`.

## Result

> **LOADED.** 1-minute load 92–195 (median 134) on the shared M4 Max (16
> cores) for the whole session; the idle gate never opened. Wall times are
> not decision-grade. The instruction and state counts are.

**jolt, the profile harness** (`profile --name <w> --scale <s> --backend
optimized`, interleaved, order alternated, 7 rounds; fib 2^20 lost its
seventh round to a full disk). Median [min–max] seconds in the spans:

| workload (`e` vars) | response plan, base | response plan, O2b | catalog plan, base | catalog plan, O2b |
|---|---:|---:|---:|---:|
| muldiv (24) | 13.61 [11.34–21.27] | 2.10 [1.69–3.03] | 2.22 [1.52–3.60] | 1.83 [1.41–2.69] |
| fib 2^16 (25) | 23.41 [18.57–34.53] | 7.63 [6.34–10.48] | 2.36 [1.97–4.04] | 1.73 [1.22–2.77] |
| fib 2^20 (26) | 14.24 [12.11–25.62] | 2.09 [1.86–2.57] | 3.39 [2.76–7.10] | 2.51 [1.86–2.94] |

**Akita, the response search alone** (`response_plan` example, one thread,
instructions retired from `time -l`; load-independent):

| `e` vars | base | O2b | states, base → O2b | root candidates completed, base → O2b |
|---:|---:|---:|---:|---:|
| 24 | 239.9 G | 66.4 G | 32,717 → 12,812 | 23 → 1 |
| 25 | 363.3 G | 227.8 G | 45,499 → 36,051 | 33 → 15 |
| 26 | 253.3 G | 74.6 G | 34,063 → 13,896 | 23 → 1 |

On the pool the search now completes 14,020 / 38,202 / 15,142 states (1.06–
1.09x the sequential count), down from 43,544 / 68,328 / 48,393 (1.33–1.50x,
O2's numbers).

Idle projection, not measured: scaling P0b's idle first-use response plan
(2.12 / 3.79 / 2.59 s) by the LOADED medians' ratios gives about 0.3 / 1.2
/ 0.4 s. muldiv and fib 2^20 should meet the ~1 s target; fib 2^16 is near
it. The catalog plan does not regress (0.73–0.82x the base).

## Where the time went (after O2)

Profiled with `sample` on the sequential search for `e` = 2^25, and with
temporary counters (removed):

- **Every state costs a few hundred microseconds, spread wide.** Candidate
  generation was 60% of the search. A third of all time was
  `planned_next_witness_len`, and more than half of that built a
  setup-prefixed candidate's relation geometry twice and materialized its
  whole witness layout to read one length.
- **The modeled and universal split walks repeated work.** A request with a
  source moment walks its splits twice; 36% of the per-split work rebuilt a
  core the other walk had already finished.
- **The pool did 1.5x the states of a sequential search,** and the extra
  states were not duplicates. Pool-width waves at the root completed
  first-level subtrees that the ordered traversal then pruned: with one root
  candidate per wave the pool computes exactly the sequential states (66
  first-level states instead of 104 at 2^25). Waves at setup-prefixed
  states changed no state counts; they only added barriers.
- **One memo lock.** About a tenth of the pool's thread time waited on the
  memo mutex (`ScheduleMemo::get`).
- **The critical path.** With the extra states gone, a work/span probe (own
  time per state, children of one wave in parallel) put the span at 11–14%
  of the work, so 16 threads could not use more than 7–9x. The path ran
  through the root's candidates one after another, and through the
  first- and second-level states' sequential generation.
- **The root's batch order.** The root traversed one batch per (opening
  work, inner basis), each in its own lower-bound order. A weak batch's
  candidates completed before a later batch's better incumbent existed to
  prune them.

## The fib 2^16 anomaly

The planner's cost is not monotone in `e` because the root's cost is the
number of root candidates it must complete, and that depends on how the
best candidates fall into the padded setup-envelope tiers (the first
coordinate of the V6 objective, a power of two).

All three sizes have about 1,840 root candidates. At 2^24 one root shape
(d = 512/64/64) reaches the 2^18 envelope; at 2^26 one shape reaches 2^19.
Once that single best-tier candidate completes, every other root candidate's
bound is strictly worse and is pruned. At 2^25 the lowest tier, 2^19, is
reached by 15 shapes across d_A = 256, 512 and 1024 with near-equal bounds
(exact scores 721.5–723.2 M against an incumbent whose suffix the bound
cannot see), so none prunes another and all 15 complete. Odd `e` splits the
root witness between two d_A tiers whose setups pad to the same power of
two.

In the base planner the batch order hid most of this: it completed 23 / 33 /
23 root candidates (the first batches, d_A = 64 and 128, ran before any
low-tier incumbent existed). In the global order the counts are 1 / 15 / 1,
so fib 2^16 is now the slowest by a wider margin (2.6x the states of the
other two). Each completed root candidate costs two first-level subtrees,
about 2,400 states.

## What changed

1. **Measured witness lengths** (`akita-params`). `WitnessLayout::new`'s
   chunk-major body walk moved to `witness/body.rs` and is shared by the
   materialized layout and by `measure_live_coeff_len`, which replaces
   `scalar_live_coeff_len` and accepts multi-group (setup-prefixed) levels.
   `planned_next_witness_len` measures every candidate. One body walk, one
   tail walk (`tail::measure`), one owner.
2. **Split replay** (`candidate/recursive`). The universal walk replays the
   visits the modeled walk recorded at a split when the two cores are equal.
3. **Sharded memo** (`suffix_dp/state.rs`). 64 locks by key hash, each with
   its share of the eviction quotas.
4. **Root waves and barriers** (`suffix_dp/search.rs`). The root's waves
   double from one candidate to the pool's width; every other state
   completes all its candidates' successors at once.
5. **In-state parallelism** (`suffix_dp/candidates.rs`, `source.rs`). A
   state's generation requests, its root opening works and its candidates'
   source moments run on the pool and are joined in order
   (`map_in_order`).
6. **Global root order** (`suffix_dp/candidates.rs`, `search.rs`). The root
   traverses all of an opening basis's candidates as one batch, so its
   lower-bound order is global.
7. **`response_plan` example** (`akita-planner`): plan time, catalog digest
   and artifact hash per `e` size, for comparing planner changes.

Rejected or not taken: a global cache of the per-split work (two thirds of
it repeats across states, mostly moment variants of one state at level 3
and deeper, but holding it costs about a gigabyte); a cheaper
`next_source_moment` (it needs the materialized layout, and the
geometry-reusing constructor would be a `_for_level` wrapper, which Akita's
AGENTS.md forbids; the double geometry costs 2.5% of the instructions);
parallel pricing inside a traversal (not needed once the root's waves
ramp).

## Why the selected schedule cannot change

Changes 1–5 change no value the search computes, only how it is computed or
which thread computes it:

- (1) The measured length is the materialized layout's length: the same body
  walk with no-op sinks, then `tail::measure` instead of
  `tail::materialize` (the existing measuring sink). akita-types
  `measured_live_length_matches_materialized_setup_prefixed_layout` checks
  it against the materialized layout for setup-prefixed levels, chunked
  (2 and 4 chunks) and not; the existing scalar tests cover single-group
  levels.
- (2) Everything a split walk derives after a split's core
  (`candidates_from_core`, `attach_recursive_setup_prefix`,
  `prune_locally_unprofitable_slices`, `finalize_recursive_level_candidate`)
  is a function of the core, the split, the relation domain, the request and
  the level search. The modeled and universal walks share all but the
  source moment, which enters only `candidate_core` (the modeled ℓ∞ cap).
  So at a split whose two cores are equal the universal walk visits exactly
  what the modeled walk visited, in order. A recording is keyed on the
  request's opening as well as the split, relation domain and core, so a
  walk from another request over the same level search (the selective-ℓ2
  path builds one, though today it does not walk) cannot replay it.
- (3), (4) Memo entries are functions of their keys (O2); sharding and
  eviction quotas change only which entries are resident. Wave sizes decide
  only which entries exist before the ordered traversal asks.
- (5) Each request, opening work and moment is a function of its inputs;
  the results are joined in the sequential order, and the first error in
  that order is the one returned.

Change 6 changes the order of the root's guided traversal, and with it which
root candidates are pruned. Whenever the search returns a plan, the plan is
the same in every order (byte-identical):

Let `U` be every complete schedule some root candidate yields when nothing is
pruned, and `m` its least element under `CompleteScheduleScore` (the
objective coordinates, the legacy root output length under V5, then the
full schedule descriptor). That order is total, and equal scores mean equal
descriptors, hence the same schedule.

1. What a root candidate `c` yields is a function of `c`: its successors'
   results are memo values, and a restricted (query-budget) lookup is a
   function of the key and the prefix `c` fixes.
2. Every root schedule fits the query limit. The root's successors are
   restricted queries, which admit only suffixes that fit
   (`QueryPrefix::admits`), so `select_complete_candidate`'s filter removes
   nothing.
3. `m` is never pruned. A drop (`complete_root_setup_bound_is_strictly_worse`)
   or a direct-edge prune (`complete_root_bound_is_strictly_worse`) needs
   `c`'s lower bound to be lexicographically strictly worse than the metrics
   of some `e` already in the root frontier, and `e ∈ U`. On every edge it
   removes, the bound is componentwise at most each schedule's objective
   prefix. This is the soundness the existing search already relies on:
   the envelope pads the maximum of this level's setup and the suffix's;
   the first direct capacity and output length are the candidate's own on
   the direct edge (the offloaded edge is dropped only when the policy
   plans no offloads); proof bytes and work omit only nonnegative successor
   terms. A prefix that is componentwise at least the bound, which is
   lexicographically above `e`, is itself lexicographically above `e`. So
   every pruned schedule is worse than `e`, and `m` is not among them.
4. `m` is never displaced. The frontier rejects or removes a setup
   candidate only for a `y` with `setup_primary_strictly_dominates(y, ·)`
   or `setup_projection_dominates(y, ·)`. Each implies `score(y) ≤
   score(·)`: envelope, then first direct capacity, then `(exact_score,
   proof_bytes)` through `strictly_better`/`never_worse` (the same
   lexicographic pair the score uses), then the output length (V6) or the
   setup elements (V5), then descriptor `≤`. Under V5 the score puts the
   legacy root output length before the descriptor; the descriptor tie
   requires the same first-fold descriptor, and the root's output length is
   its planned length aligned to the successor block length that
   descriptor carries, so the lengths are equal. By minimality
   `score(y) = score(m)`, so `y` is `m`'s schedule.
5. So the frontier ends holding `m` (or a copy), every other member scores
   at least `m`, and selection returns `m`'s schedule in every order that
   returns a plan: in the old batch order and in the global one alike.

Not covered: errors. The internal-invariant and overflow errors that can be
raised after a prune decision (`child_choice`, `PendingQueryEdge::new`, a
successor's `derive_selected_suffix_schedule`) are raised only by the
candidates an order explores, so whether the search succeeds, and which
error it reports, can depend on the visit order. By inspection they are
unreachable for well-formed input (checked arithmetic on bounded lengths,
and invariants that hold for every candidate the generators emit). A panic
in such a candidate is the same case. The regenerated catalogs and the
response statements raise none.

Ground truth (independent of these changes, not a reimplementation):

- `gen_schedule_artifacts --require-catalog-baseline-match` regenerates
  every tracked family with the final planner, sequential and parallel, and
  matches `artifacts/schedule-catalog.tsv` and `artifacts/schedules` byte
  for byte. LOADED, 4 jobs: 411 s parallel (peak RSS 6.7 GB), 819 s
  sequential (6.1 GB). The same check passed on an intermediate build with
  the global root order but one-candidate root waves, and the build before
  the global order (one-candidate waves, batch order) took 1,952 s and
  12.3 GB; the base build's regeneration was not timed.
- Akita's proof-dump identity over the CI test pass (1920/1920): exactly the
  5 known ZK-row omissions, no new entries.
- The response plan's one-row catalog digest at 2^24, 2^25 and 2^26
  (`response_plan`): identical before and after every commit.
- jolt: a hash of `WhiteoutPlans` (the masked catalog, the precommitted
  rows and every response artifact's bytes) exported after each of the 26
  profile runs above, base and O2b: three distinct values, one per
  workload, equal across builds.

## Gates

- **Akita** (`wo/akita-o2b`, bar run at `376095668`; `d3b2913de` after it
  changes doc comments only): clippy (`parallel,disk-persistence`,
  `+dev-protocol`, the zk/test-support/logging lane, the A46 akita-pcs and
  akita-cpu-backend `zk`/`zk-prover` lanes, the `zk_response` example, and
  `akita-planner` under none, `parallel`, `zk`, `zk,parallel`,
  `catalog-gen,parallel`, `catalog-gen,test-support,parallel`,
  `catalog-security,parallel`, `catalog-gen,dev-protocol,parallel`), fmt,
  taplo; CI nextest 1920/1920 with the dump identity above; no-disk-cache
  zk lanes 188/188 and 225/225, zk integration 37/37, workspace zk
  1889/1889, akita-cpu-backend `zk-prover` 549/549, akita-error `zk` 1/1;
  zk tree clean; the planner feature lanes 154/154 under
  `catalog-gen,test-support,parallel`, `catalog-gen,test-support` and
  `catalog-gen,test-support,zk,parallel`; schedule regeneration (clear,
  sequential and parallel; dev, parallel) byte-identical;
  `catalog_security --check` passes.
- **jolt** (`wo/o2b-planner`, built against the Akita head): clippy on
  jolt-akita (`zk`, `zk-prover`), jolt-prover (`akita,zk`;
  `akita,zk,profiling,prover-fixtures`; `akita,profiling,prover-fixtures`)
  and jolt-verifier (`akita,zk,prover-fixtures`), fmt; `fs_obligations`
  1/1, `whiteout_e2e` 6/6, jolt-akita 89/89, jolt-akita `zk-prover`
  whiteout 5/5, jolt-prover `akita,prover-fixtures` 41/41.

## Not done

- **fib 2^16 stays above the others.** Its 15 tied root candidates each need
  their subtrees. Pruning more of them needs a tighter suffix bound in
  `direct_edge_lower_bound` (today it omits the successor's payload and
  work), with its own soundness argument.
- **Duplicate states under the pool:** 6–9% of computations, two threads
  completing one state; blocking on a claimed state can deadlock (O2).
- **Stack depth** is bounded only in practice, as in O2.
