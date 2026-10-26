# Whiteout planning time (O2)

Parent: `specs/whiteout-optimization-campaign.md` (O2). The change is in
Akita (`wo/akita-o2`, crate `akita-planner`); this note lives beside the
campaign because the latency it targets is Whiteout's first verify (D32).

## Where the time goes

Measured on the response statement (`RecursiveCommitmentConfig<ResponseDigits>`,
the call `PlannedResponse::plan` makes), one thread, before the change:

| `e` variables | wall | suffix calls | distinct states | candidates |
|---:|---:|---:|---:|---:|
| 24 (muldiv) | 12.4 s | 272,275 | 32,717 | 1.53 M |
| 25 (fib 2^16) | 19.5 s | 394,300 | 45,499 | 2.26 M |
| 26 (fib 2^20) | 14.0 s | 294,129 | 34,063 | 1.78 M |

- The search is the suffix DP: a state is a level's input (level, witness
  length, basis, source moment, dimension ceiling, topology), memoized by
  key, 88% hit rate. The root is negligible.
- Exclusive time by level (ν = 24): level 2 is 55% (18,141 states, of which
  13,251 setup-prefixed), level 3 11%, levels 4–12 28%, level 1 6%.
- Per state, candidate generation is 59% (split walks; 31% of all time is
  `planned_next_witness_len`, and 34% of its calls repeat a candidate within
  one split search), child-edge pricing 17%, source moments 12%.
- No single hotspot: allocation is about 20% and spread over the layout
  builders. The cost is the number of states times a few hundred
  microseconds each.

## What changed

**Successors are completed in parallel** (`akita-planner` feature
`parallel`, enabled by jolt-akita). Before a state traverses its
candidates, it completes the successor states they request on rayon's
pool, then runs the unchanged ordered traversal, which now finds them in
the memo. The memo, the setup-prefix cache and the diagnostics are shared
across threads (mutex and atomics).

On the jolt side, `ResponsePlans::get` now plans outside its `derived`
lock and keeps the first row inserted. Holding the lock across a planning
call that runs on Rayon could deadlock: a worker waiting in a nested join
may steal another verify's job that waits on the same lock. The masked
catalog and residual matrices were already built outside their cells.

Nothing is pruned that was not pruned before. No search order, bound or
tie-break changed.

## Why the selected schedule cannot change

- **The memo is a cache of a function.** A memo entry is the frontier of
  its key, computed by `derive_selected_suffix_schedule` from the key and
  the search context alone. The sequential search already relies on this:
  a hit returns whatever the first visitor computed. Depth equals level, so
  the recursion bound is a function of the key too. Which thread computes
  an entry, and when, therefore never changes its value. Two threads may
  compute one entry (a claim set steers threads apart but does not block);
  they compute the same value.
- **The traversal is unchanged.** Each state still prices its candidates in
  the same order against the same frontier with the same guided prunes. The
  parallel step only decides which entries exist before the traversal
  asks; a missing entry is computed by the traversal as before. A guided
  traversal (root, setup-prefixed states) is completed in pool-width waves,
  each against the frontier the earlier waves built, so pruned candidates
  mostly cost nothing; this affects only work, not results.
- **Extra states.** Early completion decides prunes at the start of a
  wave; the traversal's frontier only grows inside the wave, so it may
  prune a candidate whose successors were already completed. Early
  completion therefore explores states the sequential order never visits:
  the pool completes 1.33–1.50× the sequential state count (43,544 vs
  32,717 at ν = 24, 68,328 vs 45,499 at ν = 25, 48,393 vs 34,063 at
  ν = 26), counting those and states two threads both computed.
- **Errors.** An `Err` from a state completed early is dropped; the
  traversal reaches that state (or not) exactly as before, so the errors
  a search reports are unchanged.
- **Panics are not.** A panic in an extra state propagates out of the
  rayon join, so a search that succeeds sequentially would now panic.
  Quick audit (clippy `unwrap_used`, `expect_used`, `panic`,
  `indexing_slicing` on the `akita-planner` lib): every site on the
  candidate-generation and pricing path is invariant-guarded, and the same
  code already runs for every state the sequential search visits. In
  `akita-planner` they are: the diagnostics' fixed arrays indexed by
  exhaustive enum maps; the precommitted equivalence-class indices
  (`planner.rs:89–91,262,279`, `candidates.rs:129,271`), built from the
  same key and reached only at the root, which early completion never
  runs; the response model's bucket indices (`response_model.rs:155,157`,
  bounded by its loop); `expect` on a scheduled setup prefix's slot id
  (`setup_prefix.rs:213`); and `expect` on validated costs
  (`schedule_params.rs:403,407`). The layout builders in `akita-params`
  and `akita-schedules` that generation calls also index (tail, relation
  rows, opening layout, grinding order). I did not audit those for
  reachability from states the sequential order prunes. None has fired
  in the tests, the catalog regeneration or the profiles.

Ground truth (independent of this change, not a reimplementation):

- `gen_schedule_artifacts --require-catalog-baseline-match` regenerates
  every tracked family (clear and zero-knowledge) with the parallel planner
  and matches `artifacts/schedule-catalog.tsv` row digests and
  `artifacts/schedules` byte for byte.
- Akita's proof-hash dump (CI lane) against the frozen base dump.
- The planner's own exhaustive-oracle tests (`unpruned_search`, adaptive
  frontier, chunk ownership) pass with `parallel`.
- jolt `whiteout_e2e` and `fs_obligations`.

## Results

> **LOADED.** 1-minute load 64–130 throughout (M4 Max, 16 cores).
> Wall times are not decision-grade; the cycle counts are.

**jolt, the profile harness** (`profile --name <w> --scale <s> --backend
optimized`, LTO off, two rounds, order alternated), seconds in the
`whiteout_plan_response` / `whiteout_plan_catalog` spans:

| workload | response, before | response, after | catalog, before | catalog, after |
|---|---:|---:|---:|---:|
| muldiv 2^16 | 84.8 / 49.3 | 12.6 / 10.7 | 7.4 / 5.4 | 2.3 / 2.2 |
| fib 2^16 | 103.1 / 64.7 | 22.5 / 16.1 | 9.1 / 6.5 | 2.2 / 2.0 |
| fib 2^20 | 54.5 / 34.7 | 13.4 / 9.5 | 16.0 / 9.6 | 3.3 / 3.3 |

**Akita, the response search alone** (`find_schedule` on the response key;
CPU cycles from `time -l`):

| ν | before, 1 thread | after, 1 thread | after, pool | states, 1 thread / pool |
|---:|---:|---:|---:|---:|
| 24 | 52.1 G | 53.6 G | 92.8 G (7.9 s wall) | 32,717 / 43,544 |
| 25 | 78.5 G | 80.6 G | 142.9 G (19.5 s) | 45,499 / 68,328 |
| 26 | 55.4 G | 58.9 G | 102.3 G (14.2 s) | 34,063 / 48,393 |

- Wall at lower load (19–34; base, pool, pool, base): ν = 24 16.9 / 2.9 /
  2.7 / 11.8 s, ν = 25 24.7 / 4.9 / 4.7 / 18.2 s, ν = 26 12.7 / 3.3 / 3.2 /
  14.3 s.
- One thread costs 1–6% more cycles than before (locks, the edge
  decisions made twice).
- The pool does about 1.8× the work: 1.33–1.50× the states (pruned-order
  states plus states two threads completed together), and lock contention.
- Idle projection, not measured: 52–79 G cycles is 13–20 s at 4 GHz on
  one core. Spread over 16 cores, 93–143 G cycles is 1.5–2.2 s. So about
  1.5–2 s per response plan idle, not the 1 s target.

## Not done

- The rest of the way to 1 s is single-thread work per state. Generation
  is 59% of it, and repeated split walks inside one state (modeled and
  universal moment, terminal and fold views) regenerate equal candidates.
  Sharing those walks is a change to `candidate::recursive`, with its own
  identity argument.

- Memoizing `planned_next_witness_len` per split search (34% of its calls
  repeat) cost more than it saved: 282 G vs 239 G instructions at ν = 24,
  from hashing and cloning the candidate. A global cache would cover 70% of
  calls but hold about 1.7 M candidates. Neither taken.
- Blocking on another thread's in-progress state would remove the
  remaining duplicate work but can deadlock under rayon's work stealing
  (a waiting thread may hold a lower-level claim in a stolen frame). A
  bounded `yield_now` wait overflowed worker stacks in a probe. Not taken.
- **Stack depth has no hard bound.** A worker waiting in a nested join
  steals unrelated jobs onto its stack, so nesting depth is bounded only
  in practice. Measured (review, debug build): 256 KiB worker stacks
  overflow both the response search and the catalog search, 384 KiB pass
  both; release passed at 256 KiB. Rayon's 2 MiB default leaves at least
  5× headroom.
