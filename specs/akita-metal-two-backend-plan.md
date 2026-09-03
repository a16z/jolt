# Akita: one catalog, two backends at their floors, and the path to 10 MHz

Status: proposal, 2026-09-03, on `feat/akita-metal` aadc36243 (Akita fork 3d748d499).
Numbers are the unified-branch T28 measurements in `akita-metal-10mhz-attack-strategy.md`
section 7.5 and the ledger unless marked otherwise.

## Decision

Replace the backend-specific K256 catalogs and the verifier's fallback chain with one catalog
per source class that lists every admissible row per key, a prover-side cost model that picks
the row for the backend that will commit, and a verifier that resolves rows by digest across
the whole catalog. Then remove the reason two rows exist at all by giving Metal a D128 rank-3
kernel, so both backends run the same row at their respective floors. The CPU target of about
2 MHz and the Metal target of 10 MHz at T=2^28 are reached only with the protocol-side items
in section 3; the software items alone land at about 1.7 MHz and 27 to 32 s.

## 1. Why the current shape is not idiomatic

Today (`crates/jolt-akita/src/configs.rs`, `schedules/mod.rs`): two generated K256 tables
(`jolt_fp128_onehot_k256` picks D128 rank 3 for the CPU, `jolt_fp128_onehot_k256_metal` pins
D512 rank 1 for Metal via hard-coded `METAL_ROOT_DIMS`), two prover configs
(`JoltOneHotK256Cpu`, `JoltOneHotK256Metal`), and a verifier config whose
`resolve_schedule_selection` tries the CPU table and falls back to the Metal table on
`UnsupportedSchedule` (`fallback_to_metal`, configs.rs:289). It works (the proof carries the
row digest; unknown digests fail closed) but it encodes a backend decision in the catalog
identity, resolves by exception, and needs a third table the day a third backend appears.

The geometry itself is a planner blind spot: the planner objective is setup size then
payload; commit cost never enters it, which is how D512 rank 1 became production although
D128 rank 3 has 25% less accumulate volume at the same bound and payload.

## 2. Target shape

Catalog. One `jolt_fp128_onehot_k256` table whose entries at each key carry all admissible
rows (today: D128 rank 3 at 2^19 positions and D512 rank 1 at 2^18; both at bound 1,048,575,
4 fold digits, 7 levels, 76 to 78 KB payload). The generator enumerates rows with a root
constraint; that API (`find_schedule_with_root_constraint`, `RootCandidateConstraint`) was
fork-only and does not exist on LayerZero main, so it is re-implemented in
`jolt-akita/src/planning.rs` over the upstream planner or upstreamed. Each row keeps its own
digest; the catalog identity covers the set.

Prover. A `CommitCostModel` per backend (CPU: ns per hot entry as a function of
`n_a * D`; Metal: streaming plus accumulate terms from the measured kernel model) and
`select_row(key, &dyn CommitCostModel) -> ResolvedScheduleRow` in jolt-akita that picks the
cheapest admissible row. No `METAL_ROOT_DIMS`, no per-backend config type; the backend passes
its cost model at prove time. The chosen row's digest goes into the proof exactly as now.

Verifier. `RegisteredRows` is populated from every table at startup, keyed by digest; lookups
never fall back. An unknown digest fails closed as today. Verifier preprocessing is unchanged.

Tests. The catalog drift test covers every row; a new cross-row acceptance test proves the
same statement with a D128 row and a D512 row and verifies both with one verifier; a
fail-closed test presents a digest from a foreign catalog.

End state. Once the D128 rank-3 Metal kernel (`akita-metal-d128-rank3-root-floor.md`) meets
its bar, both cost models pick the same row at T28 and the second row becomes dormant; the
mechanism stays for future geometries.

Cost: about two days in jolt-akita plus the planner query re-implementation; no verifier
protocol change; catalog regeneration.

## 3. Analyses toward 2 MHz on the CPU and 10 MHz on Metal

T=2^28 is 268.4 M cycles: 2 MHz is 134 s, 10 MHz is 26.8 s. Unified-branch Fibonacci today:
CPU 189.0 s (1.42 MHz), Metal 38.7 s (6.9 MHz); BTreeMap Metal 31.1 s, SHA-2 Metal 36.3 s.
Each item names the analysis to run before code, the lever it unlocks, and the predicted
saving. Order is by expected seconds per unit of work.

CPU (needs 55 s off Fibonacci; the software items below total 25 to 30 s):

1. Eval proof 38.9 s vs 22.3 s on the old protocol. Analysis: span-level diff of the CPU
   eval proof between `archive/akita-metal-v5-20260903` and the unified head at T28;
   count grinding attempts per fold response and the cost of
   `compute_multi_group_relation_quotient`. Lever: parallelize or bound the grinding search,
   overlap the quotient with the fold. Predicted 10 to 15 s. Protocol delta: none if the
   grinding bound is a prover-side search parameter; otherwise an upstream change.
2. Stage 6b 33.0 s vs 27.2 s and Stage 5 20.6 s vs 15.6 s. Analysis: per-member round timing
   against the archived line; the members are the same relations, so the delta is either the
   #1734 BundleStore data path or lost parallelism. Predicted 5 to 8 s together.
3. Commit 54.7 s at D128 rank 3, about 17 ns per hot entry. Analysis: bytes per hot entry
   (6 KB gathered plus accumulator traffic) against the measured per-core L2 bandwidth; the
   deferred path is bandwidth-bound, so vectorization buys little. Lever: NEON on the
   coefficient adds only if the analysis shows an issue-bound residue. Predicted 3 to 5 s.
4. Beyond software: 2 MHz needs 134 s. With items 1 to 3 the CPU lands near 160 s (1.7 MHz).
   The remaining 25 s is the commit's hot-entry count (Jolt witness: instruction chunks are
   51% and increments 37% of Fibonacci's hot entries) or a cheaper eval proof. Both are
   protocol work and the same levers Metal needs, below.

Metal (needs 12 s off Fibonacci, 4.3 s off BTreeMap, 9.5 s off SHA-2):

5. Recover the three regressions against the archived line, all measured: commit 17.3 s vs
   16.0 s (analysis: `packed_onehot_commit` bench at the T28 shape on fork 3d748d499 vs
   archive 0e52ebf17; the rebased kernel file lost 1,000 lines against the "tighten critical
   paths" commit, and the hybrid CPU tail share may differ), Stage 6a 1.90 s vs 0.56 s on
   Fibonacci (analysis: span diff of the bytecode address phase; L6a is SHA-2 only, so this is
   the port's route change), eval proof 7.29 s vs 5.85 s (`compute_multi_group_relation_quotient`
   0.73 s, `ring_switch_emit_z_planes` 0.69 s, `direct_digit_range_metal` 0.59 s: move the
   quotient and z-plane emission to the device or overlap them). Predicted 3.5 s.
6. Preserving levers already priced in the campaign and still open: release raw trace rows
   after the async prepare (1.7 s, needs a jolt-witness API change; the 4 GiB analogue
   regressed, so treat as risky), Stage 6b RA lazy rounds and Booleanity floor (0.5 to 1 s),
   eval host packing on the device or under Stage 7 (0.5 to 0.8 s), Stage 5 address installs
   (0.4 s; two attempts failed). Predicted 3 to 4 s.
7. D128 rank-3 root with per-ring-element tiles (`akita-metal-d128-rank3-root-floor.md`).
   Analysis first: re-run the geometry admissibility query under the #464 SIS tables (the
   Akita rebase could not reproduce it; the query API must be re-implemented), then the bench
   bar (slope at or below 3.2 ns per hot entry, intercept at or below 2.9 s, RSS at or below
   90 GiB). Predicted 3.0 / 4.3 / 4.7 s (BTreeMap / Fibonacci / SHA-2). Also collapses the
   dual catalog.
8. Interleaved A/B protocol for every claim above: the machine drifts 0.4 s per run and 15%
   under thermal load; the unified numbers are single cooled samples. No lever is retained
   without a paired complete-proof delta of at least 0.2 s.

Landing zones. Items 5 to 7 together: Fibonacci about 27 to 28 s (9.6 to 9.9 MHz), BTreeMap
about 24 to 25 s (10.7 to 11 MHz), SHA-2 about 28 to 29 s (9.3 to 9.6 MHz). The last 1 to 2 s
on Fibonacci and SHA-2, and the CPU's last 25 s, come only from fewer committed one-hot
symbols (a Jolt witness change constrained by the K=256 packing) or a 2x GPU. The 5x ratio
holds in every scenario where Metal reaches 10 MHz, because the CPU cannot exceed about
1.7 MHz without the same protocol work.

## 4. Not verified

D128 rank-3 admissibility under the widened SIS tables; the per-element-tile kernel's
efficiency; whether the eval-proof grinding bound is a free prover parameter; the CPU Stage 5
and 6b attribution (needs the span diff in item 2); whether Andrew's PR #1733 changes the
Stage 6 anchor in a way that interacts with our Stage 6a/6b kernels.
