# Akita Metal: unifying the campaign line, the port line, and upstream

Status: executing, 2026-09-03. Decisions taken by the user: PR #1733 (Andrew's Dory Metal
backend) is out of scope for now and our kernels stay at `crates/jolt-kernels/src/metal/`;
`feat/akita-metal` may be force-updated once the unified branch passes its gates, after
archiving; the Jolt base is main with the legacy prover removed (PR #1818's branch merged,
assumed to land). Phase 0 (archives) is done: Jolt `archive/akita-metal-v5-20260903` =
4a0ead2fd (09e649061 plus the retained source and specs committed), `archive/port-akita-metal-latest-20260901`,
`archive/metal-prod-slim-20260817` on a16z/jolt; Akita fork `archive/port-metal-latest-20260901`,
`archive/metal-commit-eval-proof-20260830`. The study ledger under `benchmark-runs/` is
gitignored and stays in the bright-ridge worktree; the retained patch also sits in the scratchpad.

## Decision

Rebuild `feat/akita-metal` (a16z/jolt) and a same-named `feat/akita-metal` branch on the
Akita fork (markosg04/akita) as main-based branches that carry the Metal work, and retire
every other Metal branch to an archive tag. The base is the existing port line
(`port/akita-metal-latest` in Jolt, `port/metal-latest` in Akita, both 2026-09-01, both
local-only), moved forward to today's mains, plus the four deltas the campaign line
produced after the port forked. Merging the campaign line into main is not an option: it
sits on the 2026-08-13 main (protocol v5), 520 commits diverged, and main absorbed the
CPU-side Akita work through reviewed rewrites (#1718, #1731, #1732, #1792, #1796).

## 1. Inventory

Jolt (a16z/jolt):

| line | head | base | contents | state |
|---|---|---|---|---|
| `origin/main` | 7d33a217c (09-03) | | Akita CPU prover at protocol v7, K256 catalog D512 at nv 40-41, no Metal, pins LayerZero akita 4505404b5 | canonical |
| PR #1818 `chore/remove-jolt-prover-legacy` | 29dfab798 (09-02) | main | deletes the legacy prover; `akita/preprocessing.rs` hand-ported | OPEN, dirty |
| `feat/akita-metal` (campaign line, this worktree) | 09e649061 + 1,664-line uncommitted patch | main 08-13 | Solinas PIOP kernels, `modular_benchmark` + 10 evaluator examples, T28 campaign, SHA-2 fix 7b2c6d617, CPU accumulator fix 09e649061, retained L1a/L1c/L6a source, specs and ledgers | verified T28 matrix; pushed at 62320b9f0 only |
| `port/akita-metal-latest` | 2094ce34a (09-01) | main 08-27 + PR #1796 | Metal import from feat 8a5f238f4, adapted to the shared-field stack, packed openings and ring switch routed to Metal, K16 commits on Metal, zero-suffix certification, per-backend K256 catalogs (CPU D128 rank 3 at 2^19 positions, Metal D512), its own Stage-1 register fix 906c717bd | local only; Akita path dependency; no benchmark harness; 20 behind main |
| `refactor/metal-prod-slim` | 7f75dbf4c (08-17) | feat a791c8a3d | 12 slimming commits (error enum collapse, offload relocation) | stale, superseded by the port's adaptation |
| PR #1733 `feat/metal` (Andrew Tretyakov) | 46494eb42 (09-02) | main 08-27, main merged 09-02 | independent Metal backend: BN254 Fr kernels, device-resident Dory reduce and streaming commitment, device slots for every optimized-tier sumcheck family, 92 kernels, 1.1 MB under `crates/jolt-kernels/src/metal/`, feature `metal`; zero code overlap with ours | OPEN |

Akita (fork markosg04/akita; upstream LayerZero-Labs/akita has no `akita-metal` crate):

| line | head | base | contents | state |
|---|---|---|---|---|
| `origin/main` (LayerZero) | 26bdbac79 (09-03) | | includes #460 transcript grinding (breaking), #464 SIS tables (breaking), #468 coefficient packing, #469 | canonical |
| `perf/metal-commit-eval-proof` | 0e52ebf17 (08-30) | main 2869b67bf | the D512 panels kernel, eval-proof acceleration, T28 commit shape; pinned by campaign Jolt | pushed to fork |
| `port/metal-latest` | 1e5515cd7 (09-01) | main 2d1ab310c (08-30) | Metal root commitment backend reconstructed on the new API, packed K16 root, packed openings, ring-switch kernel, direct range sumchecks, zero-suffix skip, telemetry | local only; 6 behind main incl. two breaking protocol changes |
| `perf/metal-commit*`, `perf/metal-eval-proof`, `quang/metal-field-kernels` | 08-18 to 08-20, 05-27 | | earlier checkpoints | superseded |

## 2. Facts that shape the decision

- Main's CPU is not faster. Built at 7d33a217c and run with `profile --backend optimized`:
  Fibonacci T25 27.68 s, T26 57.82 s (commit 26.2 s). The campaign line with the
  accumulator fix: 21.7 s and 46.2 s (commit 17.1 s). Main's
  `jolt-akita/src/trace_onehot/commit.rs:94` carries the same `matches!(D, 64 | 128 | 256)`
  gate that put D512 on the slow accumulator; main's K256 catalog is D512 at nv 40-41.
  The port line fixed the gate differently (`matches!(D, 64 | 128 | 256 | 512)` at
  `trace_onehot.rs:1247`), unmeasured; ours measures 16.6-19.1 ns per hot entry.
- The port line already contains the per-backend geometry we spent today deriving: CPU
  catalog D128 rank 3 at 2^19 positions, Metal catalog D512 rank 1. The verifier accepts
  either cataloged row. `specs/akita-metal-d128-rank3-root-floor.md` is then the Metal
  kernel that would let both backends share the D128 row.
- The port imported the kernels at feat 8a5f238f4 (08-30). Everything after is missing:
  the SHA-2 compact-rs1 fix (the port's 906c717bd rewrote the same file by 400 lines and
  may or may not cover the same defect), the CPU accumulator fix, the retained
  L1a/L1c/L6a source (saved as
  `benchmark-runs/akita-10mhz-studies/retained-l1a-l1c-l6a-on-09e649061.patch`), the
  `modular_benchmark` example and evaluator examples (the port has no examples directory),
  the specs and study ledgers.
- The two Jolt Metal implementations (#1733 and ours) both own
  `crates/jolt-kernels/src/metal/` and the `metal` feature. They serve different
  protocols: #1733 accelerates the Dory (BN254) prover, ours the packed Akita (fp128
  Solinas) prover. They cannot land in the same path.
- The Akita port sits behind two breaking upstream changes (#460 grinding, #464 SIS
  tables). Jolt main pins 4505404b5 (#460); the fork branch Jolt pins must be at or past
  that.

## 3. Target shape

Jolt `feat/akita-metal` (rewritten; current head tagged first):

1. base: `origin/main` after PR #1818 merges, or `origin/main` plus #1818's branch if it
   is still open when this starts (the port carries edits to `jolt-prover-legacy` that
   #1818 deletes; drop them).
2. the port line's Metal work rebased as a short series rather than 89 commits: kernel
   import, shared-field adaptation, Metal routes for commits/openings/ring switch,
   per-backend K256 catalogs, CI lanes. Location stays `crates/jolt-kernels/src/metal/`
   for now; the namespace split against #1733 is deferred by the user and is a rename,
   not a merge, when it comes.
3. campaign deltas re-applied: CPU accumulator fix onto `trace_onehot/commit.rs` (keep
   whichever of the two fixes measures faster at T25); SHA-2 fix verified by the
   registers evaluator at T28 rather than assumed; the retained L1a/L1c/L6a patch (expect
   conflicts in `registers_read_write/sequence.rs` and `bytecode_read_raf_address/`);
   `modular_benchmark` folded into the profile CLI as `--backend metal` the way #1733 did,
   or kept as an example; specs and ledgers.
4. Cargo pins the fork by git rev, never by path.

Akita fork `feat/akita-metal` (new name, same as Jolt's):

1. base: LayerZero `main` at 26bdbac79.
2. `port/metal-latest`'s 13 commits rebased; expect conflicts with #460 (transcript
   grinding touches the eval-proof transcript), #464 (SIS tables may re-plan the T28
   catalog: re-run the geometry query and confirm D512 rank 1 and D128 rank 3 rows
   survive), #468 (coefficient packing overlaps the packed-opening kernels).
3. pushed to the fork; a PR to LayerZero main is a separate decision.

Archive tags before any rewrite: Jolt `archive/akita-metal-v5-20260903` (09e649061 with
the retained patch committed on top), `archive/port-akita-metal-latest-20260901`
(2094ce34a), `archive/metal-prod-slim-20260817` (7f75dbf4c); Akita
`archive/metal-commit-eval-proof-20260830` (0e52ebf17),
`archive/port-metal-latest-20260901` (1e5515cd7). Then delete the superseded fork
branches (`perf/metal-commit*`, `perf/metal-eval-proof`) and prune the worktrees
`port-check`, `main-cpu-check`, `cpu-accumulate`, `akita-metal-latest`, `metal-slim`.

Phase 1 (Akita) done 2026-09-03: fork `feat/akita-metal` = 6817c72b86fefa55562efdc50841d7519f7043dd,
LayerZero main 26bdbac79 + 12 Metal commits (one of the port's 13 was already on main as #458).
The one real conflict was the direct-range sumcheck commit against #460 (grinded challenge
sampling threaded with `level` into Stage 1/2 and both Metal round loops) and #469 (the D512
direct-range kernel now emits `[q1..qd]`). Gates: workspace check, fmt, three CI clippy configs,
407 tests passed / 3 skipped for akita-metal + akita-prover, packed_onehot_commit bench runs.
Not verified there: end-to-end proofs through Jolt, CPU/Metal parity (the bench has no
checksum). Planner note: `find_schedule_with_root_constraint` / `RootCandidateConstraint` were
fork-only API and do not exist upstream; the unconstrained #464 planner at nv 41 picks D512
rank 1 with 7 levels and a 78,466-byte payload; D128 rank-3 admissibility under #464 is
unverified.

Phase 2 (Jolt) 2026-09-03: `unify/akita-metal` = c3a3e2768 (worktree `../unify`), 37 commits over
main: #1818 merged (17 conflicts, legacy crate and its byte-diff tests deleted, CI job reduced to
the akita lane, nextest override for the deleted binary removed, `MleAst` serde derive needed by
the shared-field bound), 13 port commits applied (two became empty, three superseded), Akita
pinned to fork rev 6817c72b8 with all four catalogs regenerated under #464 (protocol epoch 3;
K256 roots stay D128 on the CPU table and D512 on the Metal table). CPU-side gates green:
`akita_e2e` 7/7, jolt-akita 57/57, jolt-kernels 131/131, clippy host and host,zk. The `metal`
feature does not yet compile: 49 errors in jolt-kernels because main's optimized tier moved to
the BundleStore session-shared-rows model (#1734, 09-03) after the port branched; the port's
one-hot Metal impls were re-homed into `jolt-akita/src/trace_onehot/metal.rs` (unverified) and
its 2.1k-line RAM record model and CPU sparse coefficient-packing path were not carried. Phase 2b
(in progress): re-implement the Metal hooks against #1734, port the benchmark harness, Metal e2e
and T25 proofs.

Phase 2b (Metal against #1734) 2026-09-03: `unify/akita-metal` = b8334b5f6. The Metal hooks were
re-implemented over main's BundleStore model: the per-cycle RAM record model the Metal RAM
kernels consume moved into `jolt-kernels/src/metal/ram_records.rs` (21 files repointed),
instruction-RA initialization split restored, instruction-input Offloaded state and fast path
restored over `BundleStore`, registers val-evaluation Offloaded states restored, registers
read-write continuation from main's `CycleState`, RW-matrix closures generalized, nine new
lookup suffix tables (kinds 56-64) added to the MSL with the table count injected from
`LookupTableKind::COUNT`. Harness ported (`modular_benchmark` + ten evaluators). Gates: metal
check and clippy clean, host and host,zk clippy clean, jolt-kernels metal 128/128 (131 with
akita), jolt-akita metal 57/57, Metal e2e 4/4, CPU e2e 7/7; T22 Metal proofs verify on all
three workloads (Fibonacci 4.62 s, SHA-2 4.74 s, BTreeMap 3.27 s). Open defect: full Metal
proofs fail at T24 and above with `FinalOpeningVerificationFailed`, localized to the fork's
Stage-1 digit-range opening kernel at qualified trace sizes (nv >= 37); Metal PIOP over a CPU
commitment and Metal commitment with CPU openings both verify at T24. Being fixed in the Akita
fork branch (Phase 2c).

Phase 2c (opening defect) 2026-09-03: root cause in the Akita fork rebase. The #469 adaptation
moved the eq-factored round message to `[q_1..q_d]` in the shared `akita_direct_range_q_coefficients`
helper but the Stage-1 initial-partials kernel (`kernels/onehot.metal`,
`akita_fp128_direct_range_initial_partials`) still computed its coefficients inline and wrote
`q_0` into slot 0, so the first Stage-1 round message carried `q_0` where the verifier expected
`q_1`. Only qualified openings (nv >= 37, T24 and up) run the Metal opening, hence the size
threshold. Fixed in fork `feat/akita-metal` 3d748d499b7957f9b1adf4eb48a10a40c022e1d5 with two
permanent CPU-vs-Metal digit-range parity tests (basis 4 and 8, red before, green after);
Akita gates green (409 passed). Jolt unify re-pinned at 7323fc159; cold single-sample Metal
proofs verify: Fibonacci T24 9.07 s, T25 6.18 s; SHA-2 T25 7.51 s; BTreeMap T25 8.79 s. The
harness example needs `--features prover-fixtures,metal,profiling`.

## 4. Order and gates

1. Akita rebase (fork `feat/akita-metal`). Gate: `akita-metal` tests, the
   `packed_onehot_commit` bench parity checksum against the CPU commit, planner rows at
   nv 41 unchanged or re-derived.
2. Jolt rebase of the port series onto main, pin the fork rev. Gate: clippy in both
   feature modes, `akita_e2e` Metal proofs at T20 and T25, evaluator parity for every
   Solinas family.
3. Campaign deltas. Gate: CPU Fibonacci T25 stage 0 at or below 8.1 s; SHA-2 T28 registers
   evaluator checksum equal to CPU; the full T28 three-workload order-reversed matrix,
   scored against CPU references re-frozen on the unified binary.
4. Force-update `feat/akita-metal` with lease, push the fork branch, update
   `specs/akita-metal-share-state`-style pointers, archive and prune.

Expected cost: the Akita rebase and the Jolt rebase are each a day of conflict work with
the verification runs; the campaign deltas are half a day if the port's kernel adaptation
did not restructure the files the patch touches, longer if it did.

## 5. Verified and not

Verified today: branch inventory and distances; main's CPU speed at T25/T26; the D512 gate
on main; catalog geometries in the three schedule tables; the port's import base
(8a5f238f4) and Cargo path wiring; worktrees for both port branches exist at
`/Users/mgeorghiades/worktrees/jolt/port-check` and `.../akita-metal-latest`.

Port type-check: `cargo check -p jolt-prover --features prover-fixtures,metal --tests` in
`port-check` against the recreated Akita worktree passes with zero errors (2026-09-03 14:55).
Only a type-check; no build, test or proof was run on the port pair.

Not verified: whether the port pair proves anything (no proof has been run on it in this
session); whether #1733's Stage-1 booleanity anchor change interacts with our Stage 6a/6b
kernels; how the #464 SIS tables re-plan the T28 root; whether Andrew's team accepts the
namespace split.
