# Akita Metal: unifying the campaign line, the port line, and upstream

Status: proposal, 2026-09-03. Nothing here has been executed except the inventory,
measurements and the port type-check recorded in section 5.

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
   per-backend K256 catalogs, CI lanes. Location: `crates/jolt-kernels/src/metal_akita/`
   (or `metal/akita/` if #1733 agrees to nest), gated by `metal` and `akita` together, so
   #1733's BN254 tier keeps `metal/`. Decide this with Andrew before the rebase; it is a
   rename, not a merge.
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
