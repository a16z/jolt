# Metal wave 15 — lane M15: Miller table TILING (bounded residency)

## Verdict

**RETAIN, one cut (`lane/metal-w15-tiling` @ 1e5e3d7f2, base 2b0959e30):
shape (a) tiled streaming wins — the commit tier-2 table arm now gathers
each flush's referenced rows into a recycled per-dispatch tile (~78 MB)
instead of wiring the row-scaled whole table, and becomes the DEFAULT at
every scale (the w13 `MILLER_TABLE_MAX_ROWS` gate and the whole-table
flatten are deleted).** Kernel receipt at the 2^27 row-count geometry:
tiled **1.03 µs/pair** vs whole-table 1.16 vs fly 1.93 — the dense tile
erases the 131k-row gather tax on top of capturing w13's 2.2×-ALU win.
E2e @2^25 same-window A-B: **11.89 vs 12.40 s fly (−0.51 s)**, peak RSS
22.99 vs 24.82 GiB (no residency addition — bounded ≤ ~160 MB by
construction). In-pipeline @2^25: MillerTable **14 CBs / 1.057 s** device
(w13 whole-table 1.076, fly 1.734 → −39%). **Modeled @2^27:
−1.38..−1.56 CB-s additive ⇒ ≈ −1.1..−1.5 s wall — bar (≥1.0 s) passed
on the model. THE DECIDER IS THE ORCHESTRATOR'S 2^27 KILL-SWITCH ABBA
(`JOLT_METAL_MILLER_TILING=0` restores the fly arm) — no wall claim at
flagship scale is made from these small-scale receipts (the w13
scale-transfer lesson is the premise of this lane).** Byte parity: bit-
equal partials pinned by a new unit test; metal suites 414/414; proof-byte
ratchet 20/20 first pass; clippy host + kernels-metal clean.

## Mechanism (shape (a) — why it deletes the w13 regression)

The w13 whole table is `steps(87) × rows × 192 B` — 2.19 GiB at 2^27's
131072 rows — wired for the entire st0 pass; hazard-tracked residency of
that one buffer stretched co-running CBs +2.59 s at the w13 gate. But a
65536-pair flush only ever references ~flush/columns ≈ 4681 unique rows
(14 column segments share the same window rows). The tile
(`MillerTile::build`, miller.rs): sort+dedup the batch's row indices,
remap each pair to its tile-local row, gather ONLY those rows step-major
(same layout contract at width `n_rows = unique`), dispatch
`jk_miller_table` unchanged. Kernel params take the tile width; the
shader's `(step·n_rows + row)·192 B` addressing is layout-equivalent.

- **Residency = O(flush), scale-invariant:** 87 × 4681 × 192 B = **78 MB
  per tile**, ≤ 2 live (one in flight + one building, recycled through
  `MillerLane::tiles`) ⇒ **≤ ~160 MB transient at ANY scale** vs 2.19 GiB
  @2^27 / 1.09 GiB @2^25 / 274 MB @2^24 whole-table. Degenerate bound:
  unique ≤ batch pairs ⇒ tile ≤ 65536 rows = 1.07 GiB (single-column
  geometry only; production is flush/columns).
- **Host cost:** build 2.5 ms/flush (sort+remap+87-step parallel gather;
  measured in-probe), ~27 flushes @2^27 ≈ 68 ms total — lands on the
  tier-2 lane, which has 3.14 s of recv_wait slack (R12); ≈ the deleted
  54 ms upfront flatten. Zero-copy wrap (malloc-large ⇒ page-aligned).
- **Locality bonus:** tile rows are dense 0..4681, so each step's loads
  are one contiguous ~900 KB stripe instead of scattered runs across the
  131k-row span — the 1.16 → 1.03 µs/pair recovery at 2^27 geometry
  (w13's parked "row-locality flush ordering" door falls out for free).
- **Byte parity:** identical coefficient VALUES at identical pair
  positions, identical seg_starts/folds/merge order — only addresses
  change, no fq regrouping. Per-thread partials are BIT-equal to the
  whole-table dispatch (`miller_tile_partials_match_whole_table`, incl.
  recycled-backing rebuild). CPU-recovery path keeps table-global indices
  (`batch.row_indices`); only the device sees the remap.

## Shape comparison (the mandate's (a)/(b)/(c))

- **(a) tiled streaming: SHIPPED** — receipts above; nothing left on the
  table for (b)/(c) to recover.
- **(b) on-device line prep:** dead on price — prep is setup-owned and
  FREE on host (W5-T2); on-device prep would re-spend the G2 ladder ALU
  (~28 mul/dbl-step, the very cost the table kills) to save 78 MB of
  bytes the unified-memory GPU reads at DRAM rate anyway. Tile build is
  2.5 ms host on a slack lane; no device-side version beats that.
- **(c) hybrid split: subsumed** — bounded budget was the only reason to
  fly a fraction; the tile IS the bounded budget at 100% table share.

## Numbers

- Probe (solo GPU-locked, `miller_commit_shape` + tiled arm, 65536-pair
  production flush, cap 32, ppt 4):

  | geometry | fly | whole table | **tiled** | tile |
  |---|---:|---:|---:|---|
  | 131072 rows (2^27) | 1.93 µs/pair | 1.16 | **1.03** | 4681 rows / 78 MB / build 2.5 ms |
  | 16384 rows (2^24) | 1.91 | 1.03 | **1.02** | 4681 rows / 78 MB / build 2.4 ms |

- E2e sha2-chain (window: FrBind-class 2^20 bind 220 µs, dispatch RT
  103 µs — record-class, fresh):
  - @2^25 A-B, 50 s cooldown: **tiled 11.89 s / fly (`TILING=0`) 12.40 s
    = −0.51 s** (reproduces w13's table−fly −0.53 with bounded
    residency; trunk default @2^25 is fly — rows 2^16 > the old gate).
    Untimed diagnostic repeat: 11.75 s. Standing 2^25 record 12.55 —
    record-class indication, lane window, not a cert claim.
  - @2^24 ABBA A-B-B-A, 45 s cooldowns: 6.61/6.75 vs 6.71/6.71 =
    **−0.03 s (inside window noise)** — @2^24 on the w14 trunk the
    Miller mass no longer transfers to wall (2^24 window compressed;
    R12's additivity receipts are @2^25/2^27). Not decision-bearing.
- In-pipeline CB trace @2^25 (one untimed diagnostic, 11.75 s run):
  MillerTable **14 CBs / 1.057 s device**, ~66.9k pairs/CB @ 80.5 ms =
  1.20 µs/pair in-pipeline — matches w13's whole-table 1.076 s / 1.17
  µs/pair; fly receipt 1.734 s ⇒ **−0.68 CB-s @2^25**.
- RSS @2^25: tiled 22.99 GiB vs fly 24.82 GiB (timed pair); tiled
  diagnostic 25.45 — run variance ±2 GiB dwarfs the ≤160 MB tile bound,
  i.e. **no measurable residency addition** (vs w13 table's +1.1 GiB
  @2^25 / +2.19 GiB @2^27).

## Modeled @2^27 (NOT a wall claim — gate ABBA decides)

1.73 M device pairs × (1.93 − 1.03) µs = **−1.56 CB-s** solo-rate;
conservative in-pipeline rates (2.00 − 1.20) give **−1.38 CB-s**. Miller
is additive @2^27 (R12: co-run ≈ serial sum), tile cost ~68 ms on lane
slack, residency mechanism of the w13 +2.59 s regression deleted by
construction (≤160 MB transient, 0.2% of the 71 GiB working set).
Wall transfer 0.8–1.0× (w13 @2^25 / R12 send_wait 1:1) ⇒
**≈ −1.1..−1.5 s wall @2^27**; st0 commit-Miller CB ~3.4 → ~1.9 s.
Risks the ABBA must exclude: per-dispatch buffer churn (new MTLBuffer
wraps per flush ×27 — µs-scale each), tier-2-lane tile builds colliding
with decode spikes at full scale.

## What landed (one commit, 1e5e3d7f2)

1. `MillerTile` (miller.rs): recycled tile build — sort/dedup/remap +
   step-parallel row-subset gather; `MetalCommit::miller_tile` span
   carries `rows` per dispatch.
2. Commit slot (commitment.rs): table arm = tiled at every scale;
   `MILLER_TABLE_MAX_ROWS`, the whole-table flatten +
   `miller_table_flatten` span, and `MillerSource` (collapsed to
   `MillerLane::fly_qs`) deleted. `InFlightMiller` owns its tile
   (wrapped-buffer backing) and recycles it at settle.
   `JOLT_METAL_MILLER_TILING=0` → fly arm at all scales (the pre-W15
   flagship default); `JOLT_METAL_MILLER_COMMIT_FLY=1/0` still forces.
3. Tests: `miller_tile_partials_match_whole_table` (bit-equal partials,
   sparse + full-coverage batches on recycled backing);
   `metal_commit_matches_optimized` default arm now exercises tiled
   dispatches at flush=8 depth. Probe rig grew a tiled arm
   (`miller_commit_shape`, bench-only).

## Doors closed / notes

- On-device line prep (b) and hybrid split (c): closed on price, above.
- w13's parked "row-locality flush ordering" (~−0.17 CB-s): absorbed —
  the tile's dense renumbering IS the locality fix.
- w13's parked "setup-owned flattened table" (+2.19 GiB permanent RSS):
  dead — tiling makes any whole-table layout obsolete.
- @2^24 wall delta is now ~0 on the w14 trunk (was −0.30 on w13's): the
  2^24 window no longer prices Miller as additive. Kernel-level win
  unchanged; scales ≥2^25 carry the wall effect.
- Pre-existing (not this lane): `clippy --features metal,bench-utils`
  fails on w14's `registers_read_write.rs:1559` unfulfilled `expect` in
  the bench mod; campaign gates (host / metal) don't cover that combo.

## Discipline

- Timed: @2^24 ABBA ×4 (45 s cooldowns) + @2^25 A-B ×2 (50 s,
  `/usr/bin/time -l` RSS). **Timed 2^27: 0.** Untimed diagnostics: probe
  ×2 (131072/16384 rows), microbench window gate ×1, CB-trace @2^25 ×1.
  The sanctioned single 2^27 instrumented profile was NOT spent: tile
  bytes are flush-bounded ⇒ scale-invariant by construction, so full-
  scale residency has no open parameter a profile would pin — the wall
  question belongs to the gate ABBA anyway. All cargo under the wave-3
  cargo lock; every GPU run under the GPU lock, one at a time; FrBind-
  class window gate before timed pairs (220 µs). No sibling worktrees or
  scratch touched; not pushed.
- Gates: metal suites **414/414** (413 + new tile test); proof-byte
  ratchet **20/20 first pass**; `clippy --all --features host` +
  jolt-kernels metal-target clippy clean; fmt clean; pre-commit hooks
  green.
- KernelId::ALL **88 unchanged** (no kernels added; the tile reuses
  `jk_miller_table` verbatim). commitment.rs 2518 → **2518** (cap held;
  scale gate + whole-table arm deleted to fund the tile plumbing);
  miller.rs 1304 → 1433 (tile + test).
