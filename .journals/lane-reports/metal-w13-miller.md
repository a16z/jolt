# Metal wave 13 — lane M13: commit-shape tier-2 Miller

## Verdict

**RETAIN, one cut (`lane/metal-w13-miller`, base b5a8e3878): the commit
tier-2 device Miller switches from `jk_miller_fly_indexed` to
`jk_miller_table` over the setup-owned prepared-G2 coefficients (4
pairs/thread, cap-32 pipeline).** In-pipeline @2^25 the commit Miller CB
mass drops **1.734 → 1.076 s (−38%)** at identical 14-CB shape; e2e ABBA
@2^24 **−0.30 s** (6.87/6.87 vs 7.19/7.15, pairs −0.32/−0.28) and @2^25
**−0.53 s** (12.47 vs 13.00 — the OFF arm reproduces the standing 2^25
record exactly; 12.47 is a new-record indication, lane window). **Modeled
@2^27: −1.44 CB-s device on an additive stage ⇒ ≈ −1.4..−1.5 s wall —
bar (≥1.0 s) passed.** Byte parity: proof-byte oracle 20/20 first pass;
GT fly == table == arkworks; metal suites 411/411; clippy host + kernels
metal-target clean. Kill switch `JOLT_METAL_MILLER_COMMIT_FLY=1` restores
the fly path.

## Roof repricing (the mandate) — why the fly kernel was the wrong shape

X9/S12 method: factorize achieved vs roof on production geometry
(65536-pair flushes — the post-W7 dispatch size; production row striping,
14 column segments, ascending rows; probe example
`miller_commit_shape.rs`, cap sweep via per-process
`JOLT_METAL_PAIRING_TG_CAP`).

Exact ALU counts from the shader bodies (Karatsuba tower: fq2 mul/sqr/
mul_by_fq = 3/2/2 Fq-mul; fq12_sqr 36, 034 43, line-pair fold 69/2 pairs;
G2 dbl 28, add 37; 64 ate iterations + 21 digits + 2 Frobenius):

| kernel | Fq-mul/pair | µs/pair @65k flush | Gmul/s | vs fly |
|---|---:|---:|---:|---:|
| fly_indexed (production, uncapped) | 8664 | 1.98 | 4.37 | — |
| table ppt=1 (cap 32) | 6357 | 1.53 | 3.66 | −23% |
| table ppt=2 | 4483 | 1.15 | 3.88 | −42% |
| **table ppt=4** | **3917** | **1.03** | 3.81 | **−48%** |
| table ppt=8 | 3634 | 1.66 | 2.19 | −16% |
| chain roof: fq12 sqr / mul | — | — | 3.26 / 3.38 | — |

Factor attribution:

- **Table-vs-fly split is the whole prize.** Every Miller kernel runs in
  the same spill-bound ~3.3–4.4 Gmul/s band as the register-resident
  fq12 chains — the fly kernel is *at/above* its own band (its ladder ALU
  overlaps the f-walk's spill stalls, per the W4-fly analysis), i.e.
  **no in-kernel headroom without changing what it computes**. The fly
  shape re-derives every line on device (the G2 double/add ladder) and
  pays an unshared squaring walk per pair: 8664 vs 3917 Fq-mul/pair =
  2.2×. The prepared table has been setup-owned (free per proof) since
  W5-T2 — the original reason fly won (per-proof prep cost) died then.
- **Spills:** the residual vs the bare-CIOS roof (X9: 11.30 Gmul/s) is
  the ≥192-u32 live Fq12 state — the W4-fly PRICED-SHUT receipt; not
  reopened (split ladder +12.7..18.2% stands).
- **Thread starvation:** dead at production flushes — scale curve flat
  past 32k pairs (fly 2.14→1.98, table 1.23→1.16 µs/pair 32k→65k). The
  W4 handoff's −24% was measured in the starved 8192-flush era;
  today's 65k flushes make the table strictly better (−48%).
- **TG packing:** table cap sweep @ppt4: 32 → 1.03, 64 → 1.04, 128 →
  1.09, uncapped → 1.11 µs/pair — the shipped cap-32 default is optimal.
  Fly cap sweep inert (≤4%). Divergence: nil (uniform digit schedule).
- 2^27 row count (131072 rows, 2.19 GB coeff table): table ppt4 1.16
  µs/pair (−41% vs fly) — gather locality costs ~0.13 µs/pair vs the
  16k-row case; conservative model uses 1.17 (the in-pipeline measure).

## What landed (one commit)

1. **Default `MillerInput::Table`** in the commit slot; the
   `miller_fly_commit` gate deleted (the table wins at every measured
   size); fly arm kept behind `JOLT_METAL_MILLER_COMMIT_FLY=1`.
2. **`MILLER_TABLE_SEG_PAIRS` 2 → 4** (ppt sweep above; 8 inverts —
   8192 threads starve the device).
3. **`flatten_prepared_coeffs` parallelized over steps** (contiguous
   per-step slices, byte-identical layout): 240 → 54 ms at the 2^27 row
   count — the only per-proof host cost of the table path (spanned
   `MetalCommit::miller_table_flatten`). The fly arm's per-proof
   `normalize_batch` of the G2 table leaves the default path.

Byte-parity mechanism: unchanged partition-invariance — per-thread
partial Miller products are exact under ANY pair partition (the ladder
distributivity the CPU/device split already relies on); per-column merge
order stays dispatch order. Table and fly arms both remain covered by
`metal_commit_matches_optimized` (default arm = table; the all-device
extreme now pins the fly kill switch).

## Numbers

- Solo probe (M5 Max, GPU-locked, min over warm passes): table above.
- In-pipeline CB trace @2^25 (diagnostic run, 12.47 s e2e): MillerTable
  **14 CBs / 1.076 s device** vs W7's fly receipt 1.734 s / 14 CBs →
  −38%; 1.17 µs/pair in-pipeline = solo rate (Miller stays additive, so
  solo transfers — R12's regime holds for the table kernel too).
- e2e sha2-chain, FrBind gate 255.6 µs: ABBA @2^24 A-B-B-A with 40 s
  cooldowns: **table 6.87/6.87 vs fly 7.19/7.15 (−0.30 s)**; @2^25 pair:
  **12.47 vs 13.00 (−0.53 s)**.
- Modeled @2^27: 1.73 M device pairs × (2.00 − 1.17) µs = **−1.44 CB-s**;
  st0 commit-Miller CB 3.46 → ~2.0 s. Wall transfer measured 1.4×/0.8× of
  the CB delta at 2^24/2^25 ⇒ **−1.4..−1.5 s wall** (send_wait 2.05 s
  relaxes 1:1 with device cuts per R12; not double-counted).
- RSS: **+2.19 GiB transient during st0 @2^27** (the flattened coeff
  table, freed at pass end; @2^25 +1.1 GiB). Peak RSS sits inside st0 ⇒
  expect the honest point ~72.4 → ~74.5 GiB; margin to the ~97 GiB storm
  regime stays >20 GiB. Priced alternative (setup-owned flatten) would
  also erase the 54 ms — sub-bar, parked below.

## Doors closed (receipts)

- **Fly in-kernel headroom: NONE** — 4.37 Gmul/s achieved vs 3.3–3.4
  chain band; the only lever is computing less (prepared lines). The
  W4-fly spill restructure stays priced shut.
- TG caps on fly_indexed at 65k flushes: inert (32/64/128/uncapped
  within 4%); the W4 "cap inverts in-pipeline" question is moot — the
  kernel leaves the default path.
- Table ppt=8: inverts (starved); uncapped table: +8% vs cap 32.

## Parked

- **Row-locality flush ordering:** sorting each flush's queue by row
  within column (byte-free — per-column GT product is order-invariant)
  could recover the 131k-row gather tax, ~0.1 µs/pair ≈ −0.17 CB-s
  @2^27. Sub-bar alone.
- **Setup-owned flattened table** (stride-fixed step-major or row-major):
  kills the 54 ms flatten + the 2.19 GiB transient at +2.19 GiB
  permanent RSS. Sub-bar (~0.05 s); layout surgery not justified.
- st8 reduce-shape MillerFly (15 CBs / 1.27 s @2^25) is a different
  regime (G2 side changes per round — no reusable prepared table);
  untouched, fenced by the fold-chain NO-GO.

## Discipline

- Timed 2^27: **0**. Timed e2e decision runs: ABBA ×4 @2^24 + pair ×2
  @2^25 (40–45 s cooldowns, FrBind 255.6 µs gate). One CB-trace
  diagnostic @2^25 (untimed). Probe example runs were solo-GPU
  attribution diagnostics under the GPU lock; all cargo under the wave-3
  cargo lock. No sibling worktrees or scratch touched; not pushed.
- Gates: metal suites **411/411**; proof-byte oracle
  (`jolt-prover --features prover-fixtures`) **20/20 first pass**;
  `clippy --all --features host -D warnings` + jolt-kernels metal-target
  clippy clean; fmt clean.
- KernelId::ALL unchanged (85 — no kernels added); commitment.rs 2508 →
  2507 lines (did not grow).
- New probe example `miller_commit_shape.rs` (jolt-kernels, bench-only):
  the commit-geometry repricing rig behind this lane's receipts — flag
  for PR-handoff audit alongside the existing `miller_microbench`
  (X9-rig precedent).
