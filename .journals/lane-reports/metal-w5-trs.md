# Lane T (wave 5) — TRS PC-cache verify + public-matrix regen pricing

Desk sweep + 1 CPU microbench (no GPU time used). Base: lane/metal-w5-trs @ 2e1efd307.
Probe: scratch test, run once (2 in-process rounds), deleted — branch is clean, nothing committed.

## Receipt 1 — TRS flat-PC-cache on the Metal prove path: **GO, −0.46 s @2^27**

**Verdict: GO.** The pairing *arithmetic* is already fully prepared-exploited on the
prove path; the *preparation* of the tier-2 G2 table is not — it is recomputed inside
the prove wall every proof: `DoryTier2Prep::new` (jolt-dory/src/tier2.rs:49) prepares
`g2_vec[..max_rows]` with `max_rows = one_hot_rows.max(windows_total)` = **2^17**
@2^27 (log_k_chunk 8 → total_vars 35 → 2^18 columns, 512 windows × 256 k), called from
`commit_streaming_metal` (jolt-kernels/src/metal/commitment.rs:388) at the top of st0.

**Microbench (M5, release, rayon all-cores, 2 rounds):**
- Per-point `into_affine().into::<G2Prepared>()` over 2^17 points (exact `DoryTier2Prep::new` shape): **468 / 500 ms** — matches the record trace's `prepare_tier2 0.47 s` exactly.
- Split: batch-normalize 2^17 G2 = **6 ms**; the prepare (87 `EllCoeff` Miller line steps/point) = **451–516 ms**. ⇒ swapping per-point `into_affine` for `normalize_batch` in-wall saves ~nothing; the Miller precompute IS the cost. Hoisting is the only fix.
- Join co-runner `begin_one_hot_column_major_stream` (2^18 G1 `into_affine`): **3 ms**; miller_fly G2 normalize input (2^17): **6 ms**. ⇒ the st0 preamble `rayon::join` wall ≈ the prep leg alone; hoisting prep saves ≈ **0.46 s** of prove wall.
- Memory: `ell_coeffs = 87`, **16 704 B/prepared point** → 2^17 table = **2.09 GiB**, full 2^18 `g2_vec` = **4.18 GiB**.

**Prove-path pairing/preparation sweep (every site, mechanism one-liners):**

| site | prep behavior during prove | mass @2^27 |
|---|---|---|
| st0 tier-2 absorbs (`Tier2Accumulator::absorb`, finishes, CPU miller share) | pair against `DoryTier2Prep` — prepared once/proof, borrowed coeffs, shared ladder (W10b) | exploited ✓ |
| **st0 `DoryTier2Prep::new`** | **re-prepares 2^17 setup G2 EVERY proof, in-wall** | **0.47 s ← the finding** |
| st0 miller_fly G2 affine input | `normalize_batch` 2^17/proof | 6 ms |
| st0 G1 affine bases | per-point affine 2^18/proof (join co-runner) | 3 ms |
| st8 VMV preamble (evaluation_proof.rs:144,148) | 2 single `E::pair` on fresh points | ≈ the measured 0.035 s |
| st8 reduce rounds 1–9 (device) | GPU miller_fly computes lines on-device — zero CPU preparation | n/a |
| st8 reduce host fallback (`multi_pair_g*_setup`) | would re-prepare setup slice per call (global cache unprimed) — but record trace shows all 18 rounds served (device + tail); exposure 0 | 0 |
| st8 FastTail (host_tail.rs, n=512→2) | setup-G2 prefix prepared **once**/proof (256 pts ≈ 0.9 ms); live v2 re-prepared per round = fresh values, uncacheable by definition | ms |
| st8 `combine_hints` | G1 MSM only, no pairings (device hook) | n/a |

**Fix pricing:**
- (a) *Per-proof local prepared tables passed explicitly* — **is the status quo** (`DoryTier2Prep` is exactly that, threaded through st0 and st8-fallback). No further gain available from (a); the 0.47 s stays in-wall.
- (b) *Setup-owned prepared table* (not a global cache): compute the G2 prepared vector inside `setup_prover` (or a `OnceLock` on `DoryProverSetup`), have `prepare_tier2` return/borrow a prefix. Kills the footgun structurally — the table lives on one setup object = one URS, so prefix-matching is sound by construction (the historical bug was a process-global cache prefix-matched across *different-size URS values*, scheme.rs:114). Cost: +0.95 s in `setup_prover` (already 48.9 s, outside wall) and +4.18 GiB RSS if the full 2^18 `g2_vec` is prepared eagerly (setup doesn't know `max_rows`); +2.09 GiB if populated lazily at first `prepare_tier2` (helps only multi-proof processes — for the single-prove benchmark wall the eager variant is the one that counts). Peak RSS 72.2 → ~76.4 GiB on the 128 GiB box: fine.
- **Expected prove-wall cut @2^27: ≈ 0.46 s (0.47 join wall → ~10 ms residue), −0.7% on 63.88 s.** Value-exact (same prepared coefficients, same arithmetic), no protocol surface, kill switch trivial (fall back to in-prove prep).

**Out-of-wall note for parent:** `setup_verifier` 15.5 s = 55 CPU `multi_pair_g*_setup`
calls that each re-normalize + re-prepare the setup side because the global cache is
deliberately unprimed. The same setup-owned table (option b) would serve these too —
big preprocessing win, but outside the campaign gate metric.

## Receipt 2 — on-GPU public-matrix regen from seed: **PRICED NO-GO**

**Verdict: NO-GO — the LATTICE precondition (GB-scale public matrix streamed from DRAM
per proof) does not exist in Metal Dory; total public base data is 32 MiB and resident.**

What the tier-1 commit actually streams @2^27:
- **Public/base data:** G1 affine base table `g1_vec[..2^18]` × 64 B = **16 MiB**, built once per proof (3 ms), mapped zero-copy (`ctx.wrap_slice`, unified memory) and resident for the whole pass; plus the miller-fly G2 affine input 2^17 × 128 B = **16 MiB** (6 ms). That is the entire public-data footprint: **32 MiB, ~9 ms of st0's 16.2 s (~0.06%), ~0 sustained DRAM bandwidth** — gather reads land in a 16 MiB table that is SLC/L2-resident, so even ~30 one-hot columns × 2^27 gathers generate index-stream traffic (4 B/element, *witness* data), not base-table DRAM traffic.
- There is no per-superchunk re-streaming of public data to eliminate; the streamed mass is witness indices and witness-derived segments.

Regen-from-seed pricing, per G1 base point on-GPU: hash-to-curve ≈ 2 field sqrts + cleanup (≥ thousands of Fq muls) or PRG-scalar × fixed-base mul ≈ 250+ group ops — versus a ~64 B cache-resident read. The trade is inverted by 3–4 orders of magnitude. And the protocol blocker: dory-pcs's URS is **OsRng-generated and persisted to disk** (jolt-dory/src/urs_lock.rs; scheme.rs:109 `new_from_urs`) — random, not seed-derived. Seed-regen requires switching to a seeded-PRG transparent URS = protocol/setup change that invalidates every persisted `dory_N.urs` and the legacy byte-parity story, for a ≤9 ms/0.06% prize. No wave-6 lane.

(The prepared-G2 line tables are the other "public data" candidate — covered by receipt 1: don't regen them, hoist them.)

## Coordination

- Lane R report (`metal-w5-reattr.md`) not yet present; nothing here blocks on the st0 split — receipt-1 pricing is anchored to the trace's own `prepare_tier2 0.47 s` span, which the probe reproduces within noise.
- Bench discipline: 1 timed CPU bench (2 in-process rounds), cargo under the wave-3 lock; GPU lock never taken; no e2e runs. Probe deleted; `lane/metal-w5-trs` has zero diff vs base.

## Implementation (lane T2) — setup-owned prepared-G2 table: **DONE**

**Verdict: LANDED @ 13b01608b (lane/metal-w5-trs).** Mechanism: the Miller-prepared
G2 table now lives on `DoryProverSetup` (built eagerly in `setup_prover`, full
`g2_vec`); `DoryTier2Prep::new` borrows a prefix via `Arc` instead of re-preparing
2^17 points per proof. One setup object = one URS ⇒ prefix-match sound by
construction; dory-pcs's global cache untouched/unprimed. Kill switch:
`JOLT_DORY_SETUP_PREP=0` → empty setup table → per-proof prep fallback (old path).
`setup_verifier` refactored onto the raw SRS so it doesn't pay the table.

**Numbers (sha2-chain, metal backend; window degraded — FrBind @2^20 = 510 µs vs
350 µs bar, sibling lanes R/B live; treat walls as same-window-relative only):**
- **st0 preamble join (the prep span): 304 ms → 2.4 ms @2^25** (16 ms → 1.4 ms
  @2^21). The in-wall cost is gone; receipt's −0.46 s @2^27 stands (span scales
  2× with the 2^18-row table).
- **Byte-identity: proofs bit-identical on/off** at 2^21 (hash caa0419d5f493dbb)
  and 2^25 (hash ffc4ac3fccd2acb3), probe hashes of the bincode proof (deleted).
- **setup_prover delta: +0.26 s @2^25** (21.27 → 21.53 s; table = 2^17 pts there).
  @2^27 expect ≈ +0.95 s (2^18 pts, receipt microbench).
- **Peak RSS @2^25: on 26.37 GiB / off 25.19 GiB → +1.18 GiB.** NOT fully
  neutral: eager table prepares full g2_vec (2^17 rows @2^25 = 2.09 GiB) but the
  removed per-proof transient was only max_rows = 2^16 (1.05 GiB). @2^27 expect
  ≈ +2.1 GiB peak (4.18 GiB table − 2.09 GiB transient): 72.2 → ~74.3 GiB. Fine
  on the 128 GiB box.
- Walls @2^25: off 19.78 s; on 28.09 s then 17.91 s on the disagreement re-run —
  ±8 s ambient swing confirms walls are noise here; the span probe is the evidence.
- Suites: jolt-kernels+jolt-dory+jolt-eval (metal) 404/404 green; clippy --all
  --features host -D warnings clean. Permanent coverage added: kill-switch
  fallback arm in `prepared_finishes_match_unprepared` (streaming.rs). All
  probes deleted; diff = 4 files, +97/−27, no new abstractions beyond the
  `PreparedG2Table` alias.

### Join-arm completeness follow-up (T2b) — **DONE @ 463475d38**

Cross-check vs lane R's "prepare_tier2 0.47 ∥ base_affine_cache 0.47": the G1
affine-base arm now also lives on `DoryProverSetup` (eager affine view of the
full `g1_vec` at setup_prover; `begin_one_hot_column_major_stream` copies the
prefix, `scalar_affine_bases` borrows it). Same single kill switch —
`JOLT_DORY_SETUP_PREP=0` restores the full old per-proof path for both arms.

- **Join span @2^25: 278 ms (off) → 1.2 ms (on)**; @2^21 14.4 → 0.5 ms. Both
  arms individually ~0 when on (prep 0.0 ms, base 1.1 ms = the 8 MiB memcpy).
- **Byte-identity on/off re-confirmed** @2^21 (caa0419d5f493dbb) and @2^25
  (ffc4ac3fccd2acb3) — hashes unchanged from the G2-only commit, so the G1
  hoist is value-exact too.
- **RSS @2^25: on 26.39 / off 25.41 GiB → +0.98 GiB** — unchanged from the
  G2-only pair (+1.18) within noise; the G1 table is 2^17×64 B = 8 MiB @2^25
  (16 MiB @2^27), negligible as priced. It replaces an equal-size per-proof
  transient (the stream context Vec still exists; the table adds one copy).
- **Attribution note for R:** the off-arm probes show base-arm 278 ms ∥
  prep-arm 278 ms @2^25, but with prep hoisted (G2-only build) the base arm
  measured 2.4 ms solo — the 0.47 s `base_affine_cache` span @2^27 is mostly
  contention-dilation under the co-running Miller prep, not intrinsic
  conversion cost (URS G1 points convert in ~ms). The G2 hoist alone had
  likely already collapsed most of the join; this commit removes the residue
  and makes the preamble structurally free either way.
- Suites 404/404 green (kernels+dory+eval, metal); clippy --all --features
  host -D warnings clean; fallback parity test extended to the stripped G1
  table (base equality + finishes). Probes deleted; diff 3 files, +53/−16.

## Slimming (lane RSS, wave 7) — eager G2 table sized to the consumer bound: **DONE @ 124d89578**

**Verdict: LANDED (lane/metal-w7-rss @ 124d89578, base a4028227c).** The eager
table now prepares only `g2_vec[..2^floor(max_num_vars/2)]` instead of the full
even-padded SRS (`2^ceil`) — half the table whenever max_num_vars is odd, which
every benchmark shape is (35 @2^27, 33 @2^25). Diff: 2 files, +31/−4 (a bound
computation in `setup_prover` + a `rows` param on `prepare_g2_table` + a sizing
test).

**Consumer-bound receipt (every prepared-table consumer audited):**
- All access to the setup table flows through `DoryTier2Prep::new` — no other
  reader of `setup.1` exists. Consumers: metal `commit_streaming_metal`
  requests `one_hot_rows.max(windows_total)` = 2^(log_t+log_k_chunk −
  ceil(total_vars/2)) (commitment.rs:410); optimized-tier finish requests the
  max over columns (windows·k one-hot, windows increment)
  (optimized/commitment.rs:197). Both ≤ 2^floor(total_vars/2) — the balanced
  layout's row count (`MatrixDimensions::balanced`: row_vars = total − ceil).
- miller_fly reads raw `g2_vec[..prep.prepared().len()]` — bounded by the same
  request. hint_hook/`combine_hints` = G1 only. FastTail (host_tail.rs)
  prepares its own 256-pt prefix per proof; st8 host fallback + `setup_verifier`
  use `multi_pair_g*_setup` on the raw SRS — none touch the table.
- Verified at real geometry (scratch probe, deleted): nv 33 → g2_vec 2^17,
  table 2^16 (65 536 × 16 768 B = 1.02 GiB); @2^27 (nv 35) table 2^17 = 2.05
  GiB vs 4.10 unsliced. A larger request degrades gracefully to per-pass prep
  (`DoryTier2Prep::new`'s existing len-check → the kill-switch fallback arm),
  so an even-nv workload (bound = full table) or any oversize ask is
  correctness-safe.

**Byte-identity:** proof hashes on (sliced) / off (`JOLT_DORY_SETUP_PREP=0`)
identical AND equal to lane T's recorded unsliced-table hashes at both scales —
2^21 `caa0419d5f493dbb`, 2^25 `ffc4ac3fccd2acb3` (same DefaultHasher probe,
deleted). Sliced table ≡ unsliced ≡ per-proof path, bit for bit.

**RSS @2^25 (sha2-chain, metal, /usr/bin/time -l, one timed pair):** sliced ON
27.40 GiB / OFF 25.37 GiB. Table arithmetic is deterministic: sliced saves
exactly 1.02 GiB of resident table @2^25 and 2.05 GiB @2^27 (wave-6 honest
point 78.78 → ≈ 76.7 modeled). Honesty note: the measured ON−OFF (+2.03) runs
~1 GiB above the sliced-table model (+1.02) — same-window single-run RSS noise
(lane T's unsliced pair measured +1.18 against a 2.09 GiB table; sibling GPU
lane active tonight); the probe-verified table size is the reliable number.

**No prep re-entry:** `prepare_tier2` span 13 µs, `base_affine_cache` 17 µs
@2^21 ON (chrome trace) — the default path still serves from the setup table
(requests land exactly at the bound: 2^14 @2^21, 2^16 @2^25). ON wall @2^25
17.00 s vs OFF 21.24 s.

**Suites:** jolt-kernels+jolt-dory+jolt-eval (metal) 406/406 green (adds
`setup_table_sized_to_consumer_bound` — pins table = floor-bound rows, odd and
even nv); clippy --all --features host -D warnings clean. Probes deleted;
worktree `.worktrees/metal-w7-rss` ready for merge + cleanup.
