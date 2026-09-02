# Metal wave 12 — lane S12: st5 scan kernels (X9-method roof repricing)

## Verdict

**GO — roof repriced, dominant factor cut. Commit `2689cd009` on
`lane/metal-w12-st5scan` (base 440a7d07c).** The scan kernels were NOT at
roof: on production sha2-chain rows the phase scan ran **5× above its
compute+loads floor**, and the binding factor was `jk_simd_scatter3`'s
32-source Fr shuffle-reduce loop — not memory, not ALU, not dispatch.
Replacing the flush with a bitonic key-sort + segmented scan (+ dropping
the uniform-butterfly path, + 512→4096 simdgroups, + two-level RAF reduce):

- **Phase scan @2^24 real rows: 49.3 → 10.2 ms scan-only (−79%).**
- **Suffix scan @2^24 real rows: 28.1 → 6.1 ms (−78%).**
- **e2e @2^25 sha2-chain, same-window: scan CBs 1.736 → 0.535 s (−69%);
  kill-switch ABBA walls OFF 13.69/13.88 vs ON 12.85/12.91 = −0.91 s
  (pairs −0.84/−0.97).**
- **Proof bytes identical** across arms @2^21 (`0e93560b…d6d1`) and @2^22
  (`48ec537f…42c9`) — the same hashes as the w10 receipts; all runs verify.
- **Modeled @2^27: −6.8 s** on the 9.83 s scan mass (kill-switch ratio),
  ≈ **−7.5 s vs true trunk** (the OFF arm already carries the 4096-sg
  schedule; trunk scatter@512 is ~29% slower than scatter@4096). Bar
  (≥1.0 s) exceeded ~7×. st5 model 12.74 → ~5.5.

Kill switches: `JOLT_IRR_PHASE_SCAN_SCATTER=1` (w5 phase body) ·
`JOLT_IRR_SUFFIX_SCAN_GROUPED=1` (w5 suffix body) ·
`JOLT_IRR_PHASE_SCAN_SGS=512` (pre-w12 schedule). Existing EAGER/LEGACY
arms untouched (legacy suffix pinned to its original 512-group schedule).
`KernelId::ALL` 83 → **85**.

## Roof factorization (mandate item 1) — receipts

All @2^24 REAL sha2-chain rows (`JOLT_IRR_DUMP_ROWS` from a scale-24 e2e,
402 MB), production P1 shape (s=112, condense), 512-sg w5 schedule, CB GPU
windows, gpu-locked, FrBind 248.8 µs. Harness: `jolt-eval --bench irr_roof`
(permanent attribution rig) + two in-bench ablation variants.

| probe | ms | verdict |
|---|---:|---|
| production scan-only (w5 kernel) | 49.3 | baseline (P8 42.8, P15 39.6) |
| loads floor (identical traffic, no ALU) | 2.3 | **memory DEAD** (~770 GB/s effective) |
| quiet floor (loads + identical field math, no emit) | 9.75 | compute+loads roof at 512 sgs |
| fr_mont_mul chain @16k threads | 8.32 Gmul/s | vs 11.64 saturated ⇒ **starvation ≤1.4×, not the gap** |
| no-RMW ablation (reduce loops kept, register fold) | 49.1 | **device RMWs FREE** |
| no-loop2 ablation (detect + RMW kept, reduce loops gone) | 17.4 | **scatter3's Fr shuffle loop = 31.9 ms = 65% of the kernel** |

Row-structure receipts (host analysis of the dump, 512-sg schedule):
sha2-chain has **80% flush-fired tile-iterations** (`simd_any` couples
lanes: per-lane repeat ~28%, so ~any lane changes every iteration),
23–29/32 lanes flushing per event, d≈7.6/tile (P0-7) → 15–18 (P8-15),
18.9% uniform tiles. The w5 run-length win was fib-shaped (85% repeat,
d≈4.2); on sha2 the held state barely fires and the kernel degenerated to
~eager. Suffix twin: 28.1 ms scan vs 3.7 quiet / 3.6 loads floor — **87%
emission machinery**, gathers free at 2048 sgs.

Chain occupancy curve (Fr twin of X9's fq roof): 8.32 @16k → 10.41 @65k →
11.44 @131k → 11.64 Gmul/s @524k threads. 11.6 G saturated matches X9's
11.63 fq roof.

## The cut (mandate item 2) — mechanism

1. **`jk_flush_sorted3`**: pack `(key<<5)|lane` (key ≤ 511), 15-exchange
   bitonic sort across the simdgroup, ONE vec4-shuffle gather of each
   lane's sorted-slot 3×Fr, then a 5-step segmented inclusive scan over the
   now-contiguous equal-key runs; segment tails RMW the per-key totals.
   Replaces scatter3's 2×32 serial source turns (64 iterations, ~600 limb
   shuffles + ~80 predicated fr_adds per event) with ~15 u32 exchanges +
   ~150 vec4-shuffle limb ops + 15 fr_adds. P1 49.3 → 19.0 ms at 512 sgs.
2. **Uniform-butterfly path deleted** (phase): a uniform run costs 3
   fr_adds/row in the held state vs 6 butterflies (30 Fr shuffle+add
   steps) per tile; run-length + sorted flush strictly dominates.
   19.0 → 16.35 ms. Whole branch gone — the kernel is now one path.
3. **Schedule 512 → 4096 simdgroups** (`JOLT_IRR_PHASE_SCAN_SGS`
   restores): the occupancy knee from the chain curve. 16.35 → 10.2 ms.
   8192 is flat (10.05) with a growing reduce tax — rejected.
4. **Two-level RAF reduce**: 4096 partials rows → 32 → 1 (the single
   1536-thread pass over 4096 rows cost 1.67 ms/CB, ~0.21 s @2^27).
   Suffix reduce stays single-level (0.43 ms @2^24, ~0.05 s @2^27 — below
   noise).
5. **`jk_irr_suffix_scan` sorted**: chunk keys are suffix-invariant ⇒ sort
   once per tile, each suffix pays one gather + one 5-step scan
   (zero-valued lanes contribute exact zeros — the w5 precedent). Uniform
   tiles skip the sort and keep the single butterfly. 28.1 → 6.1 ms;
   replaces BOTH the w5 grouped path and the eager scatter arm.

Exactness: every change is an fr_add regrouping or a schedule change; Fr
sums are exact so cells (and proofs) are byte-identical — the standing
campaign argument, now backed by the 4-arm oracle fixtures (random +
skewed k=2 at 2^22+2^24), 10/10 slot round-loop parity tests, and the
cross-arm proof hashes.

## Doors closed (with receipts)

| door | verdict |
|---|---|
| grouped per-distinct-key masked butterflies (w5-style) as the flush | **LOSES at production entropy**: P1 +20%, P8 +155%, P15 +222% — d≈8–18 on sha2; w5's "loses above d≈8" confirmed at scale; superseded by sort+scan |
| vec4-packing scatter3's shuffles (minimal-diff fix) | nil (±0.5%) — the serial 32-turn structure, not shuffle width, was the cost (vec4 does pay inside the parallel sort+scan) |
| device-RMW batching / TG-memory cells | DEAD — no-RMW ablation ≡ production; RMWs are latency-hidden |
| TG width on the sorted kernel | flat 32–256 (the old kernel's width-64 win was the scatter loop's, now moot) |
| sgs > 4096 | scan flat, reduce tax grows |
| memory-side work (row packing, layout) | DEAD — loads floor 2.3 ms; even post-cut it's ~22% of the kernel |

## Remaining headroom (honest)

Post-cut phase scan 10.2 ms vs ~6.5 ms quiet floor @4096 sgs; suffix 6.1
vs 3.7. Residual machinery ≈ **0.9 s @2^27** across both. The only door
that reaches the floor is a **global presort of rows by full lookup index**
(one permutation serves all 16 phases; scans become pure run-length) —
requires a scan-private reordered rows+u_evals copy, bucket_flat remap,
and fallback-contract care (`ScanOutcome::Corrupt` rebuilds u_evals in
original order, so the handoff stays sound). Parked: priced ~−1.1 s more
at material blast radius. IrrCycleRound exposed waits 0.84 s stay parked
(not reached; scan mass delivered 7× bar).

## Gates

- metal suites **411/411** · prover-fixtures byte-diff **20/20 first
  pass** · clippy `--all --features host` `-D warnings` clean · clippy
  jolt-kernels metal+bench-utils clean · fmt clean · pre-commit hooks green.
- Proofs verify in every e2e run cited. Byte parity default upheld — no
  protocol change, no soundness argument needed.
- Pre-existing (not this lane): `st0_contention.rs:418` fails clippy only
  under `--all-targets` + metal features (w9 rig, `contains()` lint) — the
  standard gate battery doesn't hit it.

## Discipline

- Timed 2^27 runs: **0**. Instrumented 2^27 profiles: **0** (w10/w11
  anatomy + @2^25 CB traces were sufficient for attribution).
- 2^25 e2e: 4 (position-balanced kill-switch ABBA, 30–45 s cooldowns) + 2
  CB-trace runs (attribution). 2^24: 1 untimed dump run. 2^21/2^22:
  byte-parity ×4 + one sgs-override sanity run.
- All kernel decisions @2^24 on REAL sha2 rows (standing rule: production
  distribution), ≤2 timed cells per decision, gpu-lock + cargo-lock
  throughout; FrBind 248.8 µs at session start (record-class window).
- Diff audited: proof-dump probe in `modular_benchmark` reverted;
  `irr_roof` retained as the permanent roof-attribution rig (floors,
  occupancy chain, sgs/width sweeps — the shipped-candidate probes were
  removed; arm A/Bs now run through the production kill switches). Fixture
  additions (`set_simdgroups`, scan-only/width/probe runners) are
  bench-utils-gated. Flag `irr_roof` + fixture probes for the PR-handoff
  audit alongside X9's rig.
- Not pushed; `scratch/metal-saturation` untouched; sibling worktree
  untouched. Worktree `.worktrees/metal-w12-st5scan` ready for merge +
  cleanup after the wave gate.

## Suggested gate measurements (orchestrator)

- Kill-switch ABBA @2^27 (one binary): default vs
  `JOLT_IRR_PHASE_SCAN_SCATTER=1 JOLT_IRR_SUFFIX_SCAN_GROUPED=1
  JOLT_IRR_PHASE_SCAN_SGS=512` — the full-trunk-behavior restore.
- Expect st5 12.74 → ~5.5 (scan CBs 9.83 → ~2.3–3.0) and a new absolute
  record; @2^25 the ON arm already ran 12.85/12.91 against the 14.70
  record in this window.
