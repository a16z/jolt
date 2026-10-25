# Whiteout on Jolt + Akita: measured cost (V6)

Parent: `specs/whiteout-jolt-integration.md` (V6). Model sources:
`specs/whiteout-paper-deltas.md` (D9, D29), `specs/whiteout-sigma.md` (§2.5,
§5 candidates table), `akita:specs/whiteout-akita.md` §5.5.2 and W5-e (A30).

## Setup

- **Code.** jolt `wo/v6-costs`: `7f3259ecd` (region bytes, phase spans,
  one-time spans) on `5546d3170`; `9a54ffb84` adds the transported-verifier
  measurement. Akita: `~/akita-whiteout` at `299567ec6` (path patch in the
  root `Cargo.toml`).
- **Machine.** Apple M4 Max, 16 cores, 128 GiB, macOS 26.6.2. Shared
  machine: 1-minute load average at the start of each run 3.2–10.6 (batch)
  and 1.8–6.3 (transported runs).
- **Builds.** `cargo build --release -p jolt-prover --features profiling,akita`
  (clear Akita) and `--features profiling,akita,zk` (Whiteout).
- **Runs.** From the worktree root, per workload:
  `/usr/bin/time -l jolt-prover profile --name <muldiv|fibonacci> --scale <16|20> --format <fmt> --backend optimized`.
  The batch was 5 runs per cell, interleaving clear and Whiteout runs, at
  `7f3259ecd`. It also ran 2 runs per cell with `--format none`, whose prove
  times match within 0.5%. The transported-verifier runs were 2 runs each,
  at `9a54ffb84`.
- **Shapes.** `muldiv` is a fixed-size workload. Its `--scale` only names the
  run, and its trace is T = 2^12. `fibonacci --scale 16` is T = 2^16, and
  `--scale 20` is T = 2^20. The paper's and W4's headline is T = 2^24, so
  none of these shapes is the modeled one.
- **Rerun loop.** Σ and the whole proof were accepted on the first attempt in
  all 15 Whiteout runs.

Values are medians, with [min–max] where the spread matters. Times are
wall-clock span totals from `summary.json`.

## Results

| | muldiv (T = 2^12) | fib (T = 2^16) | fib (T = 2^20) |
|---|---|---|---|
| clear Akita prove | 0.19 s | 0.31 s | 1.04 s |
| Whiteout prove, total | 13.47 s [13.45–13.54] | 19.51 s [19.48–19.56] | 17.06 s [17.03–17.13] |
| Whiteout prove, minus one-time setup | 1.40 s | 1.93 s | 3.51 s |
| steady overhead over clear | +1.21 s | +1.62 s | +2.47 s |
| one-time: response plan (`whiteout_plan_response` → `find_schedule`) | 10.7 s | 16.2 s | 11.3 s |
| one-time: catalog plan (`whiteout_plan_catalog` → `find_zk_schedule`) | 1.36 s | 1.35 s | 2.22 s |
| one-time: lazy prover setup (`whiteout_catalog_prover_setup`) | 0.011 s | 0.017 s | 0.020 s |
| Σ (`whiteout_prove_tail` − response) | 0.36 s | 0.49 s | 0.71 s |
| response proof (`whiteout_prove_response`) | 0.76 s | 1.06 s | 1.69 s |
| masked Akita opening (`whiteout_prove_masked`) | 0.056 s | 0.076 s | 0.215 s |
| pads, layout + commit minus plan | 0.013 + ≤0.1 s | 0.014 + ≤0.1 s | 0.016 + ≤0.1 s |
| replay (stages + relation) | 0.051 s | 0.052 s | 0.056 s |
| verify, clear → Whiteout (parallel) | 6.8 → 185 ms | 7.0 → 207 ms | 9.9 → 250 ms |
| verify, first use on transported preprocessing | 12.3–12.4 s | not run | 13.7 s |
| peak RSS, clear → Whiteout | 0.32 → 2.86 GiB | 0.29 → 4.55 GiB | 0.60 → 7.94 GiB |
| proof bytes, clear → Whiteout | 81,977 → 112,931 | 84,120 → 115,141 | 88,685 → 117,480 |
| proof Δ | +30,954 B (+38%) | +31,021 B (+37%) | +28,795 B (+32%) |

Stages 1–7 take the same time as in clear proofs. Stage 0's steady cost is
also unchanged (0.010 / 0.014 / 0.20 s); its Whiteout total is the catalog
plan, which runs inside it. The response plan runs inside `commit_pads`, so
`commit_pads` itself shows 10.8 / 16.3 / 11.4 s.

The single-threaded verifier takes the same time as the parallel one under
Whiteout (185 / 205 / 249 ms).

O0 (`specs/whiteout-o0-profile.md`) traced this to the harness: its
single-threaded mode pinned only Akita backend calls to one thread. With the
whole verify on one thread, the Whiteout verifier is 2.6–2.9× slower than
with 16 (measured under load). O0 also splits the verify time by bucket.

### Argument-string regions

Bytes per region are differences of the `argument string region=… narg_bytes`
log lines, plus `response_bytes` from `whiteout_prove_response`. The proof
wire adds 10 B to the argument string.

| region | muldiv | fib 2^16 | fib 2^20 | clear counterpart |
|---|---|---|---|---|
| stage 0 | 224 | 224 | 224 | 224 |
| pads (t_p) | 384 | 384 | 384 | — |
| stages 1–7 | 16,448 | 18,176 | 19,904 | 16,432 / 18,160 / 19,888 (16 B less) |
| stage 8 reductions | 0 | 0 | 0 | 0 |
| Akita masked prefix (to the cut, ends with raw u_a) | 24,591 [24,590–24,592] | 24,818 | 25,275 | — |
| auxiliary blocks | 720 | 720 | 720 | — |
| Σ (compressed t_D + nonce n) | 160 | 160 | 160 | — |
| response proof | 70,393 [70,329–70,427] | 70,649 [70,625–70,671] | 70,803 [70,797–70,821] | — |
| clear Akita opening | — | — | — | 65,311 / 65,726 / 68,563 |

The proof Δ is the sum of the following:

- pads (+384);
- stages 1–7 (+16);
- masked prefix + auxiliary blocks + Σ + response proof, minus the clear
  opening (muldiv: 24,591 + 720 + 160 + 70,393 − 65,311 = +30,553).

The ~100 B spread between runs is in the response proof, plus 1–2 B in the
prefix. Both carry nonces and encoded field elements whose length depends on
the randomness.

## Predicted vs measured

Sources: paper `numbers.tex` / `06_evaluation.tex:173-241` (headline
T = 2^24, |e| = 65.5M); W4's cost model (`whiteout-sigma.md` §5, default
row D = 10, `whiteout-costs.py`); Akita §5.5.2 (W5e) and W5-e (A30).

| quantity | predicted | measured | gap | explanation |
|---|---|---|---|---|
| Σ bytes | 128 B compressed t_D + 32 B nonce = 160 B (paper; D9 after W5c) | 160 B on every run | 0 | — |
| pad bytes (t_p) | 128 B compressed (paper) | 384 B | 3× | Consistent with W5c's split chain (D9): an image above Akita's 512-coefficient compression cap becomes ⌈len/512⌉ chunks of 128 B each. t_p's coefficient count is not logged, so the three-chunk count is inferred, not checked. |
| response proof bytes | 72,516 B at \|e\| = 89.7M, ν = 27 (A30, synthetic instance) | 70.4–70.8 KB at ν = 24 / 25 / 26 (`rounds=` of the digit-range sumcheck) | −2.4% to −2.9% | Within 20%. The proof grows slowly in ν (+0.4 KB from ν = 24 to 26). |
| whole-proof Δ | +7.3 KB (paper headline); +9.66 KB (W4 model, default row) | +28.8 to +31.0 KB | 3.9–4.2× paper; 3.0–3.2× W4 | See the decomposition below. |
| Σ time | paper +1.15 s (one thread, read by W4 as Σ time); W4: ≈0.8 s at 16 threads at the headline (6.08M samples, A16) | 0.36 / 0.49 / 0.71 s at 16 threads | below both | Smaller shapes have fewer Gaussian samples, so this is not a like-for-like test. At T = 2^20 the time is already 0.89× the headline estimate. |
| steady prover Δ | paper +1.15 s | +1.21 / +1.62 / +2.47 s | +5% / +41% / +115% | The response proof (0.76–1.69 s) is outside the paper's figure as W4 reads it. On that reading the Σ share matches and the response proof is unmodeled. The paper does not price it separately, so this gap is open, not explained. |
| verifier Δ | paper +0.8 ms | +178 / +200 / +240 ms | 220–300× | The Σ-bound residual (D29, Akita A41/W5e) puts rank-19 `A_a` and `[B_a \| B′_a]` over R_64 in N, against the paper's rank 3 over R_256. Akita measured the same direction on its own harness: 0.019 → 0.076 s (single group) and 0.069 → 0.35 s (31:1 + 14:1). Akita's d = 256 estimate predicts about 3.5× less residual work. The per-component split of the Whiteout verifier was not profiled in V6. |
| residual setup | 7.0M–16.8M field elements per checked-in row (Akita §5.5.2 table) | lazy prover setup 0.011–0.020 s, +0.07 GiB footprint at T = 2^20 | — | Setup expansion is small next to the plans. The first-use cost is the planner search (below), not setup expansion. |

**Whole-proof Δ, decomposed (muldiv; the fib shapes agree to within 2.2 KB):**

- **Raw u_a: about 19.5 KB.** D29 gives 19 × R_64 = 1,216 coefficients ×
  16 B = 19,456 B at every checked-in row, and u_a is the masked prefix's
  last message. The paper's figure assumes u_a compressed to 128 B. The Jolt
  rows' residual rank is not logged here; it is taken from D29.
- **Rest of the masked prefix: about 5.1 KB.** These are the masked levels'
  messages before the cut.
- **Response proof minus the clear opening: +5.1 KB** (+4.9 KB at fib 2^16,
  +2.2 KB at fib 2^20). The response proof replaces the clear opening's
  levels after the cut. The clear opening grows with T, but the response
  proof barely does.
- **Pads, auxiliary blocks, Σ and stages: 1,280 B** (384 + 720 + 160 + 16).

Without u_a, the Δ is 9.3–11.5 KB. That is +28% to +58% over the paper's
7.3 KB and −3% to +19% against W4's +9.66 KB. Raw u_a explains the gap to
W4. The remaining 2–4 KB above the paper is the uncompressed-cut design (D29)
plus 720 B of auxiliary blocks and two extra pad chunks. The paper's
breakdown is not itemized finely enough to assign it further. Shapes also
differ: the model is at T = 2^24 and the measurements at T = 2^12–2^20.

## Findings

1. **One-time planning dominates Whiteout proving.** `find_schedule` for the
   response prover takes 10.7–16.2 s and `find_zk_schedule` for the catalog
   takes 1.3–2.2 s. Together that is 80–90% of a fresh process's prove time.
   Both are planned on first use and cached for the process.
2. **A fresh verifier pays 12–14 s on first use.** The Whiteout catalog and
   the response plan are derived verifier state (`serde(skip)`), so a
   verifier holding preprocessing from another process replans both before
   its first verify. Measured with `profile_verifier_transported`:
   - muldiv: 12.3–12.4 s, of which the catalog plan is 1.3 s, the response
     plan 10.8 s and the verify 0.2 s;
   - fib 2^20: 13.7 s, of which the catalog plan is 2.2 s and the response
     plan 11.3 s.

   This adds to W6f (N5): plan the catalog on first use, and build the prover
   setup twice. Shipping the planned rows and the response schedule as
   artifacts, or serializing them with the preprocessing, removes this cost.

   *Resolved by O1* (paper delta D32): the verifier setup persists both
   plans and a fresh verifier loads and validates them instead of planning;
   measurements are in `specs/whiteout-optimization-campaign.md` (O1).
3. **RSS grows by 9–13×:** 0.32 → 2.86 GiB (muldiv) and 0.60 → 7.94 GiB
   (fib 2^20). The `footprint_gib` counter in the existing `trace.json`
   attributes the growth as follows (fib 2^20; muldiv in parentheses):
   - **Catalog plan and prover setup:** +0.19 GiB (+0.11).
   - **Response plan:** a transient peak of 1.73 GiB (1.08). It settles at
     1.15 GiB (1.08) by stage 1.
   - **Σ** (tail before the response proof): 1.57 → 2.38 GiB (0.84 → 1.22).
   - **Response proof:** 2.38 → 7.38 GiB (1.22 → 2.42). This is the
     dominant term. It scales with `e`'s live digits (13.1M / 20.0M / 39.5M),
     and the memory is still held during verification.

   The counter is sampled, and footprint reads below peak RSS (7.42 vs
   7.96 GiB). The response proof's own breakdown (dense `A`, digit tables)
   needs an allocative run, which was not done.
4. **No retries.** Σ and the rerun loop accepted on the first attempt in
   15/15 runs. This is consistent with W4's default row (per-attempt
   rejection ≤ 2^−13.2).

## Reproducing the aggregates

The aggregation scripts are scratch tools and are not checked in. Their
inputs are reproducible:

- the `argument string region=… narg_bytes` lines and the
  `Whiteout response proof response_bytes=` line in the prover log;
- the `spans` in `benchmark-runs/latest_modular_<name>_whiteout_<scale>_optimized/summary.json`,
  queried with
  `jq '.spans | to_entries | map({k: .key, s: (.value.total_ns/1e9)})'`;
- the `footprint_gib` counter events (`ph == "C"`) in `trace.json`, read
  against the begin and end of each span.
