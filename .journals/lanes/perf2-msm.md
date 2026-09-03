# PERF-2 — BN254 MSM and packed bit-column commits

Date: 2026-09-03 · Mac mini M4 · 10 Rayon threads · CPU only · proof format unchanged.

## Result

- **L7 landed:** packed bit columns now reduce 16K-base chunks across every column with one batch inversion per tree level. The production-shaped 22-group block measured **0.479 s → 0.418 s** (load 7.43 after); the isolated 22 × 2^18 benchmark measured **72.6 ms**.
- **L2 rejected after the full gate:** batch-affine Pippenger cut an isolated 2^21 MSM by 25%, and the HyperKZG open by 15%, but four concurrent helper-group MSMs lost parallel utilization and regressed **2.66 s → 5.40 s**. Commit `d88102bb5` is canceled by `4504ec466`.
- Final k=8 payload remains **5,600 B**; k=16 remains **5,184 B**.

## L2 experiment

`d88102bb5` kept signed Booth digits and the existing window × chunk tiling. Each chunk built a per-bucket linked list; one point per occupied bucket was scheduled per pass, collisions stayed queued, and all scheduled affine additions shared one Montgomery inversion. Identity, inverse, doubling, duplicate-point, zero-scalar, random full-width, random ≤16-bit, and non-power-of-two cases passed while the experiment was active.

Three-repeat isolated MSM measurements:

| N | projective control min / median | batch-affine min / median | 1-minute load |
|---:|---:|---:|---:|
| 2^20 | 0.544 / 0.545 µs/point | 0.387 / 0.400 µs/point | control 8.07; affine 8.74 |
| 2^21 | 0.552 / 0.555 µs/point | 0.398 / 0.415 µs/point | control 8.07; affine 8.74–8.84 |

The ≤0.30 µs/point target was not reached. A 16-bit window won at 2^21; 17 bits lost. Four- and eight-lane prefix products did not produce a stable additional win and were removed.

Full-statement gate with L2 active (load 8.45–10.37):

| k=8 phase | control | L2 | verdict |
|---|---:|---:|---|
| HyperKZG open | 2.466 / 2.486 s min / median | 2.032 / 2.087 s | −0.40 s |
| 20-helper mixed commits | 2.661 s | 5.396 s | +2.74 s |
| all packed commits | 3.888 / 3.940 s | 6.396 / 6.578 s | +2.64 s |

The helper regression dominates the opening win. Post-revert targeted control: helpers **2.680 s** at load 7.43; open **2.558 s** at load 7.64. Both match the earlier control within shared-box variation.

## L7

`528428643` replaces one independent affine tree per sparse packed column with chunk-parallel, column-interleaved trees. Sixteen-thousand-base chunks provide enough Rayon tasks without holding all selected points at once. Dense 163-column inputs retain the six-base subset-table path.

Measured isolated kernel, 22 columns × 2^18 bases: **72.6 ms, 25.2 ns/add**. Production k=8 profile: **0.418 s** for 22 groups after, versus **0.479 s** before. The full L2-active gate measured 0.373 s at load 8.69; the post-revert targeted gate measured 0.418 s at load 7.43.

## Verification

- `cargo nextest run -p jolt-crypto --cargo-quiet`: 141 passed after the L2 revert.
- `cargo nextest run -p jolt-hyperkzg --cargo-quiet`: 25 passed after rebasing.
- `cargo clippy -p jolt-crypto --all-targets -q --message-format=short -- -D warnings`: passed.
- `cargo clippy -p jolt-hyperkzg --all-targets -q --message-format=short -- -D warnings`: passed.
- Full statement: `perf1_full_statement_profile`, one invocation, 169.7 s. Targeted final: `perf2_commit_open_profile`, 13.9 s.

## Landed on `wrap/spartan-hyperkzg`

- `e116291a3` fixed-base setup plus parallel KZG evaluation/division.
- `358be1e7d` Pippenger task tiling.
- `adabfe391` full-statement profile gate.
- `d88102bb5` batch-affine L2 experiment; canceled by `4504ec466`.
- `528428643` interleaved packed bit-column commits.
- `2e5fda78a`, `7c669b7d9`, `a346cf32e` repeatable PERF-2 gates and current-layout fix.

Scratch worktrees still to remove: `/Volumes/Dev/worktrees/jolt/perf1`, `/Volumes/Dev/worktrees/jolt/perf1-base`.
