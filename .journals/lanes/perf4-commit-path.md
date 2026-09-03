# PERF-4 — phase-2a commitment path and wrapper-prover profile

Date 2026-09-03 · Mac mini M4 · 10 Rayon threads · CPU only · real fibonacci `2^18` wrap.

## Result

- `fc49dc13a` makes `g1_msm` detect signed-digit bucket skew and select the projective Pippenger path for skewed inputs. Uniform full-width scalars retain the batch-affine path; existing all-u16 inputs retain `small_msm`.
- k=32 phase 2a: **57.954 s recorded / 126.224 s reproduced → 6.368 s**, below the 8 s gate. The final run started at load **15.57 / 17.89 / 19.14**.
- k=16 phase 2a: **133.419 s recorded → 10.977 s** at load **13.00 / 16.99 / 18.74**.
- Proof wire is unchanged: k=32 **5,600 B payload / 5,706 B bincode**; k=16 **5,920 B / 6,038 B**.

## Root cause

The phase path is `commit_packed` → `HyperKZGScheme::commit` → `kzg_commit` →
`Bn254::g1_affine_msm` → `g1_msm`. Arkworks `commit_rows_dense` and
`msm_bigint_wnaf` are absent from this HyperKZG path.

The batch-affine kernel consumes one entry from each signed-digit bucket per inversion pass. T2's
repeated `Z_xi` operands and padding concentrate a 65,536-point sample into buckets of size
131 / 1,347 / 32,449 across the three phase-2a groups; a uniform random sample produces about
2–4. The three groups then ended together after 204.75 s with **2.62 busy threads**: one long
bucket chain per group kept three workers active while the rest finished early. The skew-aware
projective fallback made the same group block finish in 5.91 s; full `commit_packed` phase time was
6.199 s with **9.19 busy threads** in the phase-only run.

| k=32 phase-2a group | columns | selected full subset | kernel after fix | concurrent group wall |
|---|---|---:|---|---:|
| 0 | 32 operands | 6,522,440 | projective | 5.906 s |
| 1 | 12 operands + 20 helpers | 3,831,669 | affine | 5.920 s |
| 2 | 2 helpers + inverse + padding | 589,824 | projective | 5.909 s |

Each group wall includes Rayon work stealing across all three concurrent group calls; it is not an
exclusive CPU sum.

## Phase timings

| phase | W5 recorded k32, load 8.76 | reproduced before, load 8.38 | after k32, load 15.57 | after k16, load 13.00 |
|---|---:|---:|---:|---:|
| phase 1a | 2.180 s | 2.431 s | 0.823 s | 1.842 s |
| phase 1b | 1.846 s | 2.988 s | 0.804 s | 1.469 s |
| phase 2a | **57.954 s** | **126.224 s** | **6.368 s** | 10.977 s |
| phase 2b | 0.092 s | 0.237 s | 0.093 s | 0.071 s |
| helpers | 0.281 s | 0.578 s | 0.287 s | 0.412 s |
| phase 2c | 0.400 s | 0.833 s | 0.296 s | 0.409 s |
| prove | 15.399 s | 44.992 s | 14.810 s | 19.977 s |

The reproduced pre-fix run reported load 8.38 / 9.73 / 15.85. Its 126 s phase is a contention
sample, not the comparison baseline; the recorded W5 57.954 s is the cleaner before number.

## Why there is no 96-bit tier

Phase 2a does not commit raw limbs. `Columns::xi_values` computes
`Z_xi(v) = v_0 + xi*v_1 + xi^2*v_2`; `operand_columns` commits `kappa*Z_xi(src_x)` and
`sign*Z_xi(src_y)`. Random `xi` makes all 44 operand columns full-width `Fr`, matching the T2
width census. Sampling them as 96-bit values would be incorrect.

The exact future change is a typed phase-2a column carrying the three signed limb-component vectors
plus `[1, xi, xi^2]`: commit each component with a small signed scalar MSM, combine the three G1
results, and retain the materialized `Fr` vector for sumcheck/evaluation. Y components fit 96 bits;
X components need up to 103 bits after `kappa`, so the useful shared tier is signed u128. This needs
a T2 `StreamBuilder::phase_2a` export change in `limb_table/`; PERF-4 did not touch T2-owned files.

## Prove breakdown — k=32, 14.810 s at load 15.57

| item | seconds | prove share |
|---|---:|---:|
| stage A total | 4.540 | 30.7% |
| · T1 rows / wiring | 0.821 / 0.347 | 7.9% |
| · R / CopyLink | 0.246 / 0.269 | 3.5% |
| · T2 row / scalar link | 2.798 / 0.053 | 19.2% |
| column evaluations | 0.561 | 3.8% |
| term export + term stage + shared round opening | 0.014 | 0.1% |
| stage B + reduction | <0.001 | <0.1% |
| packed RLC + claimed-point evaluation | 0.524 + 0.292 | 5.5% |
| HyperKZG open | 8.877 | 59.9% |
| · fold materialization / fold commitments | 0.033 / 4.776 | 32.5% |
| · three-point evaluations / batch RLC / quotient | 0.126 / 0.039 / 0.322 | 3.3% |
| · quotient MSM | 3.579 | 24.2% |

## Next measured levers

1. HyperKZG fold commitments: **4.776 s / 32.2%** — 4-ary folding is the largest isolated target.
2. Quotient MSM: **3.579 s / 24.2%** — uniform random scalars stay on affine; tune that kernel independently.
3. T2 stage-A row member: **2.798 s / 18.9%** — coefficient-form range work and folding the input-claim pass are the prior candidates.
4. Packed column eval + RLC/check: **1.378 s / 9.3%** — typed bit/u16 evaluation avoids full-field materialization.
5. T1 stage-A rows + wiring: **1.168 s / 7.9%** — fused bind/evaluate remains the local target.

## Verification and landing

- `cargo nextest run -p jolt-crypto --cargo-quiet`: 145 passed.
- Strict all-target clippy passed for `jolt-crypto`, `jolt-hyperkzg`, and `jolt-wrapper`.
- `cargo check -p jolt-wrapper --all-targets` passed with and without `prover-fixtures`.
- Real k32 and k16 wrapper gates passed, including the existing tamper matrix and exact byte assertions.
- The repo-wide commit hook still reports pre-existing nominal-import findings and a `jolt-zeromorph`/private-HyperKZG-field build failure; targeted gates above are green.
- Lane commit `86bf98174`; rebased/shared-tree commit **`fc49dc13a`**. Shared T2 edits were autostashed and restored by the fast-forward.
