# HyperKZG prover performance

Measured 2026-09-02 on the Apple M4 Mac mini described in `throughput.md`. Release builds, 10 Rayon threads, one warm-up, median of three samples. Setup excluded. No concurrent Cargo or Jolt process.

## Result

- The requested `n = 2^19` target was not met. Final sweep: **333.679 ms commit, 669.484 ms open** versus 120/400 ms. An isolated `2^19` run reached 250.276/636.636 ms; the table keeps the single full-sweep medians rather than mixing runs.
- Commit is 5.2–6.4x faster and open is 9.3–9.5x faster through `2^19` than Lane F's Arkworks-parallel baseline.
- The wrapper budget now fits `n = 2^18` under both 1000 ms and 700 ms: **590.023 ms** estimated. `n = 2^19` is **1249.941 ms**.
- Proofs contain one KZG quotient commitment instead of three. This changes the proof and setup formats: the verifier key now carries powers through `beta^2 * g1` and `beta^3 * g2`.

## Before and after

Milliseconds except bytes and speedup. Before is Lane F's 10-thread Arkworks-parallel table. After is one final full sweep.

| k | n | commit before | commit after | speedup | open before | open after | speedup | verify after | proof B before | proof B after |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 65,536 | 225.108 | 38.176 | 5.90x | 818.235 | 86.944 | 9.41x | 1.654 | 2,134 | 2,068 |
| 18 | 262,144 | 878.669 | 137.143 | 6.41x | 3,203.175 | 336.399 | 9.52x | 1.862 | 2,392 | 2,326 |
| 19 | 524,288 | 1,747.696 | 333.679 | 5.24x | 6,349.246 | 669.484 | 9.48x | 2.063 | 2,521 | 2,455 |
| 20 | 1,048,576 | 3,459.447 | 576.086 | 6.00x | 12,563.143 | 1,348.396 | 9.32x | 2.138 | 2,650 | 2,584 |

The `2^18` one-thread control measured 712.483 ms commit, 1,652.000 ms open, and 1.561 ms verify. Ten threads gave 5.20x commit and 4.91x open speedups in the final sweep.

## Lever accounting

Comparable isolated `n = 2^18` medians:

| configuration | commit ms | open ms | effect |
|---|---:|---:|---|
| Lane F, Arkworks parallel with per-call projective conversion | 878.669 | 3,203.175 | baseline |
| pre-normalized SRS, Arkworks serial | 768.335 | 2,770.798 | conversion cost removed |
| pre-normalized SRS, Arkworks parallel | 155.451 | 778.012 | parallel Arkworks path |
| persistent affine SRS, global signed-window MSM | 131.919 | 619.015 | no wrapper/scalar byte conversions; no nested two-thread pools |
| one degree-three quotient commitment | 132.111 | 313.327 | opening MSM work reduced from about 4n to about 2n points |

Horner evaluation halves field multiplications in each univariate evaluation; initializing the batch polynomial from `f[0]` also removes the `q^0` multiply pass. The consecutive `2^20` full sweeps moved open from 1,536.432 to 1,348.396 ms, though thermal and heterogeneous-core variance is visible at smaller sizes.

The current Nova source also stores its commitment key in affine form and batch-commits folded polynomials. Its opening path still runs three independent linear quotient openings in parallel (`u.into_par_iter().map(kzg_open)`), so Jolt's single cubic quotient is a protocol change rather than a direct port: <https://github.com/microsoft/Nova/blob/main/src/provider/hyperkzg.rs#L997-L1001>.

## Remaining floor

- Final sweep MSM rate: 0.52–0.64 microseconds per point at 10 threads; best isolated `2^19`: 0.477 microseconds per point. Pre-normalized Arkworks measured 0.593 microseconds per point at `2^18`; the signed-window path is about 12% faster there.
- `open` now has two dominant MSM-equivalents: all folded commitments total less than `n` points, and the cubic quotient is `n - 3` points. At `2^20`, two commit-equivalents account for about 1.15 s of the 1.35 s opening.
- The `2^19` commit target requires another 2.1–2.8x reduction depending on isolated versus full-sweep variance. The 400 ms open target also requires commit below about 200 ms before field work.
- Halo2curves 0.9 `msm_best` was tested separately at `2^18`: 155.5 ms, slower than the retained signed-window Arkworks implementation. It was not added as a dependency.

## Wrapper budget

Spartan costs come from Lane F: matrix products plus outer and inner sumchecks.

| k | commit + open ms | Spartan ms | estimated wrapper prover ms | <1000 ms | <700 ms |
|---:|---:|---:|---:|:---:|:---:|
| 16 | 125.120 | 31.165 | 156.285 | yes | yes |
| 18 | 473.542 | 116.481 | **590.023** | yes | yes |
| 19 | 1,003.163 | 246.778 | **1,249.941** | no | no |
| 20 | 1,924.482 | 624.408 | 2,548.890 | no | no |

Largest power-of-two domain under either budget: **`n = 2^18 = 262,144`**.

## Commits and checks

- `abab852a5` — prepared affine BN254 G1 signed-window MSM
- `cfa02939f` — Arkworks parallel features in `jolt-crypto`
- `6634e39f2` — one cubic KZG quotient plus parallel/Horner field work

```text
cargo nextest run -p jolt-hyperkzg --release --cargo-quiet
# 22 passed

cargo clippy -p jolt-hyperkzg -p jolt-crypto --release -q --message-format=short --all-targets -- -D warnings
# passed

cargo clippy --all --features host -q --message-format=short --all-targets -- -D warnings
# passed
```

The scratch benchmark and temporary dependency were deleted after measurement.
