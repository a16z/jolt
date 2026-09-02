# Spartan + HyperKZG prover throughput

Measured 2026-09-02 on the target Mac mini. Each number is the median of three release samples after one warm-up. The primary table enables Arkworks' `parallel` feature and uses a 10-thread Rayon pool.

## Decision

- **Hard `<1 s` budget: `k = 15`, 32,768 constraints and 32,768 witness variables.** Measured total: **550.374 ms**.
- **`<700 ms` margin budget: also `k = 15`, 32,768 constraints and variables.**
- `k = 16` measures 1,074.508 ms. Spartan's power-of-two domains make 32,768 the usable limit even though linear interpolation puts the continuous 1 s crossing near 61k.
- HyperKZG commit + open is 97.1% of the `k = 15` total. Spartan-only throughput there is 2.04M constraints/s; combined throughput is 59.5k constraints/s.
- Large-domain combined throughput settles near 63k constraints/s. This does not change the sub-second padded-domain cutoff.

## Machine

```text
CPU: Apple M4, 10 cores (4 performance + 6 efficiency)
RAM: 17,179,869,184 bytes (16 GiB)
OS: macOS 26.5.2 (25F84), Darwin 25.5.0 arm64
rustc: 1.95.0 (59807616e 2026-04-14)
cargo: 1.95.0 (f2d3ce0bd 2026-03-21)
Rayon threads: 10 primary; 1 control
```

The `2^22` fixture and SRS fit in RAM. No other `cargo` or Jolt process ran during the measurements.

## Primary timing table: 10 threads, Arkworks parallel MSM

All units except throughput and bytes are milliseconds. Commit, open, verify, matrix products, and both sumchecks are measurements. `Estimated prover total = commit + open + Az/Bz/Cz + outer + inner`; throughput and both constraint cutoffs are derived from that sum. PCS and Spartan components ran separately under the stated 10-thread configuration; setup and verifier time are excluded. `Proof B` is the measured postcard-serialized HyperKZG opening proof, excluding its separate polynomial commitment and the Spartan proof.

| k | constraints | commit | open | verify | Az/Bz/Cz | outer SC | inner SC incl. L_w | estimated prover total | constraints/s | proof B |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 14 | 16,384 | 59.324 | 215.570 | 1.210 | 0.512 | 3.818 | 4.465 | **283.689** | 57,753 | 1,876 |
| 15 | 32,768 | 115.440 | 418.848 | 1.185 | 0.967 | 7.046 | 8.073 | **550.374** | 59,538 | 2,005 |
| 16 | 65,536 | 225.108 | 818.235 | 1.256 | 2.044 | 13.146 | 15.975 | **1,074.508** | 60,992 | 2,134 |
| 18 | 262,144 | 878.669 | 3,203.175 | 1.276 | 10.408 | 45.591 | 60.482 | **4,198.325** | 62,440 | 2,392 |
| 19 | 524,288 | 1,747.696 | 6,349.246 | 1.314 | 24.943 | 88.877 | 132.958 | **8,343.720** | 62,836 | 2,521 |
| 20 | 1,048,576 | 3,459.447 | 12,563.143 | 1.264 | 80.422 | 175.590 | 368.396 | **16,646.998** | 62,989 | 2,650 |
| 21 | 2,097,152 | 6,842.576 | 24,856.833 | 1.262 | 167.411 | 348.056 | 841.470 | **33,056.346** | 63,442 | 2,779 |
| 22 | 4,194,304 | 13,694.035 | 49,771.163 | 1.438 | 367.033 | 697.628 | 1,778.339 | **66,308.198** | 63,255 | 2,908 |

The measured `2^22` SRS setup took 158.149 s. Setup is offline and excluded above.

## Construction and checks

- Synthetic square R1CS: `m = n = 2^k`; each A and B row contains three random columns with random 64-bit-sampled BN254 Fr coefficients.
- Witness entries are random; `z[0] = z[2] = 1`. Each C row uses columns 0, 1, 2, random `c0,c1`, and `c2 = (Az)(Bz) - c0*z0 - c1*z1`, so `(Az) * (Bz) = Cz` row-by-row.
- Matrix-vector timing is three Rayon row passes for dense `Az`, `Bz`, and `Cz`.
- Outer sumcheck copies `jolt-blindfold/src/prove.rs`'s degree-3 loop using `jolt_poly::{Polynomial, EqPolynomial, UnivariatePoly}`. Each round checks `g_i(0) + g_i(1)`; the terminal claim is checked against bound `eq_tau * (Az*Bz-Cz)`.
- Inner timing includes the O(nnz) `L_w` projection and the degree-2 sumcheck over `L_w(j) * W(j)`. Its terminal claim is checked against the two bound evaluations.
- Fiat-Shamir uses the restored crate's Blake2b transcript. R1CS/witness construction, random input generation, and SRS setup are excluded as requested estimates rather than measured online stages.

## Thread scaling and workspace feature trap

The restored workspace does not enable `ark-ec/parallel`; a 10-thread Rayon pool therefore leaves each Arkworks MSM serial. The primary table temporarily added `ark-ec = { features = ["parallel"] }` only to the scratch benchmark. Production must enable that feature to reproduce it.

| `k = 16` configuration | commit | open | Spartan work | estimated total | speedup |
|---|---:|---:|---:|---:|---:|
| 1 thread, Arkworks parallel code path | 398.722 | 1,423.876 | 42.037 | 1,864.635 | 1.00x |
| 10 threads, Arkworks parallel | 225.108 | 818.235 | 31.165 | 1,074.508 | 1.74x |
| 10 Rayon threads, current serial Arkworks MSM | 398.788 | 1,418.488 | 31.165 | 1,848.441 | 1.01x |

Current workspace configuration still fits `k = 15` under the hard gate, but barely: 960.436 ms. Its `<700 ms` budget falls to `k = 14` (497.751 ms).

The limited parallel gain has a concrete source: `jolt-crypto`'s `JoltGroup::msm` converts projective bases to affine, converts scalar wrappers, and builds bigints with serial iterators before calling Arkworks. Enabling Arkworks parallel only parallelizes the final MSM. A persistent affine SRS plus parallel preparation is the next performance target; it was not estimated here.

## Native reference operations

| operation | median/mean cost |
|---|---:|
| dependent BN254 Fr multiplication, 20M iterations | 11.930 ns |
| light-poseidon width-4 permutation via empty `PoseidonSponge::absorb`, 100k iterations | 22,805.682 ns |

The Poseidon adapter measurement includes the small empty-absorb wrapper cost; construction of round constants is outside the loop.

## Commands

```bash
export CARGO_TARGET_DIR=/Volumes/Dev/cargo-target/wrap-spartan-hyperkzg

git checkout d80d201d6^ -- crates/jolt-hyperkzg
cargo clippy -p jolt-hyperkzg --release -q --message-format=short -- -D warnings

RAYON_NUM_THREADS=10 cargo run -p jolt-hyperkzg --example wrap_throughput --release -q --message-format=short
BENCH_KS=16 RAYON_NUM_THREADS=1 cargo run -p jolt-hyperkzg --example wrap_throughput --release -q --message-format=short

# Same PCS probe with scratch dev-dependency ark-ec/parallel enabled:
RAYON_NUM_THREADS=10 cargo run -p jolt-hyperkzg --example hyperkzg_parallel_probe --release -q --message-format=short
BENCH_KS=16 RAYON_NUM_THREADS=1 cargo run -p jolt-hyperkzg --example hyperkzg_parallel_probe --release -q --message-format=short
```

Both scratch examples and their temporary dev-dependencies were deleted after measurement.

## Restoration

The lane-dispatch journal commit `0dcb35e1f` had already placed the pre-#1795 crate files in the worktree before measurement began. Commit `992ad9d23` (`restore jolt-hyperkzg from d80d201d6^ (pre-#1795)`) restores the missing workspace member, workspace dependency, and lock entry. The restored crate needed no API edits and passes release clippy with warnings denied.
