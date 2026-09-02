# Lane N2 — Groth16 and HyperKZG scale measurements

Date: 2026-09-02. Repo commit: `a3f4489af`. Machine: Apple M4, 10 cores, 16 GiB,
macOS 26.5.2. Rust: `rustc 1.95.0`. Every run used `RAYON_NUM_THREADS=10` and a
release build.

## Decision data

- Arkworks Groth16 proves this 2–3-nonzero synthetic R1CS at **7.45 µs/constraint
  (2^20)** and **9.38 µs/constraint (2^21)**. The latter needs 7.91 GB peak RSS.
- A Groth16 proof is **128 B ark-compressed** or **256 B as eight uncompressed EVM
  field words**. Native verification is about 1.1 ms.
- Under the EIP-1108 schedule, a conventional one-public-input Groth16 verifier's
  four-pair check plus one `ECMUL` and one `ECADD` has a **187,150 gas precompile
  floor**: `45,000 + 4*34,000 + 6,000 + 150`. Contract code, memory, transaction,
  and calldata gas sit above this floor. Source: [EIP-1108](https://eips.ethereum.org/EIPS/eip-1108).
- HyperKZG open is **2.60 s at 2^21** and **5.85 s at 2^22**. Restricting the
  initial evaluations to 16-bit values changes open time by -1.7% and +0.2%,
  respectively: noise, not a scalar-width speedup.
- `setup_from_secret` is expensive but one-time: **92.24 s / 0.93 GB peak RSS at
  2^21**, **193.87 s / 1.80 GB at 2^22**. The RSS figure is the whole measured
  process while it retains the SRS and runs both openings, not a setup-only sample.

## A. Groth16

### Compatibility

The scratch build uses crates.io `ark-groth16 0.5.0`, `ark-relations 0.5.1`,
`ark-r1cs-std 0.5.0`, and `ark-snark 0.5.1` with algebra crates replaced by the
repo's `a16z/arkworks-algebra` `dev/twist-shout` fork at `76bb3a4518928f1ff7f15875f940d614bb9845e6`.
It compiles and runs. No upstream-only fallback was needed. The benchmark stayed
in `/Volumes/Dev/scratch/groth16-bench` to avoid changing the workspace manifest;
its source contains an `FpVar<Fr>` type marker so `ark-r1cs-std` is checked too.

### Relation

One public `x_0 = 1`; for each of `N` rows, allocate one new witness and enforce

```text
(x_i + 1) * (x_i + 2) = x_(i+1)
```

Each A/B row has two nonzeros including the constant wire; each C row has one.
The system has exactly `N` constraints, one public variable, and `N` witness
variables. Circuit-specific setup runs once. Proving runs twice; the table reports
the faster proof.

### Results

| Constraints | Setup | Prove 1 | Prove 2 | Best | Best µs/constraint | Verify | Max RSS |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2^20 = 1,048,576 | 8.389 s | 7.812 s | 7.809 s | **7.809 s** | **7.447** | 1.077 ms | **4.020 GB** |
| 2^21 = 2,097,152 | 21.346 s | 20.788 s | 19.663 s | **19.663 s** | **9.376** | 1.112 ms | **7.909 GB** |

`/usr/bin/time -l` also reported macOS peak-footprint values of 1.997 GB and
6.666 GB; max RSS above is the requested portable peak measure.

| Encoding | 2^20 | 2^21 |
|---|---:|---:|
| ark canonical compressed | 128 B | 128 B |
| ark canonical uncompressed | 256 B | 256 B |
| EVM proof ABI (`A.x,A.y,B.x[2],B.y[2],C.x,C.y`) | 256 B | 256 B |

### Native prover estimate

Arkworks is over the 5 µs/constraint trigger. A 2022 Consensys AWS hpc6a chart
reports about 8.5 s for circom/rapidsnark and 5.0 s for gnark at 8,000,000 BN254
constraints; arkworks is about 30 s in the same chart. Source:
[Devcon 6 slides, page 7](https://yelhousni.github.io/devcon6.pdf). Size-only linear
projection gives:

| Backend | Public rate | 2^20 estimate | 2^21 estimate |
|---|---:|---:|---:|
| rapidsnark | ~1.06 µs/constraint | ~1.11 s | ~2.23 s |
| gnark | ~0.63 µs/constraint | ~0.66 s | ~1.31 s |

The current gnark docs state over two million constraints/s on the 8M test,
which gives a tighter `<0.50 µs/constraint`, `<0.52 s`, and `<1.05 s` bound:
[gnark benchmark notes](https://docs.gnark.consensys.net/overview#gnark-is-fast).
These are circuit-size projections on a 96-core AMD host, not Apple M4 predictions.
Neither native prover was installed.

## B. HyperKZG

The scratch driver at `/Volumes/Dev/scratch/hyperkzg-scale` calls the repo's
`HyperKZGScheme<Bn254>` directly. Each scale creates one SRS with
`setup_from_secret`, then measures a random full-width polynomial and a separate
polynomial whose field evaluations are sampled from `u16`. Both proofs verify.
The same SRS remains resident for both cases.

### Results

| Evaluations | Input values | Commit | Open | Verify |
|---:|---|---:|---:|---:|
| 2^21 | full-width random Fr | 1.141 s | **2.600 s** | 2.464 ms |
| 2^21 | random u16 embedded in Fr | 0.379 s | **2.556 s** | 2.627 ms |
| 2^22 | full-width random Fr | 2.369 s | **5.847 s** | 3.391 ms |
| 2^22 | random u16 embedded in Fr | 0.867 s | **5.857 s** | 3.159 ms |

| SRS capacity | `setup_from_secret` | Whole-process real time | Max RSS |
|---:|---:|---:|---:|
| 2^21 | **92.235 s** | 99.30 s | **0.926 GB** |
| 2^22 | **193.873 s** | 209.62 s | **1.800 GB** |

Small input scalars make the initial commitment MSM 2.7–3.0x faster. Open does
not retain that benefit: the first multilinear fold combines each value with a
full-width challenge, so subsequent folds and the quotient-witness MSM use
full-width scalars.

## Exact commands

Scratch manifests use `[replace]` entries for the same arkworks fork as the repo.
The HyperKZG manifest uses absolute path dependencies into this worktree.

```bash
export CARGO_TARGET_DIR=/Volumes/Dev/cargo-target/wrap-spartan-hyperkzg

cd /Volumes/Dev/scratch/groth16-bench
cargo build --release -q --message-format=short
RAYON_NUM_THREADS=10 /usr/bin/time -l \
  /Volumes/Dev/cargo-target/wrap-spartan-hyperkzg/release/groth16-scale 20
RAYON_NUM_THREADS=10 /usr/bin/time -l \
  /Volumes/Dev/cargo-target/wrap-spartan-hyperkzg/release/groth16-scale 21

cd /Volumes/Dev/scratch/hyperkzg-scale
cargo build --release -q --message-format=short
RAYON_NUM_THREADS=10 /usr/bin/time -l \
  /Volumes/Dev/cargo-target/wrap-spartan-hyperkzg/release/hyperkzg-scale 21
RAYON_NUM_THREADS=10 /usr/bin/time -l \
  /Volumes/Dev/cargo-target/wrap-spartan-hyperkzg/release/hyperkzg-scale 22
```

The Groth16 binary argument is `log2(constraints)`. The HyperKZG argument is
`log2(evaluations)`; it performs full-width and u16-valued cases under one SRS.
