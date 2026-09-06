# Akita Metal: four-workload M4 Max handoff

Measure Fibonacci, SHA2-chain, BTreeMap and BLAKE2b-512 chain at every padded
trace scale from 2^24 through 2^28. The question is whether the arithmetic
mean of their measured padded proving rates reaches 10 MHz at T28. A lower
mean is a miss, and an incomplete or invalid row cannot establish the target.
Do not multiply measured M4 results by a hardware projection to claim a pass.

## Scope and measurement contract

The Akita fork's `feat/akita-metal` selects
`7878e5ba14b0a9f7cc89ddbc1331a543efa11db7`, the validated staged-radix26
commit kernel. Jolt pins that exact public Git revision. No local path
overrides, protocol, security-parameter, verifier or transcript changes.

One observation per cell, 20 cells total, serial on one otherwise idle Mac.
Build the host and all four guests before measuring. Use `--format none`,
the existing preprocessing/prove/verify timing boundary, full 120-second
cooldowns, a 180-second process limit, 88 GiB sampled family-RSS stop, zero
reported swaps and the existing GPU watchdog. Stop on any failed proof,
watchdog, timeout, memory guard or mismatched trace size. No automatic retry
or dropping outliers. A full sweep has an 85-minute hard budget; expected
measurement time is roughly 45 minutes, excluding installation/AOT builds.
This is an exceptional external evidence refresh, not a local optimization loop.

BLAKE2b uses a 64-byte seed of 0x05 and chains the full 64-byte digest.
Use 5,000 / 10,000 / 20,000 / 40,000 / 80,000 hashes at T24 through T28.
The target-trace CLI override is `171520000 >> (28 - scale)`; the harness's
2144 rows/hash input scaler selects those exact counts. Check each digest
independently. The larger 112,682-hash T28 input exceeded the memory guard;
it is not the workload in this matrix. Other inputs retain harness defaults.
Every actual trace must exceed half its padded size and be at most that size.

For each workload i at scale s:

```
padded_MHz_i = 2^s / (prove_seconds_i * 1e6)
actual_MHz_i = trace_length_i / (prove_seconds_i * 1e6)
mean_padded_MHz_s = sum(padded_MHz_i for the four workloads) / 4
```

Average rates, not wall times followed by an inversion. Preserve exact
trace lengths, individual times, memory, verification markers and raw logs.
The optional 1.13x M5 projection is a separate column, never a measured result.
A single observation is descriptive: a near-threshold result is not a robust
margin claim. Retain all observations if follow-up replication is requested.

## Local integration validation

Reuse the unchanged kernel's completed Akita Metal 32/32, Jolt Metal 329/329,
PCS 23/23 in both serial/parallel modes, both Jolt and all three fork clippy
configurations. The accepted production-file hashes must match before pinning.
For the new handoff code: fresh locked build, both Jolt clippy modes, focused
profiling/Metal clippy, runner contract tests and a four-workload T24 smoke.
Do not rerun the full T24–T28 matrix locally; Andrew supplies that independent
measurement. Local validation reserve: 25 minutes after build launch, serial.

Historical T28 observations, not a fresh four-cell matrix:

| Workload | M4 seconds | M4 padded MHz | Projected M5 MHz |
|---|---:|---:|---:|
| Fibonacci | 30.5042 | 8.8000 | 9.9440 |
| SHA2-chain | 27.2369 | 9.8556 | 11.1368 |
| BTreeMap | 27.0895 | 9.9092 | 11.1974 |
| BLAKE2b-chain, 80k | 30.7036 | 8.7428 | 9.8794 |
| Arithmetic mean | | 9.3269 | 10.5394 |

Thus the current evidence exceeds 10 MHz only under the M5 projection, not
on M4. Kernel acceptance details and inherited validation limitations are in
`benchmark-runs/akita-commit-occupancy-2026-09-06/acceptance.md`.

## Andrew: run from a fresh checkout

Requirements: Apple Silicon Mac with Metal, 128 GiB unified memory for T28,
Command Line Tools, Rust/rustup, Python 3.9+ and network access for installation.
Full Xcode and third-party Python packages are not needed. Use AC power and
the same power mode throughout; close GPU-heavy apps and other builds/provers.

```bash
git clone --branch feat/akita-metal https://github.com/a16z/jolt.git jolt-akita-metal
cd jolt-akita-metal
cargo install --path . --locked
cargo build --locked --release -p jolt-prover --example modular_benchmark \
  --features prover-fixtures,metal,profiling
python3 scripts/akita_metal_matrix.py --output benchmark-runs/andrew-m4
```

The script prepares all four guests before starting cooldowns and freezes
the binary, Git revision, dirty diff, lockfile, guest hashes, machine identity
and timing contract in `manifest.json`. It never builds the host binary for
you. If using a non-default Cargo target directory, pass its binary explicitly
with `--binary /absolute/path/release/examples/modular_benchmark`.

The runner samples family RSS every 0.5 seconds (the prior campaign used
2 seconds). Limits and the timed boundary are unchanged; sampling can still
overshoot a threshold. It rejects non-default watchdog/decomposition/census
environment settings. Do not disable the watchdog or raise memory limits.

Results are updated after each verified observation:

- `REPORT.md`: per-scale measured mean MHz and measured >=10 MHz decision;
  the optional M5 projection is labeled separately.
- `results.csv`: individual times, actual/padded rows and rates, memory,
  verification status and raw-file hashes.
- `summary.json`, `manifest.json`, `events.jsonl`, `cells/`: machine-readable
  results, provenance and immutable individual raw logs.

Send the whole `benchmark-runs/andrew-m4/` directory back. Do not average only
the fastest programs. A T28 mean is printed only after all four T28 cells pass.

After an interruption during cooldown, resume within the original 85-minute
measurement budget without changing source, binary, machine or scales:

```bash
python3 scripts/akita_metal_matrix.py --output benchmark-runs/andrew-m4 --resume
python3 scripts/akita_metal_matrix.py --output benchmark-runs/andrew-m4 --report-only
```

An incomplete raw proof or a workload lock stops resumption. First confirm
there is no surviving prover, inspect the failure and archive a stale lock
manually if applicable. Keep that output directory intact and use a new one
for an explicitly agreed rerun; never keep only the faster timing. Resume is
not authorization to retry a watchdog, memory or correctness failure.

A bounded smoke uses the same guards but only T24:

```bash
python3 scripts/akita_metal_matrix.py --output benchmark-runs/m4-smoke --max-scale 24
python3 -m unittest discover -s scripts/tests -p test_akita_metal_matrix.py
```

Validation results are recorded separately in the handoff evidence directory;
the full 20-cell matrix is intentionally left for Andrew.
