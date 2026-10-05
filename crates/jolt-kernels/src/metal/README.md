# Akita Metal backend

This directory contains the Apple Metal implementation of the Akita prover's
hybrid sumcheck backend. It is available on macOS through the `metal` feature.

Use `JoltAkitaBackend::metal()` for the supported route set. The production
profile compiles the source fragments listed in
[`production_manifest.json`](production_manifest.json).

```rust
let prover = JoltAkitaBackend::metal()?;
```

The host owns Fiat-Shamir. A Metal member returns each round polynomial before
the host absorbs it and draws the next challenge. Admission and source checks
happen before a member's first polynomial; errors after that point are fatal
instead of falling back mid-transcript.

Large witness buffers move through proof-scoped owners and typed leases. Their
receipts bind the Metal device, source generation, allocation identities,
logical lengths, and completion state. The final declared consumer removes the
session owner; outstanding kernels retain only the buffers they still use.

## Benchmarking

The proof-valid benchmark is:

```bash
cargo run --release -p jolt-prover --example modular_benchmark \
  --features prover-fixtures,metal,profiling -- \
  --name fibonacci --scale 26 --backend metal --format chrome
```

Run the same command with `--backend optimized` for the CPU comparison. The
benchmark uses the production profile and does not expose shader tuning flags.
Use `--format none` for the no-subscriber timing baseline and `chrome` for
stage diagnostics. `prove_s` measures the packed prover: trace-row assembly,
commitment, sumcheck stages, and joint opening. Guest compilation, tracing,
preprocessing, and verification are outside that interval. Verification is
mandatory and prints `PROOF_VERIFIED backend=metal value=true` on success.

For a serial Fibonacci, SHA-2, BTreeMap, and BLAKE2b-512 chain sweep at padded
trace scales 2^24 through 2^28:

```bash
cargo install --path . --locked
cargo build --locked --release -p jolt-prover --example modular_benchmark \
  --features prover-fixtures,metal,profiling
python3 scripts/akita_metal_matrix.py --output benchmark-runs/metal-matrix
```

This runner requires macOS, Command Line Tools, Python 3.9+, and 128 GiB of
unified memory for T28. Full Xcode is not required. Use an idle Mac on AC power
with a fixed power mode. The runner builds all four guests before measuring;
pass `--binary` when using a non-default Cargo target directory. Use
`--max-scale 24` for a bounded smoke check.

Each cell gets one observation after a 120-second cooldown, a 180-second
process limit, and an 88 GiB process-family RSS stop sampled every 0.5 seconds.
The GPU watchdog remains enabled. Any failed proof, watchdog, timeout, swap,
memory guard, or mismatched trace size stops the sweep; there are no automatic
retries or outlier exclusions. The measurement budget is 85 minutes, excluding
guest preparation. Sampling can overshoot the RSS threshold.

BLAKE2b chains the full 64-byte digest from a 64-byte seed of 0x05. The sweep
uses 5,000 / 10,000 / 20,000 / 40,000 / 80,000 hashes at T24 through T28 and
independently checks the output with Python's `hashlib`. Other workloads use
the benchmark's default inputs.

`results.csv` records individual times, actual and padded rows, rates, and
memory. `manifest.json` records machine, source, binary, lockfile, and guest
identities; `events.jsonl` and `cells/` retain raw evidence and failures.
`REPORT.md` and `summary.json` report means only for complete four-workload
scales. Padded MHz is `2^scale / (prove_s * 1e6)`; the mean is the arithmetic
mean of individual rates, not the reciprocal of mean wall time. Actual-row
MHz is reported separately. No hardware projection is applied. One observation
per cell is descriptive, not an uncertainty or margin estimate.

`--resume` requires unchanged inputs and the original deadline. An incomplete
raw proof prevents resumption: investigate it and preserve its evidence before
starting a new output directory. Never retry a watchdog or memory failure
without resolving its cause. `--report-only` validates existing raw evidence
and regenerates the report without running a prover.

The `metal_*_cpu_eval` examples are manual component diagnostics on real trace
witnesses. They compare CPU and Metal round outputs, not full-proof throughput.
For example, use `RAYON_NUM_THREADS=16 cargo run --release -p jolt-prover
--example metal_outer_remainder_cpu_eval --features prover-fixtures,metal --
--help` to inspect its workload and arm controls. Run them serially; their
fixture preparation is not a substitute for the guarded matrix above.

## Protocol and validation

The K256 CPU and Metal routes share the canonical schedule catalog. At T28 the
root uses D128, rank 3, and 2^19 positions per block. This replaces the earlier
Metal-specific D512/rank-1 selection under the same SIS policy, challenge
distribution, and digit basis. It is a public-parameter/catalog transition,
not a verifier bypass. Do not mix setup artifacts from the retired Metal
catalog with the current selection. Catalog identity checks reject a
mismatched schedule; the old generic CPU catalog already admitted D128.

Packed one-hot inputs distinguish absent zeroes from committed lane zero:
lane zero contributes only when both its row bit and column-mask bit are set.
CPU/Metal parity must preserve that distinction as well as all nonzero lanes.

Before changing source assembly, run the manifest test and the macOS Metal
all-target clippy job. Protocol-facing changes also require the Akita end-to-end
proof and verifier tests.

```bash
cargo nextest run -p jolt-kernels --features metal --test-threads 1
python3 -m unittest discover -s scripts/tests -p test_akita_metal_matrix.py
```

## Small K16 traces

The Akita commitment and opening routes admit packed arity 31 (a `2^21`-row
K16 trace with column capacity 64), in addition to their existing large-trace
ranges. This uses the existing D512 schedule and kernels; it does not change
proof parameters or the verifier. Adjacent arities retain CPU routing pending
qualification. PIOP kernels keep their independent shape and size checks, and
K16 packed decomposition still runs on the CPU.

The shared instruction source stores the full 56-bit logical bytecode PC in a
separate column. Its five `u64` columns cost 40 bytes per row, an increase of
16 MiB at `2^21` rows (2 GiB at `2^28`) over the old four-column representation.
This fixes the former 14-bit source-packing limit; individual bytecode kernels
still have their own supported-domain checks.

On a Metal-capable Mac, run the small trace commitment/opening regression with:

```sh
cargo nextest run --release -p jolt-akita --features metal small_k16_trace
cargo nextest run --release -p jolt-kernels --features metal product_cap_fallback_releases_metal_sources
```

The instruction-RA sequence supports four-factor groups with either 4-bit
or 8-bit committed chunks. The production K16 route is qualified at `2^21`
cycles and can be disabled with
`instruction_ra_virtualization.enable_small_k16 = false`. It uploads the
shared stage-5 lookup indices when a resident address plane is unavailable;
the existing K256 route continues to consume that plane directly. Both routes
use the same lazy-prefix, dense-transition, and CPU-tail machinery.

Production instruction-input, registers claim-reduction, and RAM value-check
routes now admit traces from `2^21` cycles. Their existing shape and source
checks still apply. Other PIOP cutoffs, proof parameters, and verifier code
are unchanged by this follow-up.

The chunk-width and CPU-handoff regression is:

```sh
cargo nextest run --release -p jolt-kernels --features metal metal_k16_and_k256_sequences
```

On the local Longfellow fixture (1,102,270 cycles padded to `2^21`, K16,
`2^18` bytecode entries), an M5 Max with 18 Rayon threads measured median
prover times of 3.249 s CPU, 2.179 s before this follow-up, and 1.899 s with
the default Metal configuration after it. Each backend had five measured
samples, each following a fresh-process warmup; configuration order rotated
across rounds. All 30 proofs verified and all 15 altered-output checks rejected.
These are non-ZK measurements without tracing or threshold overrides, not a
claim about other guests or larger-trace throughput.
