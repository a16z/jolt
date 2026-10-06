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
K16 packed decomposition uses Metal for the qualified resident `2^21`-row,
D512/capacity64 shape; other K16 shapes retain CPU routing.

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

Instruction Read-RAF also admits the nine-factor cycle tail at `2^21` rows.
The address phase and first cycle message remain on the CPU; the first cycle
bind fills shared Metal storage directly, and subsequent rounds retain those
tables on the GPU. `instruction_read_raf.small_k16_cutoff_elements` controls
readback to the CPU and defaults to 1,024 rows. The existing five-factor route
keeps its separate 65,536-row default. Nine-factor cycle tails at other trace sizes stay on CPU.

The five- and nine-factor arithmetic and binding regression is:

```sh
cargo nextest run --release -p jolt-kernels --features metal product_messages_and_bindings_match_direct_field_evaluation
```

Production instruction-input, registers claim-reduction, RAM value-check, and
RAM RA virtualization routes admit traces from `2^21` cycles. Their shape and
source checks still apply. RAM RA virtualization consumes the shared address
column directly and supports five 4-bit factors as well as the existing two
or three 8-bit factors. It keeps the initial rounds compact, then retains the
folded factor tables on the GPU.

Bytecode Read-RAF cycle processing supports five 4-bit RA factors (degree seven)
and the existing two 8-bit factors (degree four). The GPU constructs coefficients
from nine split equality roots and the shared compact instruction rows, evaluates
the first message, and materializes bound tables during the first bind. The
five-factor route returns to the CPU at at most 1,024 rows; the configured
`bytecode_read_raf_cycle.cutoff_elements` remains the upper bound. Address-phase
CPU preparation now accumulates field values only for addresses visited by each
worker, then writes one full-domain output.

Booleanity address and Hamming-weight preparation also support 4-bit chunks.
Their K16 kernel uses a separate 16-bin histogram per SIMD group and combines
uniform selectors before updating the histogram. Both preparations reuse the
resident instruction rows; Hamming-weight preparation consumes the final lease.
The existing 8-bit selector specializations remain in use for K256.

These changes preserve proof parameters, transcript ordering, and verifier code.
The component regressions and full kernel suite can be run with:

```sh
cargo nextest run --release -p jolt-kernels --features metal compact_and_dense_rounds_match_direct_products
cargo nextest run --release -p jolt-kernels --features metal production_kernel_matches_optimized_cpu_through_handoff
cargo nextest run --release -p jolt-kernels --features metal pushforward_matches_exact_cpu_oracle_across_selector_tiles
cargo nextest run --release -p jolt-kernels --features metal
```

The chunk-width and CPU-handoff regression is:

```sh
cargo nextest run --release -p jolt-kernels --features metal metal_k16_and_k256_sequences
```

Packed selector validation now reuses the witness producer's exact entry count
and certified zero suffix for K16 as well as K256. The Akita view still checks
all selector bounds; the zero suffix also lets commitment skip padded blocks.
A separate validation span distinguishes this CPU work from GPU coefficient
packing. This requires the companion Akita revision pinned in `Cargo.toml`.

The K16 packed decompose-fold route supports both sparse challenges and embedded
subring-64 challenges. It consumes the compact selector bytes directly, including
committed zeros, and skips blocks wholly inside the certified zero suffix. D128
and D512 K256 routes remain supported. Akita's CPU-oracle regression covers signed
challenges, 59 live columns, multiple digit counts, and a suffix that ends inside
a ring:

```sh
# In the pinned companion Akita checkout:
cargo nextest run --release -p akita-metal k16_decompose_fold_matches_cpu
```

A 12-pair alternating-order comparison on the local Longfellow fixture
(1,102,270 cycles padded to `2^21`, K16, `2^18` bytecode entries, D512,
59 live columns / capacity 64) measured median prover times of **1.622 s**
for `d5cd6f3` and **1.361 s** with the current architecture: **16.1% less
wall time (1.19x throughput)**. All 12 matched pairs improved. The M5 Max
used 18 Rayon threads, the default Metal configuration, and no tracing.
Each sample was the second proof in a fresh process; the first warmed caches.
Timing includes guest tracing plus proving, excludes preprocessing and
verification, and describes warm non-ZK execution on this fixture only.
All 48 untraced proofs verified and all 24 altered-output checks rejected;
the two additional profiling processes also verified all four proofs and
rejected altered output.

Separate profiles from that campaign measured bytecode cycle preparation and
rounds at 170 ms before versus 20 ms after, Booleanity address preparation at
23 versus 6 ms, Hamming preparation at 30 versus 4 ms, and RAM RA rounds at
47 versus 11 ms. These explain the changed work but are single profiled
observations, not additional end-to-end samples. Absolute times drifted
during the campaign; do not combine results from earlier campaigns to infer
a cumulative speedup. Proof size remained 93,928 bytes, with proof parameters,
transcript ordering, and verifier code unchanged.
