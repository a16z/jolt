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

The Akita commitment and opening routes admit packed arities 30 and 31
(`2^20`- and `2^21`-row K16 traces with column capacity 64), in addition to their existing large-trace
ranges. This uses the existing D512 schedule and kernels; it does not change
proof parameters or the verifier. Adjacent arities retain CPU routing pending
qualification. PIOP kernels keep their independent shape and size checks, and
K16 packed decomposition uses Metal for the qualified resident `2^20`- and
`2^21`-row, D512/capacity64 shapes; other K16 shapes retain CPU routing.

K16 root commitment accumulates radix-26 digits and propagates carries after
at most 16 signed contributions. It shares the bounded accumulator and final
field reducer with the D128/rank-3 route. Two D512 positions fit in five shared
memory planes, within the existing kernel's buffer and dispatch geometry.
The committed-zero mask and certified zero suffix keep their existing meaning;
K256 continues to use its coefficient-panel path. CPU commitment parity covers
dense K16 selectors as well as sparse selectors and zero suffixes.

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
or 8-bit committed chunks. The production K16 route is qualified at `2^20`
and `2^21` cycles and can be disabled with
`instruction_ra_virtualization.enable_small_k16 = false`. It uploads the
shared stage-5 lookup indices when a resident address plane is unavailable;
the existing K256 route continues to consume that plane directly. Both routes
use the same lazy-prefix, dense-transition, and CPU-tail machinery.

Instruction Read-RAF also admits the nine-factor cycle tail at `2^20` and `2^21` rows.
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
RAM RA virtualization routes admit traces from `2^20` cycles. Their shape and
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

The P-256 SDK uses joint sparse signed-digit recoding for each of the two
independent Fake GLV scalar-multiplication checks. It also avoids subtracting
the modulus when a field-addition result is already canonical. Both changes
preserve the scalar identities and all input/advice validity checks; the
signed expansion accommodates the carry beyond a full-width `u128`.

These guest-side changes can move a workload below a trace-padding boundary.
For the Longfellow fixture, actual execution drops from 1,102,270 to 1,040,282
cycles, so the derived padded trace halves from `2^21` to `2^20`. The smaller
Metal commitment/opening, instruction-RA, Instruction Read-RAF, and production
instruction-input/register-claim/RAM routes keep that workload on the GPU.
The guest program and derived proof shape change; the proof protocol and
verifier rules do not. Workloads that do not cross a padding boundary will
not receive the same reduction in proving work.

The arithmetic and both-size commitment/opening regressions are:

```sh
cargo nextest run --release -p jolt-inlines-p256 --features host
cargo nextest run --release -p jolt-akita --features metal small_k16_trace
```

A 12-pair alternating-order Longfellow comparison measured median prover time
of **1.370 s with the previous guest versus 1.048 s with the optimized guest**:
**23.5% less wall time (1.31x throughput)**. All 12 pairs improved. Both guests
ran in the same final prover binary on an M5 Max with 18 Rayon threads,
default Metal configuration, and tracing disabled. Saved guest ELFs fixed the
programs throughout the run; each measured proof followed one warmup in a
fresh process. Timing includes guest tracing plus proving and excludes
preprocessing and verification. The fixture retains K16, `2^18` bytecode
entries, D512 and column capacity 64. Proof size changes from 93,928 to 90,647
bytes with the smaller derived trace shape.

All 48 untraced proofs and four profiling proofs verified; all 26 altered-output
checks rejected. Separate single profiles measured opening time at 541 ms
before and 329 ms after, and stage 5 at 214 ms before and 144 ms after.
These profiles explain the reduction in work; the quoted end-to-end gain
comes from the untraced paired campaign. This is a warm non-ZK result for
this workload and padding transition, not a general GPU throughput claim.

A subsequent 12-round, rotating-order comparison held that optimized guest
fixed and measured the radix-26 K16 commitment change separately from thread
count. Each of four configurations produced one warmup and one measured proof
per fresh process, with the same timing boundary and default Metal policy:

| Configuration | Before kernel change | After kernel change | Less wall time |
| --- | ---: | ---: | ---: |
| 18 Rayon threads | 1.032 s | 0.994 s | 3.6% |
| 8 Rayon threads | 0.962 s | 0.924 s | 4.0% |

The kernel improved 11/12 pairs at 18 threads and 12/12 at 8 threads. Combining
the kernel change with `RAYON_NUM_THREADS=8` reduced time by **10.5%** against
the prior 18-thread configuration, improving all 12 pairs. That combined
number includes thread tuning; it is not the kernel-only gain. Both the Jolt
and Akita pools used the stated thread count. Eight threads is a measured
choice for this workload on the M5 Max, not a new backend default.

All 96 untraced and eight profiling proofs verified, and all 52 altered-output
checks rejected. The program still executes 1,040,282 cycles, pads to `2^20`,
and produces 90,647-byte proofs. Single 18-thread profiles measured packed
Metal commitment dispatch at 123 → 85 ms and stage 0 at 165 → 125 ms. The
kernel uses the existing field, schedule, commitment and verifier equations.
