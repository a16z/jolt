# Profiling Jolt

The modular prover (`crates/jolt-prover`) is the primary profiling target.
One command proves a named workload and emits two artifacts from the same
span stream: a Perfetto-viewable chrome trace for humans, and a
machine-queryable `summary.json` for scripts, agents, and `jolt-eval`
telemetry objectives.

```bash
cargo run --release -p jolt-prover --features profiling -- \
    profile --name sha2-chain --format chrome
```

Run these commands from a Jolt checkout. The command above uses the default
elliptic-curve backend, [Dory](../../how/dory.md). Add the `akita` Cargo
feature to use the lattice backend, [Akita](../../how/akita.md):

```bash
cargo run --release -p jolt-prover --features profiling,akita -- \
    profile --name fibonacci --backend optimized --format chrome
```

The `akita` feature selects the protocol for both proving and verification
at compile time. Jolt's Akita integration currently supports
non-zero-knowledge proofs only; `akita` and `zk` cannot be enabled together.
The Akita harness loads its schedule catalogs from
`crates/jolt-akita/schedules/` by default. Set `JOLT_AKITA_SCHEDULE_DIR` to
use another directory containing those `.aks` files. Packaged multichunk catalogs
cover only small test fixtures. Multichunk profiling requires a catalog bundle
generated for the selected geometry and workload, supplied through that environment variable.

For `profile` with the `akita` feature, `--akita-chunk-profile` selects
`w<chunks>r<levels>` (`w1r0` by default). For example:

```bash
cargo run --release -p jolt-prover --features profiling,akita -- \
    profile --name fibonacci --backend optimized --akita-chunk-profile w4r2
```

Chunked profiles append `_w<chunks>r<levels>` to the artifact name,
after the optional `_optimized` suffix, so their latest symlinks remain
independent. The CSV workload name and `summary.json`'s `run.workload`
also carry the chunk-profile suffix. `w1r0` has no workload suffix.
The `benchmark` sweep uses `w1r0`.

Workloads and default scales (`--scale <log2 trace length>` overrides):

| `--name` | default scale |
|---|---|
| `fibonacci` | 2^16 |
| `sha2-chain` | 2^22 |
| `sha3-chain` | 2^22 |
| `btreemap` | 2^20 |

`--backend` selects the kernel implementation for either Dory or Akita
(both subcommands): `reference`
(default) is the naive test oracle — absolute numbers are provisional,
attribution is meaningful relatively — while `optimized` is the performance
tier, slotting into the same instrumented seams.

`--memory-budget` (optimized backend only) is an explicit budget in bytes,
with an optional binary suffix (`KiB`, `MiB`, `GiB` or `TiB`, e.g. `32GiB`).
The profile reserves bytes against it in this order and exits with an error
at the first reservation that does not fit:

1. After tracing, before proving: the optimized prover's known eager
   reservation, which is the trace buffer at its reserved capacity plus the
   largest stage-1–5 envelope over the trace's rows before its constant
   padding suffix.
2. Under Akita, before PCS setup: the rest of the trace output, and the
   program and program preprocessing buffers.
3. Under Akita, before the setup allocates its one-hot backend: the setup's
   flat public matrix.

Each envelope counts:

- the Spartan, instruction-input and register (claim reduction, read/write
  checking and value evaluation) tables, with every register read/write entry
  generation at its worst case of three entries per cycle;
- the instruction claim reduction's construction collect and its storage after
  preparation;
- the RAM read/write kernel's storage through its first bind, with the RAF
  evaluation's folded and unmap tables;
- the RAM value check's scalar tables at their largest, and the RAM access
  columns that may still be held.

A reservation past the address space fails every budget. Under Akita, every
budget fails for the multi-chunk profiles and auxiliary (field-inline) setups
that the matrix check does not price. The reservations omit most other
allocations, so passing them does not mean the proof fits, and they are not a
lower bound on the memory a proof needs. The budget does not cap process
memory.

Artifacts are grouped by run: each invocation writes into
`benchmark-runs/{timestamp}_{trace_name}/` (with `{trace_name}` =
`modular_{workload}_{scale}` for Dory or
`modular_{workload}_akita_{scale}` for Akita, with hyphens in the workload
mapped to underscores; optimized runs append `_optimized`), and
`benchmark-runs/latest_{trace_name}` is symlinked to the
newest successful run — the stable path every example below reads. All
paths are under the current working directory. The directory name carries
the run identity, so the files inside use fixed names:

- `trace.json` — chrome trace. Open in
  [Perfetto](https://ui.perfetto.dev/) or query with `trace_processor` SQL.
- `summary.json` — schema-versioned aggregates (see below).

For example, the W4R2 command above writes its summary to
`benchmark-runs/latest_modular_fibonacci_akita_16_optimized_w4r2/summary.json`.
The query examples below use Dory's reference paths; substitute the
corresponding Akita or optimized path for those runs.

The run also compiles and traces the guest, derives the PCS setup, proves it,
and verifies the proof. PCS setup and `prove()` are timed separately. The
verifier is timed once under an explicit host-sized Rayon pool and once under
an exactly one-worker pool. The `profiling` feature enables the system
monitor, so CPU/memory counters render as native Perfetto counter tracks
directly from the emitted trace — no offline post-processing step.

## Benchmark sweeps

The `benchmark` subcommand sweeps workloads across scales — one `profile`
subprocess per (workload, scale), continuing past failures, with `--resume`
skipping pairs whose `latest_` link already exists:

```bash
cargo run --release -p jolt-prover --features profiling -- \
    benchmark --min-scale 18 --max-scale 21 --resume
# --benchmarks fibonacci,sha2-chain limits the workload set
```

Use the same Cargo feature to sweep Akita workloads:

```bash
cargo run --release -p jolt-prover --features profiling,akita -- \
    benchmark --min-scale 18 --max-scale 21 --backend optimized --resume
```

Results accumulate in `benchmark-runs/modular_timings.csv` (per-run CSVs live
in the run directories). Akita Single rows append `_akita` to the workload
name; chunked rows also append `_w<chunks>r<levels>`, for example
`fibonacci_akita_w4r2`. The summary table defaults to Single; select a
chunked profile with `--protocol akita --akita-chunk-profile w4r2`.
`--resume` checks the selected protocol and kernel implementation's
artifact path. In addition to prover throughput and proof size, the
CSV records `setup_time_s`, `verifier_parallel_time_s`,
`verifier_single_thread_time_s`, and the explicit parallel worker count.
Existing CSVs using the previous header are migrated in place with empty
values for these new fields. Render them with:

```bash
python3 scripts/benchmark_summary.py     # per-scale table
python3 scripts/benchmark_summary.py --protocol akita \
    --metric verifier_single_thread_time_s
python3 scripts/benchmark_summary.py --protocol akita \
    --akita-chunk-profile w4r2 --metric prover_time_s
python3 scripts/plot_benchmarks.py       # speed + proof-size plots
python3 scripts/plot_memory_usage.py     # peak memory per run (from summary.json)
```

Mind the machine: the reference backend retains ~18 GiB regardless of scale
and grows steeply with it — large-scale sweeps on the reference backend are
for big-memory hosts; use `--backend optimized` above small scales.

The span labels are a versioned public schema — the current taxonomy lives in the
`jolt-profiling` crate docs (`crates/jolt-profiling/src/taxonomy.rs`), the
normative source for label names, level policy, and the hot-loop rule.

## Querying without the Perfetto UI

`summary.json` answers the canonical questions with `jq` alone:

```bash
S=benchmark-runs/latest_modular_sha2_chain_22/summary.json

# Total prover wallclock (seconds) and dark time (root time not covered by
# any stage span)
jq '.root | {s: (.wall_time_ns/1e9), dark: .dark_time_fraction}' $S

# Total time across all instances of one span label
jq '.spans."EqPolynomial::evals".total_ns / 1e9' $S

# Top 10 spans by inclusive time
jq '.spans | to_entries | sort_by(-.value.total_ns) | .[:10]
    | map({label: .key, s: (.value.total_ns/1e9)})' $S

# Top 10 spans by self time (inclusive minus same-thread children)
jq '.spans | to_entries | sort_by(-.value.self_ns) | .[:10]
    | map({label: .key, s: (.value.self_ns/1e9)})' $S

# Per-stage wallclock breakdown
jq '.stages | map({label, s: (.wall_time_ns/1e9)})' $S

# Per-stage memory: boundary-RSS delta (retained growth) and windowed
# sample-max (null when the stage closed between monitor samples)
jq '.stages | map({label, rss_delta_gib, peak_memory_gib})' $S

# Hardware instructions retired, whole prove and per stage, with counting scope
# (null without a hardware counter; see Measurement semantics)
jq '{hardware_counters, root: .root.instructions,
     stages: (.stages | map({label, instructions}))}' $S

# Apple CPU cycles and V6 perflevel0 diagnostics
jq '{hardware_counters, root: (.root | {instructions, cycles, perflevel0}),
     stages: (.stages | map({label, instructions, cycles, perflevel0}))}' $S

# Top 10 span labels by instructions retired (orchestrating-thread spans only)
jq '.spans | to_entries | map(select(.value.instructions != null))
    | sort_by(-.value.instructions) | .[:10]
    | map({label: .key, count: .value.count, instructions: .value.instructions})' $S
```

Arbitrary ad-hoc queries go through Perfetto's `trace_processor` SQL against
the trace (the same SQL the Perfetto UI runs, now scriptable). Provision the
pinned pip package once:

```bash
uv pip install perfetto==0.57.2
```

```python
from perfetto.trace_processor import TraceProcessor

tp = TraceProcessor(trace="benchmark-runs/latest_modular_sha2_chain_22/trace.json")
q = tp.query("""
    SELECT name, COUNT(*) AS n, SUM(dur)/1e9 AS total_s
    FROM slice GROUP BY name ORDER BY total_s DESC LIMIT 15
""")
for row in q:
    print(row.name, row.n, row.total_s)
```

## Measurement semantics

- **Time metrics** cover the `jolt_prover::prove` root span only — guest
  compilation, tracer execution, and preprocessing are excluded.
- **Per-label totals** sum inclusive durations across all threads and may
  exceed wallclock under rayon parallelism.
- **Self time** subtracts same-thread children only; work on rayon workers
  attributes to its own labels.
- **Two peak-memory numbers, complementary**: `peak_rss_gib` is the
  process-lifetime `getrusage` high-water mark (cannot miss short spikes,
  but includes guest compile/trace); `root.peak_memory_gib` is the max over
  monitor samples inside the root span (prove-only, but sampled at ≥ 50 ms).
- **RSS under-reports on macOS under memory pressure**, where the kernel
  compresses cold pages out of the resident set. `peak_footprint_gib` is the
  process-lifetime peak physical footprint (the "peak memory footprint" of
  `/usr/bin/time -l`), which counts compressed pages; use it to compare
  memory across runs. On Linux it equals `peak_rss_gib`. The monitor's
  `counters.footprint_gib` samples the same measure over time (macOS only);
  `memory_gib` and `root.peak_memory_gib` remain resident-set based.
- **Hardware counts** (`root.instructions`, `stages[].instructions`,
  `spans.<label>.instructions`, and the corresponding `cycles` and
  `perflevel0` fields) are sampled at every enter and exit of a span
  on the orchestrating thread — the thread that set up tracing. Counts
  include parallel work during the interval and are inclusive, so nested
  spans are not additive. Spans inside rayon closures overlap their siblings
  and are not sampled; a label is `null` unless every instance ran on the
  orchestrating thread with usable readings. Background work is charged to
  whichever span is open when it executes, including idle workers spinning
  before sleeping. An unavailable count is `null`; a measured zero remains 0.

### Hardware counter scope

The top-level `hardware_counters` object records `source`, `privilege_scope`,
and `thread_scope`. Check it before comparing counts across runs:

| Source | Privilege scope | Threads and processes counted | Available fields |
|---|---|---|---|
| `linux_perf_event` | `user` | Opening thread and subsequently spawned descendants, including child processes; excludes pre-existing threads such as the chrome writer and monitor | `instructions` |
| `macos_rusage_v4` | `user_and_kernel` | All threads in this process, including pre-existing workers and profiler threads; excludes child processes | `instructions`, `cycles` |
| `macos_rusage_v6` | `user_and_kernel` | Same as V4 | `instructions`, `cycles`, `perflevel0: {instructions, cycles}` |

The corresponding `thread_scope` values are `opening_thread_and_descendants`
and `all_process_threads`. Unsupported hosts have `hardware_counters: null`;
setup logs a warning and all hardware counts remain `null`.

Linux uses `perf_event_open` with inheritance and kernel/hypervisor work
excluded. It requires an exposed PMU and suitable permissions (normally
`perf_event_paranoid` ≤ 2). Intervals where the kernel multiplexed the event
yield `null`, rather than an incomplete count. Initialize tracing before the
Rayon pool so worker instructions are included.

macOS reads `proc_pid_rusage(getpid(), RUSAGE_INFO_V6, ...)` on hosts with at
least two performance levels (`hw.nperflevels`), falling back to V4 if the
kernel rejects V6; single-level hosts, such as Intel Macs, read V4. The
backend first requires `kern.monotonic.supported = 1`; a successful API call
alone does not establish that counters are available. Both versions expose the
same hardware-backed total instructions and cycles. V6 additionally reports,
as `perflevel0`, the portion executed on XNU's `CLUSTER_TYPE_P` cores, which
`hw.perflevel0` describes: the highest performance level. Its name varies by
chip (`sysctl hw.perflevel0.name`): "Performance" on M1 through M4, but
"Super" on an M5 Max, whose "Performance" cores are `hw.perflevel1` and
excluded. XNU leaves this subset at zero on single-level hosts, which is why
they use V4. These are fixed CPU counter values attributed by
XNU's [Recount subsystem](https://github.com/apple-oss-distributions/xnu/blob/main/doc/observability/recount.md),
not estimates from elapsed time or configured `kpc` events. See Apple's
[CPU counter documentation](https://github.com/apple-oss-distributions/xnu/blob/main/doc/observability/cpu_counters.md).

Apple's process snapshots are not simultaneous, instruction-exact boundaries
across running workers. They include CPU work in the kernel and the profiler,
and do not identify cache misses, branch misses, or stalls. Cycles sum CPU
work across threads; they are not elapsed time. Core placement and frequency
affect cycle comparisons, and a `perflevel0` cycle fraction is not a fraction
of wall time. Compare repeated runs on the same machine with the same thread
count and power settings. The profiler emits raw counts: a ratio of CPU
cycles to guest instructions must use the actual live trace length, not the padded
`2^scale` length.

On Linux, fibonacci 2^16 (optimized backend) measurements spread ~0.1% at the
root and up to a few percent on sub-second stages, against ~0.01% under
`RAYON_NUM_THREADS=1`. One worker reduces scheduling noise for A/B comparisons;
it does not remove kernel and profiler work from Apple's counts. Linux's
counter save/restore cost measured under 1% of CPU time on fibonacci 2^20.
Measure macOS sampling overhead with the baseline procedure below instead of
assuming the Linux result applies.

### Validating on Apple Silicon

The hardware test below requires a physical Apple Silicon Mac with counters
available. It checks that work on a thread created before opening the counter
is included and fails if the counter cannot be used; it is ignored in ordinary
test runs so unsupported CI hosts do not claim to validate the hardware path.

```bash
sw_vers
sysctl -n machdep.cpu.brand_string kern.monotonic.supported hw.nperflevels \
    hw.perflevel0.name
cargo nextest run -p jolt-profiling --cargo-quiet
cargo nextest run -p jolt-profiling --cargo-quiet --run-ignored only \
    -E 'test(counters::macos::tests::counts_preexisting_workers)'

# Reinstall after switching revisions before building a guest.
cargo install --path . --locked
RAYON_NUM_THREADS=1 cargo run --release -p jolt-prover --features profiling -- \
    profile --name fibonacci --backend optimized --scale 16 --format chrome
jq '{hardware_counters, root: (.root | {instructions, cycles, perflevel0}),
     stages: (.stages | map({label, instructions, cycles, perflevel0}))}' \
    benchmark-runs/latest_modular_fibonacci_16_optimized/summary.json
```

Expect `macos_rusage_v6` on Apple Silicon (`macos_rusage_v4` on a kernel
without V6 or a single-level host) and positive root instruction and cycle
counts. V4 leaves `perflevel0` null. Under V6, `perflevel0` stays within the
totals and is zero only for intervals that ran entirely on lower performance
levels. Repeat with the normal worker count to exercise
parallel proving, then use the following procedure to check full-profiler
overhead. Keep the worker count the same in each baseline/instrumented pair.

## Overhead and dark-time budgets

The full subscriber stack must stay within 5% of an uninstrumented run, and
dark time (root wallclock not covered by any stage span) within 5%.
Verified by a manual procedure, not CI (wallclock thresholds on shared
runners are flaky gates) — on sha2-chain at scale 2^22, median of 3 runs on
the same machine:

```bash
# Baseline: no subscriber at all; prove() timed with std::time::Instant
cargo run --release -p jolt-prover --features profiling -- \
    profile --name sha2-chain --format none

# Instrumented: full stack (chrome + summary + monitor)
cargo run --release -p jolt-prover --features profiling -- \
    profile --name sha2-chain --format chrome
jq '.root | {s: (.wall_time_ns/1e9), dark: .dark_time_fraction}' \
    benchmark-runs/latest_modular_sha2_chain_22/summary.json
```

Budgets: `root.wall_time_ns` (chrome) ≤ 105% of the `--format none` Instant
measurement, and `dark_time_fraction` ≤ 0.05.

## Memory profiling (allocative)

With the `allocative` feature, the same profile command additionally
captures per-batch heap snapshots and renders the whole memory story as a
self-contained page:

```bash
cargo run --release -p jolt-prover --features profiling,allocative -- \
    profile --name fibonacci --format chrome
open benchmark-runs/latest_modular_fibonacci_16/memory.html
```

**`memory.html`** is the human view — one time axis carrying
the continuous RSS envelope (the monitor's `memory_gib` counter), the stage
spans as labeled bands, and at each snapshot instant a stacked composition
column of the live batch kernels, colored by relation family on one shared
byte scale, topped with the gray "unattributed" residual up to the envelope
(allocator retention + unvisited allocations). Click a column for the
snapshot's full-depth icicle; a table view carries every exact byte count.

The machine views: each snapshot persists as exact-bytes folded-stacks text
(`{run_dir}/{StageLabel}_prepared.folded`,
`root;child BYTES` per line), and the per-snapshot totals land in
`summary.json`'s `heap` section (per-root bytes, keyed by snapshot label),
so heap attribution is one `jq` away:

```bash
jq '.heap | map_values({gib: (.total_bytes / 1073741824),
                        top: (.roots | to_entries | max_by(.value) | .key)})' \
    benchmark-runs/latest_modular_fibonacci_16/summary.json
```

One snapshot per driver batch, taken right after every member kernel's
`prepare`, with all tables materialized and nothing bound yet — **the
stage's retained-memory peak**.
This is where the multi-GiB naive kernel tables show up. Every
`SumcheckKernel` is `MaybeAllocative`, so the live members are visited
directly, keyed by their concrete kernel type (which names the relation);
the proof session rides along, its carries attributed by concrete type name
through the per-entry visitors captured at park time. (End-of-stage
snapshots were dropped: stage working sets free on exit, so they were
near-empty by construction — anything genuinely carried across a boundary,
like the 6b→7 precommitted reduction, appears inside the consuming kernel
in the *next* stage's `_prepared` snapshot. A lingering RSS plateau after a
large `_prepared` graph is allocator retention, not live data. Stages 0 and
8 have no sumcheck batch and therefore no flamegraph — their memory lives
inside the commit and joint-opening slot calls; use the counter tracks and
the per-stage RSS table there.)

## jolt-eval telemetry objectives

Time, memory, heap, and instruction metrics are reachable as string-keyed `jolt-eval` objectives
(`telemetry:<workload>:<metric>`), so optimization agents can target any
span without editing `jolt-eval`. End-to-end instruction counts are
`telemetry:<workload>:instructions` and
`telemetry:<workload>:instructions:<span-label>`; deterministic
microbenchmark counts are available through the opt-in iai-callgrind lane
(`callgrind:<bench-name>:instructions`). See `jolt-eval/README.md`.

```bash
cargo run -p jolt-eval --bin measure-objectives -- \
    --objective telemetry:fibonacci:prover_time_s
# Heap attribution as an objective (builds the profile run with allocative;
# exact bytes; the root frame after the snapshot label is verbatim):
cargo run -p jolt-eval --bin measure-objectives -- \
    --objective telemetry:fibonacci:heap:Stage2Batch_prepared
```
