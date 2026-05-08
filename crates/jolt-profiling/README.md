# jolt-profiling

Profiling and tracing infrastructure for the Jolt proving system.

Part of the [Jolt](https://github.com/a16z/jolt) zkVM.

## Overview

Provides a unified interface for performance analysis across all Jolt crates. Individual library crates instrument their functions with `#[tracing::instrument]`; the host binary depends on `jolt-profiling` to configure the subscriber that captures those spans.

## Public API

### Tracing Setup

- **`setup_tracing(formats, trace_name)`** — Initializes the global tracing subscriber. Supports console output (`Default`) and Perfetto/Chrome JSON traces (`Chrome`). Returns flush guards that must be kept alive.
- **`TracingFormat`** — Output format enum: `Default` (console), `Chrome` (Perfetto JSON).

### Memory Profiling

- **`start_memory_tracing_span(label)` / `end_memory_tracing_span(label)`** — Tracks physical memory deltas across labeled code regions.
- **`report_memory_usage()`** — Logs all collected memory deltas and warns about unclosed spans.
- **`print_current_memory_usage(label)`** — Logs current physical memory at point of call.

### Measurement Helpers

- **`time_it(f)`** — Measures closure runtime in milliseconds and returns the closure result.
- **`median_f64(values)` / `median_u64(values)`** — Computes medians for benchmark reports without panicking on empty input.
- **`PeakRssSampler::start()`** — Samples peak resident memory for perf gates on non-wasm targets.

## Core-vs-Bolt Perf Oracles

Jolt-on-Bolt perf gates should use this crate as the shared instrumentation
layer for both the `jolt-core` reference path and the generated Bolt path. A
gate should run matching inputs through both paths, capture the same named spans,
and compare at least prove time, verify time, proof size, and peak RSS.

Required span families:

```
core.setup
core.prove
core.verify
bolt.setup
bolt.prove
bolt.commitment
bolt.commitment.batch
bolt.commitment.dory_commit
bolt.stage1 ... bolt.stage8
bolt.evaluate
bolt.evaluate.claims
bolt.evaluate.materialize_joint_polynomial
bolt.evaluate.joint_opening_hint
bolt.evaluate.dory_open
bolt.verify
bolt.verify.evaluation_state
bolt.verify.dory_verify
```

`setup_tracing` also records observed span names in-process so perf gates can
check the spans that actually ran, independent of whether Chrome trace output is
enabled.

Perf harnesses should live near the semantic oracle or CI job that owns their
protocol details. This crate owns the reusable measurement and tracing
primitives, not a protocol-specific benchmark runner.

### System Metrics (`monitor` feature)

- **`MetricsMonitor::start(interval_secs)`** — Spawns a background thread sampling CPU usage, memory, active cores, and thread count. Outputs structured `counters.*` fields for Perfetto postprocessing.

### CPU Profiling (`pprof` feature)

- **`pprof_scope!(label)`** — Creates a scoped CPU profiler guard that writes a `.pb` flamegraph on drop.
- **`PprofGuard`** — The underlying guard type (stub when `pprof` feature is off).

### Heap Flamegraphs (`allocative` feature)

- **`print_data_structure_heap_usage(label, data)`** — Logs heap size of `Allocative`-instrumented values.
- **`write_flamegraph_svg(flamegraph, path)`** — Renders an `allocative::FlameGraphBuilder` to an SVG file.

## Feature Flags

| Flag | Description |
|------|-------------|
| `monitor` | Background system metrics sampling (CPU, memory, cores) |
| `pprof` | Scoped CPU profiling via `pprof` with `.pb` output |
| `allocative` | Heap flamegraph generation from `allocative`-instrumented types |

## Dependency Position

```
tracing ─┐
tracing-chrome ─┤
tracing-subscriber ─┼─► jolt-profiling
memory-stats ─┤
sysinfo (opt) ─┤
pprof (opt) ─┤
allocative (opt) ─┘
```

Imported by host binaries and benchmarks. Library crates depend only on `tracing`.

## License

MIT OR Apache-2.0
