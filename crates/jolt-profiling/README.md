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
