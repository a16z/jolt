# Profiling Jolt

## Execution profiling

Jolt is instrumented using [`tokio-rs/tracing`](https://github.com/tokio-rs/tracing) for execution profiling.
We use the [`tracing_chrome`](https://github.com/thoren-d/tracing-chrome) crate to output traces in the format expected by [Perfetto](https://ui.perfetto.dev/).

To generate a trace, run e.g.

```cargo run --release -p jolt-core profile --name sha3 --format chrome```

Where `--name` can be `sha2`, `sha3`, `sha2-chain`, `fibonacci`, or `btreemap`. The corresponding guest programs can be found in the `examples` directory. The benchmark inputs are provided in `bench.rs`.

The above command will output a JSON file in the workspace rootwith a name `trace-<timestamp>.json`, which can be viewed in [Perfetto](https://ui.perfetto.dev/).

## Memory profiling

Jolt uses [allocative](https://github.com/facebookexperimental/allocative) for memory profiling.
Allocative allows you to (recursively) measure the total heap space occupied by any data structure implementing the `Allocative` trait, and optionally generate a flamegraph.
In Jolt, most sumcheck data structures implement the `Allocative` trait, and we generate a flamegraph at the start and end of stages 2-5 of the Jolt [DAG](../../how/architecture/architecture.md#jolt-as-a-dag) (see [`jolt_dag.rs`](https://github.com/a16z/jolt/blob/main/jolt-core/src/zkvm/dag/jolt_dag.rs)).

To generate allocative output, run:

```RUST_LOG=debug cargo run --release --features allocative -p jolt-core profile --name sha3 --format chrome```

Where, as above, `--name` can be `sha2`, `sha3`, `sha2-chain`, `fibonacci`, or `btreemap`.

The above command will log memory usage info to the command line and output multiple SVG files, e.g. `stage3_start_flamechart.svg`, which can be viewed in a web browser of your choosing.
