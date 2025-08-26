# Jolt

![imgs/jolt_alpha.png](imgs/jolt_alpha.png)

Just One Lookup Table.

Jolt is a zkVM (zero-knowledge virtual machine) for RISC-V, built to be the simplest, fastest, and most extensible general-purpose of its kind. This repository currently contains an implementation of Jolt for the RISC-V 32-bit Base Integer Instruction Set + M Standard Extension for Integer Multiplication and Division (RV32IM). _Contributors are welcome!_

## Resources

### Docs
[The Jolt Book](https://jolt.a16zcrypto.com/)
- 🚧 currently undergoing updates 🚧

### Papers

[Jolt: SNARKs for Virtual Machines via Lookups](https://eprint.iacr.org/2023/1217) \
Arasu Arun, Srinath Setty, Justin Thaler

[Twist and Shout: Faster memory checking arguments via one-hot addressing and increments](https://eprint.iacr.org/2025/105) \
Srinath Setty, Justin Thaler

[Unlocking the lookup singularity with Lasso
](https://eprint.iacr.org/2023/1216) \
Srinath Setty, Justin Thaler, Riad Wahby

### Blog posts
Initial launch:
- [Releasing Jolt](https://a16zcrypto.com/posts/article/a-new-era-in-snark-design-releasing-jolt/)
- [FAQ on Jolt's initial implementation](https://a16zcrypto.com/posts/article/faqs-on-jolts-initial-implementation/)

Updates:
- Nov 12, 2024 [blog](https://a16zcrypto.com/posts/article/jolt-an-update/) [video](https://a16zcrypto.com/posts/videos/an-update-on-jolts-development-roadmap/)
- Aug 18, 2025 (Twist and Shout upgrade) [blog](https://a16zcrypto.com/posts/article/jolt-6x-speedup/)

### Background
- [Proofs, Arguments, and Zero-Knowledge](https://people.cs.georgetown.edu/jthaler/ProofsArgsAndZK.pdf)

## Quickstart

> [!NOTE]
> Jolt is in alpha and is not suitable for production use at this time.

For developers looking to build using Jolt, check out the [Quickstart guide](https://jolt.a16zcrypto.com/usage/quickstart.html).

For developers looking to contribute to Jolt, follow the instructions below.

## Installation

You will need Rust [nightly](./rust-toolchain.toml).

If you have `rustup` installed, you do not need to do anything as it will
automatically install the correct toolchain and any additional targets on the
first `cargo` invocation.


Clone this repo:

```git clone git@github.com:a16z/jolt.git```

To check if `rustup` has picked the right version of Rust run `rustup show`
inside the cloned repository.

```cd jolt; rustup show```.

## Build

This repository uses workspaces, and each workspace can be built individually, e.g.

```cargo build -p jolt-core```

For faster incremental builds, use the `build-fast` profile:

```cargo build --profile build-fast -p jolt-core```

## Test

Unit and end-to-end tests for `jolt-core` can be run using the following command:

```cargo test -p jolt-core```

Examples in the [`examples`](./examples/) directory can be run using e.g.

```cargo run --release -p sha2-chain```

## Performance profiling

### Execution profiling

Jolt is instrumented using [tokio-rs/tracing](https://github.com/tokio-rs/tracing) for execution profiling.

To generate a trace, run e.g.

```cargo run --release -p jolt-core profile --name sha3 --format chrome```

Where `--name` can be `sha2`, `sha3`, `sha2-chain`, `fibonacci`, or `btreemap`. The corresponding guest programs can be found in the [`examples`](./examples/) directory. The benchmark inputs are provided in [`bench.rs`](./jolt-core/src/benches/bench.rs).

The above command will output a JSON file in the workspace rootwith a name `trace-<timestamp>.json`, which can be viewed in [Perfetto](https://ui.perfetto.dev/).

### Memory profiling

Jolt uses [allocative](https://github.com/facebookexperimental/allocative) for memory profiling.
Allocative allows you to (recursively) measure the total heap space occupied by a data structure implementing the `Allocative` trait, and optionally generate a flamegraph.
In Jolt, most sumcheck data structures implement the `Allocative` trait, and we generate a flamegraph at the start and end of stages 2-5 (see [`jolt_dag.rs`](https://github.com/a16z/jolt/blob/main/jolt-core/src/zkvm/dag/jolt_dag.rs)).

To generate allocative output, run:

```RUST_LOG=debug cargo run --release --features allocative -p jolt-core profile --name sha3 --format chrome```

Where, as above, `--name` can be `sha2`, `sha3`, `sha2-chain`, `fibonacci`, or `btreemap`.

The above command will log memory usage info to the command line and output multiple SVG files, e.g. `stage3_start_flamechart.svg`, which can be viewed in a web browser of your choosing.

## CI Benchmarking

We have enabled [benchmarking during CI](https://a16z.github.io/jolt/dev/bench/) to track performance changes over time in terms of prover runtime and peak memory usage.

## Acknowledgements

*This repository started as a fork of https://github.com/arkworks-rs/spartan. Original Spartan [code](https://github.com/microsoft/Spartan) by Srinath Setty.*

## Licensing

Jolt is dual licensed under the following two licenses at your discretion: the MIT License (see [LICENSE-MIT](https://github.com/a16z/jolt/blob/main/LICENSE-MIT)), and the Apache License (see [LICENSE-APACHE](https://github.com/a16z/jolt/blob/main/LICENSE-APACHE)).

Jolt is Copyright (c) a16z 2023. However, certain portions of the Jolt codebase are modifications or ports of third party code, as indicated in the applicable code headers for such code or in the copyright attribution notices we have included in the directories for such code.

## Disclaimer

*This code is being provided as is. No guarantee, representation or warranty is being made, express or implied, as to the safety or correctness of the code. It has not been audited and as such there can be no assurance it will work as intended, and users may experience delays, failures, errors, omissions or loss of transmitted information. Nothing in this repo should be construed as investment advice or legal advice for any particular facts or circumstances and is not meant to replace competent counsel. It is strongly advised for you to contact a reputable attorney in your jurisdiction for any questions or concerns with respect thereto. a16z is not liable for any use of the foregoing, and users should proceed with caution and use at their own risk. See a16z.com/disclosures for more info.*
