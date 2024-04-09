# Jolt

![imgs/jolt_alpha.png](imgs/jolt_alpha.png)

Just One Lookup Table.

## Resources

-   [Introducing Lasso and Jolt](https://a16zcrypto.com/posts/article/introducing-lasso-and-jolt/)
-   [Understanding Lasso and Jolt, from theory to code](https://a16zcrypto.com/posts/article/building-on-lasso-and-jolt/)
-   [Lasso paper](https://people.cs.georgetown.edu/jthaler/Lasso-paper.pdf)
-   [Jolt paper](https://people.cs.georgetown.edu/jthaler/Jolt-paper.pdf)

## Quickstart

For developers looking to build using Jolt, check out the [Quickstart guide](https://jolt.a16zcrypto.com/usage/quickstart.html).

For developers looking to contribute to Jolt, follow the instructions below.

## Installation

You will need Rust [nightly](./rust-toolchain). 
To compile the guest programs to RV32I, you will need to add the compilation target using rustup:

```rustup target add riscv32i-unknown-none-elf```

Finally, clone this repo:

```git clone git@github.com:a16z/jolt.git```

## Build

This repository uses workspaces, and each workspace can be built individually, e.g.

```cargo build -p jolt-core```

For faster incremental builds, use the `build-fast` profile:

```cargo build --profile build-fast jolt-core```

## Test

Unit and end-to-end tests for `jolt-core` can be run using the following command:

```cargo test -p jolt-core```


## Performance profiling

Jolt uses [tracing_chrome](https://crates.io/crates/tracing-chrome) for performance profiling. 

To generate a trace, run:

```cargo run --profile build-fast -p jolt-core trace --name sha3 --format chrome```

Where `--name` can be `sha2`, `sha3`, `sha2-chain`, or `fibonacci`. The corresponding guest programs can be found in the [`examples`](./examples/) directory. The benchmark inputs are provided in [`bench.rs`](./jolt-core/src/benches/bench.rs).

The above command will output a JSON file, e.g. `trace-1712455107389520.json`, which can be viewed in [Perfetto](https://ui.perfetto.dev/). 

## Acknowledgements

*This repository started as a fork of https://github.com/arkworks-rs/spartan. Original Spartan [code](https://github.com/microsoft/Spartan) by Srinath Setty.*

## Disclaimer

*This code is being provided as is. No guarantee, representation or warranty is being made, express or implied, as to the safety or correctness of the code. It has not been audited and as such there can be no assurance it will work as intended, and users may experience delays, failures, errors, omissions or loss of transmitted information. Nothing in this repo should be construed as investment advice or legal advice for any particular facts or circumstances and is not meant to replace competent counsel. It is strongly advised for you to contact a reputable attorney in your jurisdiction for any questions or concerns with respect thereto. a16z is not liable for any use of the foregoing, and users should proceed with caution and use at their own risk. See a16z.com/disclosures for more info.*
