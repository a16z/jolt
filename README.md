# Jolt

![imgs/jolt_alpha.png](imgs/jolt_alpha.png)

Just One Lookup Table.

Jolt is a zkVM (zero-knowledge virtual machine) for RISC-V, built to be the simplest, fastest, and most extensible general-purpose of its kind. This repository currently contains an implementation of Jolt for the RISC-V 32-bit Base Integer instruction set (RV32I). _Contributors are welcome!_

The Jolt [paper](https://eprint.iacr.org/2023/1217.pdf) was written by Arasu Arun, Srinath Setty, and Justin Thaler. 

## Resources

- [Docs](https://jolt.a16zcrypto.com/) (The Jolt Book)
- Blog posts
  - [Accelerating the world computer: Implementing Jolt, a new state-of-the-art zkVM](https://a16zcrypto.com/posts/article/accelerating-the-world-computer-implementing-jolt)
  - [Building Jolt: A fast, easy-to-use zkVM](https://a16zcrypto.com/posts/article/building-jolt/)
  - [FAQs on Jolt’s initial implementation](https://a16zcrypto.com/posts/article/faqs-on-jolts-initial-implementation)
  - [A new era in SNARK design: Releasing Jolt](https://a16zcrypto.com/posts/article/a-new-era-in-snark-design-releasing-jolt)
  - [Introducing Lasso and Jolt](https://a16zcrypto.com/posts/article/introducing-lasso-and-jolt/)
  - [Understanding Lasso and Jolt, from theory to code](https://a16zcrypto.com/posts/article/building-on-lasso-and-jolt/)
- Papers
  - [Lasso paper](https://eprint.iacr.org/2023/1216.pdf)
  - [Jolt paper](https://eprint.iacr.org/2023/1217.pdf)

## Quickstart

> [!NOTE]
> Jolt is in alpha and is not suitable for production use yet.

For developers looking to build using Jolt, check out the [Quickstart guide](https://jolt.a16zcrypto.com/usage/quickstart.html).

For developers looking to contribute to Jolt, follow the instructions below.

## Installation

You will need Rust [nightly](./rust-toolchain.toml).

If you have `rustup` installed, you do not need to do anything as it will
automatically install the right toolchain and install additional target on the
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

Jolt uses [tracing_chrome](https://crates.io/crates/tracing-chrome) for performance profiling. 

To generate a trace, run:

```cargo run --profile build-fast -p jolt-core trace --name sha3 --format chrome```

Where `--name` can be `sha2`, `sha3`, `sha2-chain`, or `fibonacci`. The corresponding guest programs can be found in the [`examples`](./examples/) directory. The benchmark inputs are provided in [`bench.rs`](./jolt-core/src/benches/bench.rs).

The above command will output a JSON file, e.g. `trace-1712455107389520.json`, which can be viewed in [Perfetto](https://ui.perfetto.dev/). 

## Acknowledgements

*This repository started as a fork of https://github.com/arkworks-rs/spartan. Original Spartan [code](https://github.com/microsoft/Spartan) by Srinath Setty.*

## Disclaimer

*This code is being provided as is. No guarantee, representation or warranty is being made, express or implied, as to the safety or correctness of the code. It has not been audited and as such there can be no assurance it will work as intended, and users may experience delays, failures, errors, omissions or loss of transmitted information. Nothing in this repo should be construed as investment advice or legal advice for any particular facts or circumstances and is not meant to replace competent counsel. It is strongly advised for you to contact a reputable attorney in your jurisdiction for any questions or concerns with respect thereto. a16z is not liable for any use of the foregoing, and users should proceed with caution and use at their own risk. See a16z.com/disclosures for more info.*
