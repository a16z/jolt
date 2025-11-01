ZKLean Extractor
----------------

This program extracts a Lean4 representation of the Jolt frontend suitable for the [ZkLean](https://github.com/GaloisInc/zk-lean) library. This allows for formal reasoning in Lean about the correctness of the frontend. The R1CS constraints and instruction lookups are all taken automatically from the Jolt codebase, so any changes will be reflected in the Lean code.

Running
=======

This tool allows you to produce a package containing a ZkLean representation of the Jolt VM by running the following command from the root of the Jolt repo:
```sh
cargo run --release -p zklean-extractor -- -o -p ./jolt-zk-lean
```
This will generate a Lean package in `./jolt-zk-lean`, creating this directory if it doesn't exist, and overwriting the files in it if it does. To use a different output directory, simply use that directory for the argument to `-p` instead. Assuming you have a valid ZkLean installation, you can then compile the Lean module by running the following:
```sh
cd ./jolt-zk-lean/src
lake build
```
In addition to the `Jolt` module, the generated lean will contain a `Tests` module that checks to ensure that the extracted lookup-table MLEs evaluate in the same way as the corresponding Rust functions. To run these, you can run `lake test` in the same directory as above.

For additional tool options, you can view the tool's help message by running the following from the root directory of the Jolt repo:
```sh
cargo run --release -p zklean-extractor -- -h
```

Testing
=======

You can run tests on the internal representation produced by the `zklean-extractor` executable by running the following from the root of the Jolt repo:
```sh
cargo test -p zklean-extractor
```

These tests use the [`proptest`](https://docs.rs/proptest/latest/proptest/index.html) library to ensure that extracting a representation of each constraint and MLE and executing it produces the same result as computing the constraint or MLE on its own. By default we run proptest for 256 iterations, however this can be changed on the command line. For example, to run 512 iterations instead, you can run
```sh
PROPTEST_CASES=512 cargo test -p zklean-extractor
```
