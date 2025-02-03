ZKLean Extractor
----------------

This program extracts a representation of the MLEs and constraints for each `JoltInstruction` and `LassoSubtable` suitable for the ZkLean library.

Running
=======

You can produce a lean file in the ZkLean source directory by cloning XXX and running the following command from the root of the Jolt repo:
```sh
cargo run -p zklean-extractor -- -f $PATH_TO_ZKLEAN_REPO/src/ZkLean/SubtableMles.lean
```

Testing
=======

You can run tests on the internal representation produced by the `zklean-extractor` executable by running the following from the root of the Jolt repo:
```sh
cargo test -p zklean-extractor
```

This tests use the [`proptest`](https://docs.rs/proptest/latest/proptest/index.html) library to ensure that extracting a representation of each constraint and MLE and executing it produces the same result as computing the constraint or MLE on its own. By default we run proptest for 256 iterations, however this can be changed by running
```sh
PROPTEST_CASES=$DESIRED_NUMBER_OF_ITERS cargo test -p zklean-extractor
```
