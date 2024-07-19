# Jolt EVM Verifier

This packages implements a verifier for the Jolt prover which can run in the EVM. 

## *WARNING THIS PACKAGE IS NEITHER COMPLETE NOR REVIEWED FOR SECURITY. DO NOT USE IN PRODUCTION.*

## Install 

To run the tests you must install both the rust dependencies and also those for solidity.

## Build

First build the rust integration scripts as they will be run to get examples for the integration tests.

```shell
$ cargo build --manifest-path script/Cargo.toml --release
```

Then we can build the solidity contracts with:


```shell
$ forge build --deny-warnings
```

## Test

You must run the tests with FFI as it is used to run the rust programs which get proofs for the integration tests

```shell
$ forge test --via-ir --ffi
```
