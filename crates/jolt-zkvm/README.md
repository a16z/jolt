# jolt-zkvm

Top-level zkVM prover and verifier orchestration for the Jolt proving system.

Part of the [Jolt](https://github.com/a16z/jolt) zkVM.

## Overview

This crate composes all Jolt sub-crates into a complete proving system for RISC-V (RV64IMAC) execution traces. It is currently a stub under active development on the `refactor/crates` branch.

Once complete, it will provide:
- Witness generation from execution traces
- Streaming polynomial commitment via Dory
- Batched sumcheck proving across all protocol stages
- Opening proof accumulation and verification
- Optional zero-knowledge via BlindFold

## Dependency Position

```
jolt-field ─┐
jolt-poly  ─┤
jolt-transcript ─┤
jolt-crypto ─┤
jolt-sumcheck ─┼─► jolt-zkvm
jolt-openings ─┤
jolt-spartan ─┤
jolt-dory ─┤
jolt-blindfold ─┤
jolt-instructions ─┘
```

`jolt-zkvm` sits at the top of the dependency DAG, depending on all other `jolt-*` crates.

## Feature Flags

None yet. Feature flags will be migrated from `jolt-core` as the refactor progresses.

## License

MIT
