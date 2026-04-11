# Task 10: Remove Domain-Specific Debug Instrumentation

## Status: TODO

## Anti-Pattern
`runtime.rs` contains ~100 lines of `#[cfg(debug_assertions)]` blocks with domain-specific knowledge:
- Batch index checks (`*batch == 4 && inst_idx == 0` for BytecodeReadRaf)
- Polynomial name awareness (`PolynomialId::BytecodeReadRafF(s)`)
- Dot product diagnostics for specific polynomial pairs
- Per-instance dumping for "Booleanity debugging"

This instrumentation was useful during transcript debugging but:
1. Couples the runtime to specific protocol stages
2. Will break/mislead when stage numbering changes
3. Adds noise to the code

## What to Remove
All `#[cfg(debug_assertions)]` blocks in `runtime.rs` that reference specific:
- Batch indices
- PolynomialId variants by name
- Protocol-specific diagnostic computations

## What to Keep
Generic debug logging that doesn't reference specific protocol elements:
- Round/instance index logging
- Claim value logging
- Buffer size logging

## Test
```bash
cargo clippy -p jolt-zkvm --message-format=short -q --all-targets -- -D warnings
cargo nextest run -p jolt-equivalence transcript_divergence --cargo-quiet
```

## Risk: None
Debug instrumentation has no effect on release builds and removing it doesn't change behavior.

## Dependencies: Tasks 05, 08 (do this after the big refactor so we don't remove instrumentation we need during debugging)
