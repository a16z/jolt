# Baseline Results (2026-04-11)

## jolt-equivalence test suite
- **14 tests run**: 13 passed, 1 failed, 8 skipped
- **Failed**: `transcript_divergence` — diverges at op #1439 (start of stage 6, not yet implemented)
- **Last matching op**: #1438

## Key number to maintain
After each refactoring task, `transcript_divergence` must still match through op #1438.
If the divergence point regresses (e.g., to op #1200), we broke something.

## Test command
```bash
cargo nextest run -p jolt-equivalence --cargo-quiet 2>&1 | tail -30
```
