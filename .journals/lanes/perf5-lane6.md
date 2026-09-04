# PERF-5 lane 6 — stream tail

Date: 2026-09-03. Base: `dbe2a2f9e`. Fixture:
`fibonacci_2_18_blake3.bin`, `k = 32`, `N = 2^23`. The current honest
online baseline is **26.072 s**.

## Status

Implementation compiles and passes clippy. Tests and measurements are deferred
until lane 4 releases the machine, per the campaign's machine-sharing rule.
No `jolt-hyperkzg` files changed.

## Changes

1. Stage-A members publish the multilinears they already reduced to one value:
   T1 committed/wired columns, each CopyLink's fixed and helper columns, every
   T2 claimed column, and the Spartan witness carry. The key owns the mapping
   between those values and physical packed slots.
2. The only live columns without a member binding are T1's six verifier-key
   columns. They are evaluated with one row-equality table and typed sparse
   scans. The key enumerates every physical padding slot as zero. The term
   stage therefore receives a complete column vector without scanning the 26
   packed group polynomials.
3. Packed RLC now runs by parallel row blocks. Type dispatch occurs once per
   group and block; bit/u16/u32 sources stay typed; zero values, zero weights,
   and the key's padding slots are skipped. No group becomes a full-field
   polynomial for this pass.
4. T2 digit-link setup reads the row-major typed matrix once into its three
   owned working columns. This removes nine temporary full-field column vectors
   and the digit clone.

## Expected savings

| item | prior | expected after | expected saving |
|---|---:|---:|---:|
| column evaluations at `r_A` | 1.147 s | 0.02–0.08 s | **1.07–1.13 s** |
| packed RLC | 0.998 s | 0.35–0.65 s | **0.35–0.65 s** |
| member setup | 0.791 s aggregate | attribution pending | **0.02–0.06 s** |

The member bucket includes T1's precomputed first round, so its full 0.791 s
is not setup waste. Expected total saving: **1.4–1.9 s** before any lane-5b
4-ary-fold interaction.

## Deferred gates

1. Add a temporary, uncommitted `WRAP_PROOF_OUT` hook around
   `encode_to_vec(&wrapped, standard())` in the real fixture.
2. Under the wrapper gate lock, run `real_wrapper` at `dbe2a2f9e` and this
   branch with the same fixed setup secret; `cmp` the serialized proof files.
3. Remove the hook, run wrapper nextest, then run the locked feature-enabled
   real fixture with all tampers. Confirm **7,392 B payload / 7,529 B bincode /
   352 B statement** and unchanged verifier counts.
4. Record the tail split and honest online wall. Keep the RLC rewrite only if
   it beats the prior 0.998 s pass.

If lane 5b exposes a first-fold accumulation hook, the row-block RLC can feed
that hook without an intermediate pass. This branch leaves RLC materialized
once because concurrent `jolt-hyperkzg` edits are out of scope.

## Compile gates

| command | result |
|---|---|
| `cargo check -p jolt-wrapper --all-targets --features prover-fixtures` | pass |
| wrapper clippy, all targets, feature-enabled, warnings denied | pass |
