# Compact register-entry layout experiment

Date: 2026-07-31 EDT

## Question

Can the register read/write matrix use widths implied by its actual schedule
to reduce the first-bind old/new entry overlap without slowing Stage 4?

The candidate makes three representation changes:

1. cycle and address row indices become `u32`, with a constructor guard that
   requires fewer than `2^32` trace rows;
2. read and write lookup coefficients use separate index types; and
3. the write coefficient uses `u8` while the read coefficient remains `u16`.

The write lookup table is safe in one byte because the read table saturates
first. Starting from `(read, write) = (4, 2)` entries, the three lookup binds
produce:

```text
(4, 2) -> (16, 4) -> (256, 16) -> (65536, 256)
```

The prover dereferences both coefficient indices before the next bind.

## Fixed evaluator

- Machine: Apple M4 Max
- Control revision: `cb2c9dc6b1403151b573541dbd8f7f0e59df05d0`
- Candidate revision: `98090a25e6a8d6442634cf5f40642c1b150b67d9`
- Candidate tree: `7c0bcd2a5b7e5ea17c3c246a9c69e8dff9d899aa`
- Rust: `rustc 1.95.0 (59807616e 2026-04-14)`
- Physical one-hot `K = 256`
- `PERF_LOG_K_CHUNK=8`
- `PERF_LOOKUPS_RA_VIRTUAL_LOG_K_CHUNK=32`
- Release profile, successful proof verification required

Benchmark command:

```bash
/usr/bin/time -l env \
  PERF_LOG_T=26 \
  PERF_LOG_K_CHUNK=8 \
  PERF_LOOKUPS_RA_VIRTUAL_LOG_K_CHUNK=32 \
  PERF_TRACE=1 \
  cargo nextest run --release \
    -p jolt-prover-legacy --features akita \
    -E 'test(sha2_chain_akita_perf)' \
    --run-ignored all --no-capture --cargo-quiet
```

The same command with `PERF_LOG_T=22` supplies the discovery screen and with
`PERF_LOG_T=28` supplies the exact-target check.

## Metrics and gates

The layout test must establish these 64-bit Akita sizes:

```text
lookup cycle entry  48 -> 40 bytes
field cycle entry   80 -> 72 bytes
address entry       96 -> 88 bytes
```

The u8 and u16 lookup indices must agree through the three pre-dereference
rounds. All proof modes must verify.

Promotion requires:

- no complete Stage 4 regression in the repeated `T = 2^22` screen;
- repeated `T = 2^26` construction, bind, and Stage 4 spans to improve;
- an exact `T = 2^28` proof with zero swaps and no changed-path regression;
- no replacement trace-scaled allocation;
- standard/ZK Dory regressions, all Akita tests, scoped/workspace Clippy, and
  formatting to pass.
