# Sparse cycle-bind offset experiment

Date: 2026-07-31 EDT

## Question

Can `ReadWriteMatrixCycleMajor::bind` replace its per-row input and output
slice vectors with prefix offsets, reducing Stage 4 construction memory
without changing the sparse merge or slowing it?

The control creates three metadata vectors for `G` bound-row groups:

```text
row lengths                    16G bytes
immutable input slice objects  16G bytes
mutable output slice objects   16G bytes
```

The candidate keeps one `Vec<(usize, usize)>` of cumulative input/output
offsets, or `16G` bytes. The removed owner is exactly `32G` bytes at runtime.
For the first register bind on the SHA workload, `G` is approximately `T/2`,
so the target reduction is approximately `16T`: 1 GiB at `T = 2^26` and
4 GiB at `T = 2^28`.

## Fixed evaluator

- Machine: Apple M4 Max
- Control revision: `f529539c2c26bd58186918b33b81e43bc0b114f8`
- Candidate revision: `03d50f06f93c87a07d6a3a6ff65834a2f6081472`
- Candidate tree: `06ff90f957171715d6284ae01e029d14c2704641`
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

## Equivalence and safety obligations

1. The prefix table must describe the same input and output boundaries as
   the old sequence of `split_at` calls.
2. Input ranges must be immutable and disjoint; output ranges must be
   mutable, disjoint, and contained in the allocation's spare capacity.
3. Each range must call the unchanged `bind_rows` implementation, preserving
   row order, column order, coefficient arithmetic, and lookup-table updates.
4. The final initialized length must equal the sum of the dry-run output
   lengths.
5. Natural, forced-K256, and committed-program Akita proofs and standard/ZK
   Dory proofs must verify.

## Promotion gates

- repeated `T = 2^22` screens must not regress total cycle-bind or Stage 4
  time;
- repeated `T = 2^26` screens must preserve the timing signal, report zero
  swaps, and not raise process RSS;
- an exact `T = 2^28` proof must verify with zero swaps and no changed-path
  regression;
- no trace-scaled replacement allocation is permitted;
- scoped and workspace Clippy and formatting must pass.
