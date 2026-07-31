# D128 task-local rotation experiment

Date: 2026-07-31 EDT

Machine: Apple M4 Max

Accepted parent: Jolt `fa4912273`, Akita `a56b933c`

Target: forced-K256 SHA2-chain proof at exactly `T = 2^28`

## Question

Can the D128 root-opening path recover the speed of a fully expanded rotation
table without retaining its 232 MiB trace-wide allocation?

The parent keeps 256 trace blocks by 29 columns of compact `[i8; 128]`
challenges. Every contribution rotates a compact row and adds its nonzero
entries into the destination ring. This uses little memory, but repeatedly
decodes the same rotations across 4,096 positions in a position task.

The candidate may expand one trace block's rotations inside each active
position task:

```text
29 columns * 128 rotations * [i16; 128]
    = 950,272 bytes per active task
```

The table is overwritten for the next trace block and dropped when the task
finishes. At 16 workers the maximum new live scratch is 15,204,352 bytes
(14.5 MiB). The compact prepared challenges, committed polynomial, opening
hint, proof, and transcript are unchanged.

## Fixed evaluator

```bash
PERF_LOG_T=28 \
PERF_LOG_K_CHUNK=8 \
PERF_LOOKUPS_RA_VIRTUAL_LOG_K_CHUNK=32 \
PERF_TRACE=1 \
cargo nextest run --release \
  -p jolt-prover-legacy \
  --features akita \
  -E 'test(sha2_chain_akita_perf)' \
  --run-ignored all \
  --no-capture \
  --cargo-quiet
```

Peak RSS and process swaps are measured by wrapping the command with
`/usr/bin/time -l`.

## Promotion gate

Promote only if all of the following hold:

1. the exact D128/K256 accumulation probe improves by at least 20%;
2. full-trace `TracePackedOneHot::decompose_fold` improves by at least 20%;
3. total Akita opening improves by at least 1.5 seconds;
4. the proof verifies and the natural, forced-K256, and committed-program
   Akita end-to-end tests pass;
5. the analytical live-memory increase remains task-local, repeated
   measurement shows no structural RSS regression, and process swaps stay at
   zero;
6. D64 and other K values retain the existing path.

Reject if the gain comes from a different opening substage, if a trace-wide
expanded table survives, or if RSS growth is materially larger than the
bounded worker scratch.

