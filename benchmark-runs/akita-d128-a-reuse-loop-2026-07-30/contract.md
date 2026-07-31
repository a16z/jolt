# Akita D128 A-ring reuse experiment

Date: 2026-07-30 EDT

## Question

Does processing two aligned D128 trace blocks in one Rayon task reduce the
`T = 2^28` root accumulation by at least 5% without changing the proof or
the global analytical memory peak?

The accepted narrow-accumulator parent has a final root span of 82.49
seconds. The target geometry has 256 trace blocks per semantic column, one
part per block, three A ranks, and 29 active columns.

## Mechanism

Every trace block uses the same A matrix positions. The current task order
streams those positions again for each block. A batch of two blocks can load
one A ring, apply it to both blocks' destinations, and then advance to the
next position.

Two narrow D128 rank accumulators occupy 116 KiB in total, equal to the old
one-block wide accumulator. The decoded-row tile is halved from 8192 to 4096
rows for each block so the total lane and mask storage per task does not
grow. The total reduced partial output across all tasks is unchanged.

## Acceptance

The primary metric is `trace_onehot_commit_accumulate`. Batch two must:

- improve the exact target span by at least 5%;
- verify the proof;
- keep the global analytical peak unchanged;
- stay below 90 GiB RSS with zero swaps;
- avoid a reproducible 3% regression in unaffected spans.

Batch four is tested only if batch two passes. A differential unit test must
compare the batched block result against independently accumulated blocks,
including an odd final batch.

The evaluator, trace parser, workload, D128 schedule, and accepted narrow
accumulator are frozen. Candidate code is restored on rejection and committed
separately on acceptance.
