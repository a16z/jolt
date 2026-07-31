# Akita D128 A-ring reuse result

Date: 2026-07-30 EDT

## Result

Rejected. Processing two aligned trace blocks in one task made the exact
`T = 2^28`, `K = 256`, D128 root accumulation slower:

| Variant | Root accumulation | Commit | Evaluation proof | Prover | Peak RSS | Swaps |
|---|---:|---:|---:|---:|---:|---:|
| Accepted narrow-D128 parent | 82.49 s | 82.80 s | 32.92 s | 193.70 s | 81.080 GiB | 0 |
| Two-block batch | 87.01 s | 87.31 s | 32.21 s | 196.13 s | 81.653 GiB | 0 |

The intended schedule ran: the trace reports 128 tasks, each covering two of
the 256 trace blocks. The proof verified. Root accumulation nevertheless
regressed by 5.48%, opposite the required 5% improvement.

## Interpretation

The experiment did reuse an A ring in the innermost loop, but that load
reduction did not translate into less elapsed time. The most likely
explanation is that A rings were already being shared effectively through
the cache hierarchy across concurrent one-block tasks. In contrast, two
live 58 KiB destination accumulators enlarge each task's write-heavy working
set and weaken L1 locality. Halving the decoded-row tile held that temporary
storage constant, so the added destination state is the important change.

Batch four was not tested. It would further enlarge the live accumulator
footprint after batch two had already failed decisively.

The candidate source was restored. No prover change was landed.

## Artifacts

- Trace: `benchmark-runs/perfetto_traces/akita_28_d128_block2.json`
- Raw log: `trial-001-block-batch2.log`
- Parsed trace: `trial-001-block-batch2.analysis`
- Frozen experiment definition: `run.json`, `run.sha256`, and `contract.md`
