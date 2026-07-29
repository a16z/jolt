# Stage-7 cached RA-index reuse

Date: 2026-07-29 EDT  
Parent: `e3a81febee37e0209a1a192435e932c4d88f60cb`

## Result

Keep. At `2^26`, reusing the RA indices already retained by the packed trace
cache reduced stage 7 from 1.841 s to 0.628 s (65.9%). In the adjacent traced
pair, whole proving fell from 66.94 s to 65.96 s, a 0.98 s reduction.

| Run | K | Prove | `prove_packed` | Stage 7 | RA `G` build | Verify |
|---|---:|---:|---:|---:|---:|---:|
| Parent, `2^25` | 256 | 32.10 s | 32.079 s | 0.853 s | 0.809 s | 100.10 ms |
| Candidate, `2^25` | 256 | 33.75 s | 33.733 s | 0.333 s | 0.289 s | 104.76 ms |
| Parent, adjacent `2^26` | 256 | 66.97 s | 66.941 s | 1.841 s | 1.737 s | 197.54 ms |
| Candidate, adjacent `2^26` | 256 | 66.00 s | 65.961 s | 0.628 s | 0.524 s | 203.00 ms |

The `2^25` whole-prover result was confounded by a 1.14 s increase in the
unrelated commitment span. The named Stage-7 span supplied the screen signal;
the adjacent `2^26` pair supplied the promotion result. The full-size Stage-7
saving was 1.213 s, and 0.980 s remained end to end.

## Change

`JoltOneHotTraceRows` already stores one `RaIndices` record per cycle. Stage 6
uses this vector, but stage 7 previously walked the original trace again,
redoing lookup decoding, bytecode-PC lookup, and RAM-address remapping before
building the same pushforward tables. The candidate retains one `Arc` handle
through stage 6 and passes its slice to
`compute_all_G_from_ra_indices` in stage 7.

This does not change a polynomial, claim, challenge, transcript message, or
proof encoding. For each cycle `j` and RA polynomial `i`, both paths supply the
same cached lane `ra_indices[j].get_index(i, params)` to the same
`compute_all_G_impl`; only the derivation of that lane is removed.

## Validation

- Natural-K16, forced-K256, and committed-program Akita muldiv proofs passed.
- The full `2^25` and `2^26` benchmark proofs verified.
- The restored candidate diff hash matched the originally tested candidate:
  `f30ed93ec65b2782fcf6ddd08f1285c12e8f1180f1b7cb1c1bf5848012cb7d68`.
- Peak RSS was not captured: sandboxed `/usr/bin/time -l` could not read the
  required system counter, and these traces contain no monitor counter events.
  The change retains an additional `Arc` handle but allocates no additional
  RA-index vector.

## Traces

The decisive comparison traces are:

- `benchmark-runs/perfetto_traces/akita-s3-parent-adjacent-2e26.json`
- `benchmark-runs/perfetto_traces/akita-s3-candidate-2e26.json`

The screen traces are retained locally as:

- `benchmark-runs/perfetto_traces/akita-s3-parent-2e25.json`
- `benchmark-runs/perfetto_traces/akita-s3-candidate-2e25.json`

The invalid K16 screen is retained as
`benchmark-runs/perfetto_traces/s3-parent-k16-2e22-excluded.json`.
