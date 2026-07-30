# Packed RA source experiment

Date: 2026-07-29 EDT

## Decision

Can the K256 packed one-hot byte buffer replace the retained
`Vec<RaIndices>` without slowing the Akita prover?

## Contract

- Candidate: share the existing row-major byte allocation between the Akita
  commitment/opening source and the Stage 6/7 RA consumers. Add a one-bit RAM
  presence map so a real RAM lane zero remains distinct from an absent RAM row.
- Frozen surface: K256, protocol messages, committed polynomial values,
  sumcheck scheduling, benchmark harness, and trace workload.
- Expected memory effect: replace 54 B/cycle with 1/8 B/cycle while the RA
  source is live, saving 53.875 B/cycle: 3.367 GiB at `2^26` and 13.469 GiB at
  `2^28`.
- Expected performance effect: neutral or faster because consumers read the
  same native bytes from a denser source. No full-row widening adapter is
  permitted in a hot loop.
- Falsifying outcome: proof mismatch, a focused Stage 6/7 regression above 2%,
  or a full-prover regression above the established 0.48-second noise band.

## Measurement

- Budget: one direct-source implementation, one correctness/parity pass, up to
  three `2^22` screens, and one interleaved `2^26` candidate/control pair if
  the screen passes.
- Primary metrics: maximum RSS, Stage 6a, Stage 6b, Stage 7, packed commit,
  opening, and total prove time.
- Guards: K256 printed by the harness, proof verification, zero swaps, host
  muldiv, host+ZK muldiv, formatting, and clippy.
- Known confounds: system-wide thermal/load drift, untouched phase noise, and
  RSS phase switching. Target spans and exact retained bytes take precedence
  over a one-shot whole-prover movement.

## Runs

Append one line per run as:

`run | revisions | size | key metrics | verdict`

`control | d4ec43f67 | 2^22 | prove 6.486 s; S6a+S6b+S7 620.881 ms; max RSS 14.792 GB | baseline`

`ra-v1-a | working tree | 2^22 | prove 6.662 s; S6a+S6b+S7 647.648 ms; RSS unavailable (sandboxed time(1)) | discard as compile-adjacent/noisy`

`ra-v1-b | working tree | 2^22 | prove 6.476 s; S6a+S6b+S7 635.301 ms; max RSS 14.482 GB; swaps 0 | inconclusive: memory passes, focused guard +2.32%`

`ra-v1-c | working tree | 2^22 | prove 6.442 s; S6a+S6b+S7 635.229 ms; max RSS 14.478 GB; swaps 0 | reject v1 accessor: focused guard +2.31% repeated`

## V2 accessor follow-up

The retained representation is validated; only its row-access implementation
may change. V2 may hoist RAM presence checks and remove redundant dynamic
bounds checks, but may not add a widened cache or change the evaluator.

- Budget: one accessor-only tuning pass, one focused same-binary
  packed-versus-fixed-width microbenchmark, and up to two `2^22` screens.
- Promotion still requires the original focused and whole-prover guards.

`ra-v2-micro | working tree | 2^20 | steady fixed 12.21–12.59 ms; packed 12.18–13.28 ms | observed: near parity, no stable win`

`ra-v2-layout-check | working tree | 2^22 | constructor rejected noncontiguous semantic ranges before prove | crash: false adjacency invariant`

`ra-v2-a | working tree | 2^22 | prove 6.575 s; S6a+S6b+S7 685.286 ms; max RSS 14.477 GB; swaps 0 | discard as noisy: Stage 6a outlier`

`ra-v2-b | working tree | 2^22 | prove 6.481 s; S6a+S6b+S7 627.539 ms; max RSS 14.594 GB; swaps 0 | keep for 2^26 validation: +1.07% focused, -0.07% total`

V2 proceeds to the target-scale check. The repeated pushforward spans remain
slower than the fixed-width source at `2^22`, but denser Stage 6b gathers
offset the cost and the predeclared aggregate guard passes in the clean run.
The `2^26` run is decisive; no speedup is claimed from the screen.

`ra-v2-target | working tree | 2^26 | prove 66.683 s; S6a+S6b+S7 9.014 s; max RSS 42.719 GB; swaps 0 | reject v2: exact memory win, focused guard +5.52%`

The target-scale control is `mem-defer-delta-2e26-b`: prove 67.184 s,
S6a+S6b+S7 8.543 s, and maximum RSS 46.457 GB. V2 removes 3.737 GB observed
(3.616 GB predicted) but adds 0.471 s to the focused aggregate. The lower
whole-prover time is phase noise and is not evidence of a speedup.

## V3 row-access follow-up

V3 replaces the bitset with one RAM-validity byte per cycle and makes the
borrowed row carry only a source reference and row offset. This trades 0.875
B/cycle of the v2 saving for cheaper access:

- Expected saving: 53 B/cycle, 3.3125 GiB at `2^26` and 13.25 GiB at `2^28`.
- Budget: one implementation, one `2^22` screen, then one `2^26` run only if
  the focused guard passes.
- All original correctness and performance guards remain unchanged.

`ra-v3-screen | working tree | 2^22 | prove 6.396 s; S6a+S6b+S7 0.602 s; max RSS 14.619 GB; swaps 0 | promote: -3.04% focused, -1.38% total`

`ra-v3-target | working tree | 2^26 | prove 67.052 s; S6a+S6b+S7 9.206 s; max RSS 42.911 GB; swaps 0 | reject: focused guard +7.77%`

## Verdict

Reject the shared packed RA source. It reliably removes the expected storage:
the v3 target is 3.546 GB below the 46.457 GB control versus 3.557 GB
predicted. It also reliably slows the target-scale Stage 6/7 RA scans and
gathers, even though unrelated phase movement leaves total prove time flat.

The small screen transferred poorly: its focused aggregate improved 3.04%,
while the same aggregate regressed 7.77% at `2^26`. No candidate code is
retained. The negative result and all traces are retained before moving to
the next memory target.
