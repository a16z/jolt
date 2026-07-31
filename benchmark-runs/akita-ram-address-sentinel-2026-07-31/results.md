# Packed sparse RAM-address results

Date: 2026-07-31 EDT

Machine: Apple M4 Max

Implementation: Jolt `e78c5d738`

## Outcome

Accepted. Stage 4 RAM value checking and the Stage 5 RAM-RA claim reduction
now store missing addresses as an out-of-domain integer sentinel. This
removes 8 bytes per trace row from each stage-local address stream without a
protocol, transcript, claim, or verifier change.

At `T = 2^26`, two candidate runs reduced maximum RSS by 497.24 MiB on
average, close to the exact 512 MiB owner reduction. The combined Stage 4/5
time improved by 6.2%. At `T = 2^28`, the two candidate proofs completed
without swaps and the combined Stage 4/5 mean improved by 1.1%.

## Representation and equivalence

The old representation was:

```text
Vec<Option<usize>>: Some(k) for a RAM access, None otherwise
```

The candidate stores:

```text
Vec<usize>: k for a RAM access, K otherwise
```

Every valid remapped address is strictly less than `K`. `RaPolynomial`
therefore interprets only `K` as absent. The Stage 5 scans perform the same
test before indexing the address equality table. Unexpected values other
than the sentinel still reach the bounds check and fail as they did before.

For every row, both representations contribute either
`eq(r_address, k)` for the same address `k` or zero. All subsequent binds and
claims are consequently identical. A differential test checks every binding
state and the final claim in both binding orders.

## Analytical memory

On the measured 64-bit target:

```text
size_of::<Option<usize>>() = 16
size_of::<usize>()         = 8
reduction                  = 8T bytes per affected stage
```

| Trace length | Reduction in Stage 4 | Reduction in Stage 5 |
|---:|---:|---:|
| `2^22` | 32 MiB | 32 MiB |
| `2^26` | 512 MiB | 512 MiB |
| `2^28` | 2 GiB | 2 GiB |

The columns are phase-local, so the peak reduction is 8 B/cycle, not
16 B/cycle.

Using the retained-state D128/K256 model available for this experiment:

| Window | Before | After | After, B/cycle |
|---|---:|---:|---:|
| Stage 4, representative SHA-2 density | 67.244 GiB | 65.244 GiB | 260.978 |
| Stage 5 transition after compact buckets | 67.059 GiB | 65.059 GiB | 260.24 |

This table counts retained objects but not the old cycle-major bind's
construction overlap. The immediate follow-up audit found that its first
register bind simultaneously held old and new entries plus three
per-bound-row metadata vectors. That transition, and the accepted reduction
from roughly 87.05 to 83.05 GiB, are derived in
[`../akita-sparse-bind-offsets-2026-07-31/results.md`](../akita-sparse-bind-offsets-2026-07-31/results.md).
The 65.244 GiB row should therefore not be read as the complete Stage 4
construction ceiling.

## Integrated results

### `T = 2^22` discovery screen

| Variant | Runs | Prover mean | Stage 4 mean | Stage 5 mean | Stage 4+5 |
|---|---:|---:|---:|---:|---:|
| Control | 2 | 4.643832 s | 0.155551 s | 0.311796 s | 0.467347 s |
| Candidate | 3 | 4.721781 s | 0.184155 s | 0.338924 s | 0.523079 s |

This small screen was noisy and negative. Separate transient spikes appeared
in Stage 4 in the first candidate and Stage 5 in the third; the middle
candidate's combined span was 0.475571 seconds, 1.8% above the control mean.
The constructors became cheaper, but the 32 MiB phase-local deletion was too
small to establish a complete-stage win. The candidate advanced only to the
predeclared `T = 2^26` crossover check.

### `T = 2^26`

| Variant | Prover | Stage 4 | Stage 5 | Stage 4+5 | Maximum RSS |
|---|---:|---:|---:|---:|---:|
| Control 1 | 45.519607 s | 2.448522 s | 4.549971 s | 6.998493 s | 33.790161 GiB |
| Control 2 | 46.868442 s | 2.515552 s | 4.806450 s | 7.322002 s | 33.768982 GiB |
| Candidate 1 | 44.667211 s | 2.256839 s | 4.390855 s | 6.647694 s | 33.288605 GiB |
| Candidate 2 | 45.579603 s | 2.328985 s | 4.451792 s | 6.780777 s | 33.299362 GiB |
| **Control mean** | **46.194024 s** | **2.482037 s** | **4.678211 s** | **7.160248 s** | **33.779572 GiB** |
| **Candidate mean** | **45.123407 s** | **2.292912 s** | **4.421323 s** | **6.714235 s** | **33.293983 GiB** |

Stage 4 improved by 7.6%, Stage 5 by 5.5%, and their combined mean by 6.2%.
Maximum RSS fell by 497.24 MiB. The 14.76 MiB difference from the exact
512 MiB allocation is ordinary process-high-water noise. Both proofs
verified and reported zero swaps.

The changed internal spans also show the expected mechanism. The RAM-RA
initializer fell from 47.865 to 38.041 ms, its two address scans from
25.872 to 18.995 ms, and its challenge ingestion from 15.195 to 9.916 ms.
RAM-value initialization improved slightly; its round computation was 3.1%
slower, but the full Stage 4 result improved.

### `T = 2^28`

| Variant | Prover | Commitment | Stage 4 | Stage 5 | Stage 4+5 | Maximum RSS |
|---|---:|---:|---:|---:|---:|---:|
| Control | 152.655506 s | 57.351768 s | 12.377535 s | 14.503752 s | 26.881287 s | 75.461227 GiB |
| Candidate 1 | 155.243591 s | 58.979631 s | 12.153528 s | 14.899256 s | 27.052784 s | 78.368057 GiB |
| Candidate 2 | 148.491482 s | 54.330131 s | 12.092290 s | 14.015971 s | 26.108261 s | 80.274612 GiB |
| **Candidate mean** | **151.867537 s** | **56.654881 s** | **12.122909 s** | **14.457614 s** | **26.580523 s** | **79.321335 GiB** |

The candidate mean improved Stage 4 by 2.1%, Stage 5 by 0.3%, their combined
time by 1.1%, and complete proving by 0.5%. Both proofs verified and reported
zero swaps.

The raw `ru_maxrss` points do not expose the 2 GiB phase-local deletion.
Recent accepted-parent measurements had already ranged from 74.937 to
80.141 GiB at `2^28`, and the candidate points remain inside that range. In
contrast, the repeated `T = 2^26` high-water landed in the affected window
and tracked the analytical deletion to within 15 MiB. The defensible memory
claim is therefore the exact live-owner reduction plus the replicated
`T = 2^26` process-peak result, not a claimed `T = 2^28` RSS delta.

## Validation

Passed:

- sentinel-versus-`Option` RA equivalence in both binding orders;
- all 49 `jolt-akita` tests;
- natural, forced-K256, and committed-program Akita `muldiv` proofs;
- standard and ZK Dory `muldiv` suites;
- exact `T = 2^22`, `T = 2^26`, and `T = 2^28` proof verification;
- scoped `jolt-akita` and legacy `host,akita` Clippy;
- workspace Clippy with `host`;
- workspace Clippy with `host,zk`;
- formatting and `git diff --check`.

## Retained traces

All files are in `benchmark-runs/perfetto_traces/`.

| Trace | Purpose | SHA-256 |
|---|---|---|
| `akita_22_ram_sentinel.json` | discovery candidate A | `89b2aee01662f8a5289296624679f29d39172c5572fedaf25079216caf84cfd3` |
| `akita_22_ram_sentinel_repeat.json` | discovery candidate B | `e76c1844424724e829dc90060f5ee19ca928e434f5bb9e77fdaf24cf0519be59` |
| `akita_22_ram_sentinel_third.json` | discovery candidate C | `ee402022b145b7a2969d7880ddb3c6d891dc3d643894ac9ab903b765d3f3aeb3` |
| `akita_26_ram_sentinel.json` | crossover candidate A | `cde07b27c8c1a41e0efce72fb529b3aa216c31c8e70b2f7d5ba5212548ef4188` |
| `akita_26_ram_sentinel_repeat.json` | crossover candidate B | `c6ec3b981025af4c73409abd3d42354583652f860c3f3bd66b1fb681090aab17` |
| `akita_28_ram_sentinel.json` | exact-target candidate A | `cfa45aebfe9188879c4ecd407e10adbc4bef217b1d234b3f5de597571d3e4626` |
| `akita_28_ram_sentinel_repeat.json` | exact-target candidate B | `c576c82bc646e860aad9769c9d87d87ddd3af9368d5f660ead71fbff310b4f64` |
