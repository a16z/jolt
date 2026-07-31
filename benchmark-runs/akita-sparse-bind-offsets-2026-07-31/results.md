# Sparse cycle-bind offset results

Date: 2026-07-31 EDT

Machine: Apple M4 Max

Implementation: Jolt `03d50f06f`

## Outcome

Accepted. `ReadWriteMatrixCycleMajor::bind` now stores cumulative input/output
offsets and derives each worker's disjoint slices from the two allocations.
It no longer materializes one immutable and one mutable fat-slice object per
bound-row group.

Across the promoted runs:

- cycle-major bind time improved 12.9% at `T = 2^22`, 9.4% at `T = 2^26`,
  and 10.8% at `T = 2^28`;
- Stage 4 improved 7.1%, 2.3%, and 4.4%, respectively;
- repeated `T = 2^26` maximum RSS fell by 525.55 MiB on average;
- the exact `T = 2^28` proof used 79.412 GiB and reported zero swaps.

## What changed

The control first computed `(input_len, output_len)` for every bound-row
group. It then built two more vectors by repeatedly splitting the input and
output buffers into fat slices before zipping those vectors in parallel.

The candidate converts the length pairs in place to cumulative
`(input_start, output_start)` offsets. A parallel indexed traversal uses the
next pair, or the total lengths for the final group, to recover the same
ranges. The input pointer addresses an immutable initialized allocation. The
output pointer addresses spare capacity partitioned into non-overlapping
ranges. Each worker invokes the unchanged merge routine, and the vector
length is set only after every range has been initialized.

No polynomial, arithmetic, ordering, protocol, transcript, or verifier code
changes.

## Analytical memory and corrected Stage 4 transition

For `G` bound-row groups:

```text
control metadata   = 3 * 16G = 48G bytes
candidate metadata =     16G = 16G bytes
removed             =     32G bytes
```

The largest instance is the first register cycle bind, where `G` is
approximately `T/2` on the SHA workload. Its metadata reduction is therefore
approximately `16T`:

| Trace length | First-register-bind reduction |
|---:|---:|
| `2^22` | 64 MiB |
| `2^26` | 1 GiB |
| `2^28` | 4 GiB |

The generic change also removes slice metadata from the RAM matrix and later
rounds, but those have fewer groups and do not add to the first-bind peak.

A temporary source probe measured the concrete Akita entry layouts:

```text
cycle-major with lookup coefficients  48 bytes
cycle-major with field coefficients   80 bytes
address-major                         96 bytes
```

It also measured the following register support at `T = 2^22`:

| Cycle binds completed | Entries | Entries / T | Representation |
|---:|---:|---:|---|
| 0 | 6,894,627 | 1.643807 | lookup |
| 1 | 5,509,920 | 1.313667 | lookup |
| 2 | 4,002,718 | 0.954322 | lookup |
| 3 | 2,880,952 | 0.686872 | lookup |
| 4 | 2,065,004 | 0.492335 | field |

The first bind therefore holds approximately
`48 * (1.643807 + 1.313667) = 141.96` bytes per cycle of old and new entries.
With the existing 8 B/cycle prefix table, the 40 B/cycle Stage 4 increment
and RAM-address state, and 142.258 B/cycle of common D128 state, the current
representative construction estimate is:

```text
control:   142.258 + 40 + 141.96 + 24 = 348.22 B/cycle = 87.05 GiB
candidate: 142.258 + 40 + 141.96 +  8 = 332.22 B/cycle = 83.05 GiB
```

This supersedes the retained-state-only 65.244 GiB Stage 4 row in the prior
RAM-sentinel write-up. It also explains much of the earlier gap between the
long-lived ownership model and process RSS. The calculation remains a
capacity estimate: allocator page residency determines how much of the
logical 4 GiB deletion appears in `ru_maxrss`.

## Integrated performance

### `T = 2^22`

The three parent traces contain two unrelated Stage 4 spikes, so their median
is the robust comparator.

| Metric | Parent median | Candidate mean (2 runs) | Change |
|---|---:|---:|---:|
| Prover | 4.703519 s | 4.705384 s | +0.04% |
| Cycle-major binds | 0.086857 s | 0.075628 s | **-12.9%** |
| Stage 4 | 0.167997 s | 0.156046 s | **-7.1%** |

Both proofs verified.

### `T = 2^26`

| Variant | Prover | Commitment | Cycle-major binds | Stage 4 | Maximum RSS |
|---|---:|---:|---:|---:|---:|
| Parent 1 | 44.667211 s | 16.148366 s | 1.049960 s | 2.256839 s | 33.288605 GiB |
| Parent 2 | 45.579603 s | 16.685763 s | 1.097432 s | 2.328985 s | 33.299362 GiB |
| Candidate 1 | 45.398628 s | 16.598814 s | 0.956095 s | 2.180810 s | 32.851547 GiB |
| Candidate 2 | 46.437120 s | 17.272988 s | 0.988747 s | 2.298482 s | 32.709961 GiB |
| **Parent mean** | **45.123407 s** | **16.417065 s** | **1.073696 s** | **2.292912 s** | **33.293983 GiB** |
| **Candidate mean** | **45.917874 s** | **16.935901 s** | **0.972421 s** | **2.239646 s** | **32.780754 GiB** |

Cycle-major binds improved by 9.4%, Stage 4 by 2.3%, and maximum RSS by
525.55 MiB. Complete proving moved by 0.79 seconds, while unchanged
commitment moved by 0.52 seconds in the same direction; the changed spans do
not regress. Both proofs verified with zero swaps.

### `T = 2^28`

| Metric | Parent mean (2 runs) | Candidate | Change |
|---|---:|---:|---:|
| Prover | 151.867537 s | 154.288338 s | +1.6% |
| Commitment | 56.654881 s | 59.430681 s | +4.9% |
| Cycle-major binds | 6.749553 s | 6.019282 s | **-10.8%** |
| Stage 4 | 12.122909 s | 11.589929 s | **-4.4%** |
| Maximum RSS | 79.321335 GiB | 79.412201 GiB | within prior range |

The unchanged commitment was 2.776 seconds slower than the parent mean,
larger than the 2.421-second whole-prover movement. The changed Stage 4 span
improved by 0.533 seconds. The proof verified and reported zero swaps.

The `T = 2^28` process high-water does not expose the 4 GiB logical deletion,
just as prior runs varied by several GiB under unchanged ownership. The
replicated `T = 2^26` RSS drop and the exact allocation formula support the
memory result; no stronger `ru_maxrss` claim is made.

## Validation

Passed:

- natural, forced-K256, and committed-program Akita `muldiv` proofs;
- standard and ZK Dory `muldiv` suites;
- all 49 `jolt-akita` tests;
- exact `T = 2^22`, `T = 2^26`, and `T = 2^28` proof verification;
- scoped legacy `host,akita` Clippy;
- workspace Clippy with `host`;
- workspace Clippy with `host,zk`;
- formatting and `git diff --check`.

## Retained traces

All files are in `benchmark-runs/perfetto_traces/`.

| Trace | Purpose | SHA-256 |
|---|---|---|
| `akita_22_rw_offsets.json` | discovery candidate A | `3f51cd52a4a901f8d6e5d8bccb5669797b04f703488afe7845e800d81d9fbd5c` |
| `akita_22_rw_offsets_repeat.json` | discovery candidate B | `bcb0866cd5414973c430ca88f0c8667c97ed64175cfe2cfca41b2cb2bde54344` |
| `akita_26_rw_offsets.json` | crossover candidate A | `fb6eab26759c9b0c7ad9d5b6ea55e04110d785088e091ce08dc7ab50890f7a5b` |
| `akita_26_rw_offsets_repeat.json` | crossover candidate B | `2324291304c8ee0226bb739ef55a5e35f9866b8a91d0fc3618498d86c11a41bf` |
| `akita_28_rw_offsets.json` | exact-target candidate | `ca844e429c00cb5346bbc12bf051d7f6e0333b2adf5c3cd1eaea9cca71fa09f0` |

## Next target

The remaining first-bind peak is the old/new entry overlap:
approximately 141.96 B/cycle for the measured SHA support. A compact initial
entry or a segmented/in-place first bind could remove materially more than
another metadata tweak, but it must preserve the parallel merge's time
advantage. That is the next experiment; shrinking the already necessary
8 B/cycle prefix table is secondary.
