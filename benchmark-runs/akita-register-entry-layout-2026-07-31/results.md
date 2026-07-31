# Compact register-entry layout results

Date: 2026-07-31 EDT

Machine: Apple M4 Max

Implementation: Jolt `98090a25e`

## Outcome

Accepted. Register read/write entries now use `u32` rows and an 8-bit write
lookup index until coefficient dereferencing. Concrete 64-bit Akita layouts
fall from 48 to 40 bytes for lookup-backed cycle entries, 80 to 72 bytes for
field-backed cycle entries, and 96 to 88 bytes for address-major entries.

At the promoted scales:

- `T = 2^26` matrix construction improved 19.3%, cycle binds 7.9%, and
  Stage 4 7.0% across two runs;
- `T = 2^28` construction improved 8.8%, binds 13.2%, and Stage 4 10.4%;
- every proof reported zero swaps.

## Schedule invariant

Read and write coefficients previously shared a 16-bit index type. The read
table starts with four values and the write table with two. Each bind squares
the table size. After three rounds they contain 65,536 and 256 values,
respectively. The existing phase transition detects read-table saturation
and dereferences both indices before another bind.

The candidate therefore stores the write index in `u8` without truncation.
A unit test exhaustively composes every index pair through those three rounds
and compares it with the original `u16` representation.

Rows use `u32`. The matrix constructor now rejects traces with `2^32` or more
rows. This is an internal prover capacity bound, not a change to the
polynomial, transcript, claims, or verifier. It covers the `2^28` target with
a factor-of-16 row margin.

## Analytical memory

The measured first-register-bind support is:

```text
E0 / T = 1.643807
E1 / T = 1.313667
```

Both the old and new buffers use the lookup-backed entry type during that
bind. The exact layout delta for the measured support is:

```text
8 * (E0 + E1) / T = 23.6598 B/cycle
```

| Trace length | First-bind entry reduction |
|---:|---:|
| `2^22` | about 94.64 MiB |
| `2^26` | about 1.479 GiB |
| `2^28` | about 5.915 GiB |

Later cycle and address layouts each lose another 8 bytes per live entry,
but they have already collapsed well below the first-bind support and do not
add to its peak.

Stacking this result on the prefix-offset metadata change updates the
representative first-bind construction estimate:

```text
parent:    142.258 + 40 + 8 + 48*(1.643807 + 1.313667)
         = 332.22 B/cycle = 83.05 GiB at T = 2^28

candidate: 142.258 + 40 + 8 + 40*(1.643807 + 1.313667)
         = 308.56 B/cycle = 77.14 GiB at T = 2^28
```

This is a source-capacity estimate. The repeated `T = 2^26` and exact
`T = 2^28` process maxima did not move in the predicted direction, which
means another phase or page-residency pattern set `ru_maxrss`. The allocation
reduction itself is fixed by `size_of` and the measured entry counts; no
process-RSS reduction is claimed.

## Integrated performance

### `T = 2^22`

| Metric | Parent mean (2 runs) | Candidate mean (2 runs) | Change |
|---|---:|---:|---:|
| Prover | 4.705384 s | 4.736719 s | +0.7% |
| Matrix construction | 0.018691 s | 0.020049 s | +7.3% |
| Cycle-major binds | 0.075628 s | 0.078595 s | +3.9% |
| Register compute | 0.032103 s | 0.030805 s | -4.0% |
| Register ingestion | 0.075576 s | 0.074792 s | -1.0% |
| Stage 4 | 0.156046 s | 0.155331 s | **-0.5%** |

The small screen showed a layout trade rather than a local primitive win.
The full Stage 4 span did not regress, so the candidate advanced to the
predeclared cache-crossover scale.

### `T = 2^26`

| Variant | Prover | Construction | Cycle binds | Stage 4 | Maximum RSS |
|---|---:|---:|---:|---:|---:|
| Parent 1 | 45.398628 s | 0.268441 s | 0.956095 s | 2.180810 s | 32.851547 GiB |
| Parent 2 | 46.437120 s | 0.306473 s | 0.988747 s | 2.298482 s | 32.709961 GiB |
| Candidate 1 | 45.196751 s | 0.230737 s | 0.892048 s | 2.073059 s | 32.866943 GiB |
| Candidate 2 | 46.457286 s | 0.233309 s | 0.898336 s | 2.091282 s | 32.868408 GiB |
| **Parent mean** | **45.917874 s** | **0.287457 s** | **0.972421 s** | **2.239646 s** | **32.780754 GiB** |
| **Candidate mean** | **45.827019 s** | **0.232023 s** | **0.895192 s** | **2.082171 s** | **32.867676 GiB** |

Construction improved by 19.3%, binds by 7.9%, and Stage 4 by 7.0%.
Complete proving was flat to 0.2% faster. The 0.087 GiB RSS movement is
opposite the known allocation reduction and below recent run variance. Both
proofs verified with zero swaps.

### `T = 2^28`

| Metric | Parent | Candidate | Change |
|---|---:|---:|---:|
| Prover | 154.288338 s | 153.551760 s | -0.5% |
| Commitment | 59.430681 s | 59.209915 s | -0.4% |
| Matrix construction | 0.984544 s | 0.897658 s | **-8.8%** |
| Cycle-major binds | 6.019282 s | 5.225115 s | **-13.2%** |
| Register compute | 2.231437 s | 2.126327 s | -4.7% |
| Register ingestion | 6.655820 s | 5.802660 s | -12.8% |
| Stage 4 | 11.589929 s | 10.383574 s | **-10.4%** |
| Maximum RSS | 79.412201 GiB | 79.930710 GiB | within prior range |

The proof verified and reported zero swaps. Unlike earlier candidates, the
unchanged commitment was nearly adjacent, so the 0.74-second whole-prover
improvement is directionally consistent with the 1.21-second Stage 4
reduction. The causal claim remains the changed spans.

## Validation

Passed:

- exact Akita layout-size regression test;
- exhaustive small-versus-wide lookup-index composition through saturation;
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
| `akita_22_reg_compact.json` | discovery candidate A | `a204e111a769d0461211573b2ee2ddae21df6af033da5df48188e7401f55040c` |
| `akita_22_reg_compact_repeat.json` | discovery candidate B | `6bdceb4242a0ca718a118bddba60531782705886b3d55949d97e5cc1492939da` |
| `akita_26_reg_compact.json` | crossover candidate A | `785d2aa4444eed8548b516f47345a346bbd55a88a2e4b033a834a66974d952db` |
| `akita_26_reg_compact_repeat.json` | crossover candidate B | `ac93ed2a99c3fc9d40ec21dbbdb909cfd5ec8ff08fa725425e98a29f1f5ae9c8` |
| `akita_28_reg_compact.json` | exact-target candidate | `703290e6c19ba3c8085600b869dbd20ad11a746ae9d0eba869d1101625316c25` |

## Next target

The first-bind old/new entries now cost about 118.30 B/cycle. A compact
initial-only entry could remove the redundant initial field value, while an
in-place or segmented bind could remove part of the output overlap. Either
requires a specialized first-round kernel and should begin with an
exact-shape throughput experiment; the straightforward width reductions are
now exhausted.
