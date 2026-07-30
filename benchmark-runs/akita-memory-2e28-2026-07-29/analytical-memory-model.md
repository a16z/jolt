# Akita K256 prover memory model

Date: 2026-07-30 EDT

## Decision

The current Akita K256 prover does not have a safe `2^28` capacity margin on a
95 GiB machine, even though its source-level allocations narrowly fit. The
current analytical peaks are:

| Window | Current `2^28` live/transition peak |
|---|---:|
| Commit | **90.52 GiB** |
| Stage 6b | **87.18 GiB** |
| Root evaluation proof | **79.25 GiB** |
| Stage 5 conservative transition | **81.15 GiB** |

The commit peak leaves only 4.48 GiB for allocator metadata, thread stacks,
unmodelled fixed state, and resident pages from earlier allocations. It also
exceeds the 90 GiB working target before any such overhead. Running `2^28`
now would therefore be a swap-risk experiment, not a capacity-safe run.

The model identifies three changes that attack different maxima and stack:

1. Build negacyclic-only NTT slots for `digit_rows`. This removes exactly
   23.515625 GiB from the `2^28` commit and 5 GiB from root `compute_v`.
2. Chunk the streamed `t_hat` arm of the root ring quotient. This prevents its
   current capacity fallback and removes a 47.03125 GiB full-matrix NTT slot
   from the evaluation proof.
3. Release the 64-byte trace row allocation after the RAM Hamming prover has
   read it at the start of Stage 6b. This removes exactly 16 GiB from the
   Stage-6b materialization window.

After those changes, the modeled `2^28` structural maximum is the conservative
Stage-5 transition at no more than 81.15 GiB for the current proof shape. That
leaves 13.85 GiB below the hard limit and 8.85 GiB below the 90 GiB working
target. A drop-completion barrier and macOS allocator pressure relief must
then keep logically dead pages inside that 8.85 GiB reserve.

This is a capacity model, not an RSS fit. Measurements are used only to check
that the predicted ownership transitions occur; they do not determine which
data structure is targeted.

## Scope and units

This model covers the transparent Akita packed prover with:

- trace length `T`, a power of two;
- one-hot chunk size `K = 256`;
- selector capacity `C = 32`;
- 29 active semantic packed columns;
- Akita ring dimension `D = 64`;
- `Fp128` coefficients of `f = 16` bytes;
- the Q128 CRT profile with `q = 5` 32-bit primes.

All GiB values use `2^30` bytes. At `T = 2^28`:

```text
1 byte/cycle = 0.25 GiB
95 GiB        = 380 bytes/cycle
90 GiB        = 360 bytes/cycle
```

The hard capacity requirement is:

```text
max_phase(logically live + construction overlap + unavoidable caches)
  + allocator/unmodelled reserve
  <= 95 GiB
```

The preferred working requirement substitutes 90 GiB, retaining 5 GiB of
additional system headroom.

## Accounting rules

Four quantities must not be conflated:

1. **Owned live bytes** are reachable from the prover or a phase-local object.
2. **Construction overlap** is old and new storage that must coexist while a
   representation is built.
3. **Logically dead bytes** have been handed to a background drop or freed by
   Rust, but their destructor may not yet have run.
4. **Resident dead pages** have been returned to the allocator but not to
   macOS, so they still contribute to RSS.

The formulas below include the first two. The final reserve must cover the
last two, thread stacks, vector headers, small equality tables, proof objects,
and preprocessing state. Fixed and `O(log T)`, `O(K)`, or program-sized terms
are called out but omitted from byte-per-cycle totals when they are far below
one MiB.

Phase-local maxima are not added across phases. For example, Stage 4's sparse
register matrix and Stage 6b's materialized RA field tables never need to be
live at the same time.

## Schedule variables

For a schedule level, define:

| Symbol | Meaning |
|---|---|
| `P` | positions per block |
| `B` | live blocks |
| `n_a` | inner/A commitment rows |
| `n_b` | outer/B commitment rows |
| `n_d` | opening/D commitment rows |
| `d_o` | outer decomposition digits |
| `d_e` | opening decomposition digits |

The one physical root polynomial has

```text
N = T * K * C = 8192T logical coefficients
B = N / (D * P)
```

The planner-selected root parameters are:

| `T` | Vars | `P` | `B` | `n_a` | `n_b` | `n_d` | `d_o = d_e` |
|---:|---:|---:|---:|---:|---:|---:|---:|
| `2^26` | 39 | `2^21` | 4,096 | 6 | 2 | 1 | 43 |
| `2^28` | 41 | `2^20` | 32,768 | 7 | 2 | 2 | 43 |

The setup envelope in ring elements is the largest matrix extent required by
the schedule. At the root, the relevant extents include

```text
A inner: n_a * P
B outer: n_b * B * n_a * d_o
D open:  n_d * B * d_e
```

and the planner envelope is:

| `T` | Setup rings | Dominating extent | Field-form setup |
|---:|---:|---|---:|
| `2^26` | 12,582,912 | `n_a * P` | 12.0000 GiB |
| `2^28` | 19,726,336 | `n_b * B * n_a * d_o` | 18.8125 GiB |

The `2^28` setup is therefore 75.25 bytes/cycle by itself. The remaining
budgets for state that scales with `T` are:

| Limit | Gross | Minus setup |
|---|---:|---:|
| 95 GiB hard cap | 380.00 B/cycle | 304.75 B/cycle |
| 90 GiB working cap | 360.00 B/cycle | 284.75 B/cycle |

## Long-lived prover state

### Trace and packed witness sources

| State | Formula | Lifetime |
|---|---:|---|
| Compact proof trace | `64T` | trace construction through Stage 7 currently |
| Packed semantic lane rows | `29T` | commit through accepted root preparation |
| Deferred RAM-valid byte | `T` | commit through RA materialization after Stage 5 |
| Materialized `RaIndices` | `54T` | after Stage 5 through Stage 7 |
| Fused-inc dense lanes | `9T` | Stage 6 through Stage 7 |
| Packed fused deltas | `8T + ceil(T/64)*8`, approximately `8.125T` | Stage 6a through its Stage-6b bind |

The packed physical polynomial of length `8192T` is never materialized. Its
root commitment and opening kernels read the 29 byte lanes and virtualize the
zero lane and selector prefix.

### Commitment hint

The retained hint is a flat signed-byte digit stream:

```text
H = B * n_a * d_o * D bytes
```

| `T` | Hint bytes | GiB | B/cycle |
|---:|---:|---:|---:|
| `2^26` | 67,633,152 | 0.062988 | 1.007813 |
| `2^28` | 631,242,752 | 0.587891 | 2.351563 |

This is schedule-dependent rather than a constant byte/cycle term: the root
planner changes `P` and `n_a` between `2^26` and `2^28`.

## Commitment

### Inner sweep and decomposition

The lazy packed sweep does not materialize the physical one-hot coefficients.
Its main outputs are:

```text
partial rows = (T*K/(D*P)) * 29 * n_a * D * f
inner rows   = B * n_a * D * f
hint digits  = H
```

| Allocation | `2^26` | `2^28` |
|---|---:|---:|
| Active-column partials | 0.021240 GiB | 0.198242 GiB |
| Recomposed inner rows | 0.023438 GiB | 0.218750 GiB |
| One hint-sized digit stream | 0.062988 GiB | 0.587891 GiB |

The implementation temporarily has three hint-sized digit carriers during
the inner-to-outer seam: the digits in `CommitInnerWitness`, `b_input_flat`,
and the cloned `DigitBlocks` retained in the hint. This construction peak is
small compared with the NTT window. During the outer product, the original
inner witness is gone but `b_input_flat` and the retained hint coexist, hence
the commit peak contains `2H`.

### Current outer NTT cache

The outer commitment requests:

```text
E_commit = n_b * B * n_a * d_o setup rings
slot(E)  = min(setup_envelope, next_power_of_two(E))
```

The current cache always stores both negacyclic and cyclic transforms:

```text
NTT bytes/ring = 2 transforms * D * q * sizeof(i32)
               = 2 * 64 * 5 * 4
               = 2,560 bytes/ring
```

| `T` | Requested rings | Slot rings | NTT cache |
|---:|---:|---:|---:|
| `2^26` | 2,113,536 | 4,194,304 | 10.0000 GiB |
| `2^28` | 19,726,336 | 19,726,336 | 47.03125 GiB |

`digit_rows` reads only the negacyclic member of the cache. The cyclic half is
not used by this operation.

### Commit peak

At the outer product, the exact scaling terms are:

```text
setup + current NTT + trace + packed rows/RAM-valid + 2H
```

| `T` | Analytical peak | Gross B/cycle |
|---:|---:|---:|
| `2^26` | 28.00098 GiB | 448.0156 |
| `2^28` | **90.51953 GiB** | **362.0781** |

Small commitment rows and proof metadata do not materially change the table.
At `2^28`, only 4.48047 GiB remains below the hard limit. A negacyclic-only
cache is 1,280 bytes/ring and lowers this peak by exactly 23.515625 GiB, to
67.00391 GiB, without changing the matrix product or proof.

## PIOP stages

The common standing state during Stages 1–5 is:

```text
setup + trace + packed rows/RAM-valid + hint
```

At `2^28` this is:

```text
18.8125 + 16 + 7.5 + 0.587891 = 42.900391 GiB
```

Some stage sizes depend on the executed workload. Define:

| Symbol | Meaning | Range |
|---|---|---:|
| `mu = M/T` | RAM load/store density | `[0, 1]` |
| `rho = N_reg/T` | distinct register sparse entries per cycle | `[0, 3]` |
| `lambda = L/T` | cycles retained in grouped instruction lookup indices | `[0, 1]` |
| `R` | words in the initial RAM state | program/input dependent |

The representative SHA-2 trace has approximately `mu = 0.027` and
`rho = 1.23`. These densities are used only in the representative column;
the formulas remain the capacity contract.

### Stage 1: Spartan outer

The old `R1CSCycleInputs` cache was `208T` and has been removed
unconditionally. The remaining linear stage materializes `Az` and `Bz`, each
an `Fp128` polynomial of length `T`:

```text
Stage1 local = 2 * f * T = 32T
```

The construction windows use bounded grids and accumulators, not another
trace-sized R1CS row cache.

### Stage 2: product remainder and RAM

The product virtual remainder retains two field polynomials:

```text
2 * f * T = 32T
```

RAM read/write checking adds an `i128` increment polynomial, one 64-byte
sparse entry per memory operation, and field-form initial RAM state:

```text
Stage2 local = 48T + 64M + 16R
             = (48 + 64mu)T + 16R
```

The other claim reductions and equality tables are sublinear in `T`.

### Stage 3: instruction inputs

Before binding, the delayed small-value representation is:

```text
4 bool columns + 3 u64 columns + 1 i128 column = 44T
```

After three binds it becomes eight `Fp128` polynomials of length `T/8`:

```text
8 * f * T/8 = 16T
```

Sequential materialization creates a conservative transition ceiling near
`49T`; it does not recreate the former eight full field columns.

### Stage 4: register and RAM value checking

The fixed-width terms are:

```text
RdInc i128                 16T
RamInc i128                16T
RAM write address Option   16T
```

Register read/write checking stores one 64-byte sparse entry per distinct
register touched in a cycle:

```text
Stage4 local = 48T + 64N_reg
             = (48 + 64rho)T
```

The structural worst case is `rho = 3`, or `240T`. The representative SHA-2
trace is about `rho = 1.23`, or approximately `126.7T`.

At `2^28`, the current hard-cap condition for Stage 4 is:

```text
219.6016 + 64rho <= 380
rho <= 2.506
```

The 90 GiB working condition is `rho <= 2.194`. Therefore the current Stage-4
representation can fit the SHA-2 target but cannot guarantee `2^28` for every
valid trace. A universal capacity guarantee eventually requires streaming or
compacting its sparse register entries.

### Stage 5: lookup read-RAF and reductions

Initial storage consists of:

- 17 instruction lookup bit columns;
- one interleaved byte/bool stream;
- one field `u` polynomial;
- RAM `Option<usize>` indices;
- register `i128` increments;
- grouped `usize` lookup indices, totaling `8L` bytes.

At the handoff to the last `log T` rounds, read-RAF materializes four RA field
polynomials plus one combined value polynomial:

```text
5 * f * T = 80T
```

The new steady state, while the other two instances remain live, is
approximately:

```text
(112 + 8lambda)T
```

Because the old compact lookup streams can coexist with the new field vectors
while background drops run, the conservative transition bound is:

```text
(145 + 8lambda)T <= 153T
```

This is 38.25 GiB of stage-local state at `2^28`, or a total phase ceiling of
81.15 GiB including the setup and common standing state.

### Stage 6a: address phases

After Stage 5, the RAM-valid byte is consumed and `RaIndices` is materialized.
The standing trace-scaled state is:

```text
trace          64T
packed lanes   29T
RaIndices      54T
fused lanes     9T
fused deltas  8.125T
hint             H
-------------------
              164.125T + H
```

The address provers share these rows; their equality tables are `O(K)` or
sublinear rather than additional `T`-length field polynomials.

### Stage 6b: cycle phases

After three cycle binds, the following field vectors have length `T/8`:

| Family | Count | B/cycle |
|---|---:|---:|
| Base booleanity RA columns | 20 | 40 |
| Fused-inc booleanity columns | 9 | 18 |
| Instruction RA virtual | 16 | 32 |
| RAM RA virtual | 2 | 4 |
| Bytecode read-RAF | 2 | 4 |
| RAM Hamming booleanity | 1 | 2 |
| Fused delta | 1 | 2 |
| **Total** | **51** | **102** |

The steady source-level peak is:

```text
setup + trace64 + packed29 + Ra54 + fused lanes9 + field102 + hint
```

At `2^28`, this is 83.90039 GiB, or 335.6016 B/cycle.

The bytecode optional-byte columns, RAM Hamming bool source, and packed fused
deltas can overlap their field outputs at the transition. A conservative
extra `13.125T` gives:

```text
Stage6b transition = 87.18164 GiB = 348.7266 B/cycle
```

The trace is not read after
`HammingBooleanitySumcheckProver::initialize` has built its one-byte Boolean
column at the start of Stage 6b. Releasing the final trace owner at that point
subtracts exactly `64T`, or 16 GiB at `2^28`, from both values. The transition
then becomes 71.18164 GiB.

### Stage 7 and reconstruction

Stage 7 retains the trace, packed lanes, RA rows, fused lanes, and hint:

```text
setup + (64 + 29 + 54 + 9)T + H
```

This is 58.40039 GiB at `2^28`. The Hamming-weight reduction tables are
`O(K * columns)`.

After Stage 7, the RA rows and trace are dropped. Reconstruction retains:

```text
setup + 29T + H
```

or 26.65039 GiB at `2^28`, plus advice/program-sized objects that do not scale
with `T` for the SHA-2 target.

### Stage summary

| Phase | Formula beyond setup | `2^26` | Representative `2^28` |
|---|---|---:|---:|
| Stage 1 | common + `32T` | 19.94 GiB | 50.90 GiB |
| Stage 2 | common + `(48+64mu)T+16R` | 21.05 GiB | 55.33 GiB |
| Stage 3 transition | common + `49T` | 21.00 GiB | 55.15 GiB |
| Stage 4 | common + `(48+64rho)T` | 25.86 GiB | 74.58 GiB |
| Stage 5 transition | common + at most `153T` | 27.50 GiB | 81.15 GiB |
| Stage 6a | `(164.125T+H)` | 22.32 GiB | 60.43 GiB |
| Stage 6b steady | `(258T+H)` | 28.19 GiB | 83.90 GiB |
| Stage 6b transition | `(271.125T+H)` | 29.01 GiB | 87.18 GiB |
| Stage 7 | `(158T+H)` | 21.94 GiB | 58.40 GiB |
| Reconstruction | `(29T+H)` | 13.88 GiB | 26.65 GiB |

The Stage-2 and Stage-4 representative values use `mu = 0.027` and
`rho = 1.23`; `R/T` is negligible for this target. The Stage-5 row is a
conservative transition bound, not a claim that all 153 bytes remain live for
the entire stage.

## Akita evaluation proof

### Recursive witness sizes

The root ring switch produces one compact signed-byte witness. Subsequent
folds shrink it rapidly:

| Witness | `2^26` bytes | `2^28` bytes |
|---|---:|---:|
| Root input, logical only | 549,755,813,888 | 2,199,023,255,552 |
| `W0` root output | 615,803,776 | 1,056,997,632 |
| `W1` | 13,265,152 | 16,685,056 |
| `W2` | 1,429,504 | 1,589,248 |
| `W3` | 483,328 | 520,192 |
| `W4` | 261,376 | 209,920 |
| `W5` | 144,384 | 135,936 |
| `W6` | 105,984 | 105,984 |
| `W7` / terminal input | 91,904 | 91,904 |

Only `W0` is material at the GiB scale. The logical root input is never
allocated; the packed source virtualizes it from 29 lane bytes/cycle.

### Root evaluation and decomposition

Root `evaluate_and_fold` emits `B` field rings:

```text
B * D * f
```

This is 4 MiB at `2^26` and 32 MiB at `2^28`.

Root decompose-fold retains `P` field rings and two centered `i32` copies: the
global centered witness and the single-chunk acceptance witness:

```text
P * D * (f + 4 + 4)
```

| `T` | Root decompose/acceptance state |
|---:|---:|
| `2^26` | 3.0000 GiB |
| `2^28` | 1.5000 GiB |

The decrease is real: the `2^28` root schedule halves `P`.

Additional root terms are:

```text
e_hat            = B * d_e * D bytes
e_folded          = B * D * f bytes
recomposed t rows = B * n_a * D * f bytes
t_hat             = H bytes
W0                = scheduled output bytes
```

### Root NTT slots

`compute_v` requests:

```text
E_v = n_d * B * d_e
```

The current both-transform slots are:

| `T` | `E_v` | Rounded slot | Cache |
|---:|---:|---:|---:|
| `2^26` | 176,128 | 262,144 | 0.625 GiB |
| `2^28` | 2,818,048 | 4,194,304 | 10.000 GiB |

`compute_v` calls `digit_rows` and reads only negacyclic transforms, so its
`2^28` slot can be 5 GiB.

The root B quotient then processes:

```text
t_len       = B * n_a * d_o
E_t         = n_b * t_len
```

Its streamed path currently requires all `t_len` terms to fit one CRT
accumulator. They do not. The fallback builds a both-transform cached slot:

| `T` | `t_len` | `E_t` / slot | Cache |
|---:|---:|---:|---:|
| `2^26` | 1,056,768 | 2,113,536 -> 4,194,304 | 10.000 GiB |
| `2^28` | 9,863,168 | 19,726,336 | 47.03125 GiB |

This branch is derivable from the capacity check. The retained `2^26` trace
confirms it reports `t_safe=false`; the trace is a check on the model, not the
source of the estimate.

The A quotient already chunks and streams its centered `z` rows. Giving the
`t_hat` arm the same chunked streamed accumulation removes this fallback and
the entire 47.03125 GiB root B slot at `2^28`.

### Range proof over `W0`

For basis 4 or 8, the digit range prover keeps `W0` as compact `i8` data
through three rounds, then materializes `Fp128` values at `W0/8`. During that
transition:

```text
compact W0 + materialized 16*(W0/8) = 3W0 bytes
```

| `T` | `W0` | Range transition |
|---:|---:|---:|
| `2^26` | 0.573512 GiB | 1.720536 GiB |
| `2^28` | 0.984406 GiB | 2.953217 GiB |

Later folds are smaller. The root-level NTT slots remain cached until the
root proof window ends, so the range transition must be compared with, rather
than added after, the ring-switch construction peak.

### Current evaluation-proof ceiling

A conservative ring-switch construction peak includes:

```text
setup
+ compute_v NTT slot
+ root-B fallback NTT slot
+ t_hat
+ root decompose/acceptance state
+ recomposed t rows
+ W0
+ e_hat
+ e_folded
```

The packed lane rows are not included: the accepted root preparation releases
them before `ring_switch_build_w` creates the large B slot.

| `T` | Analytical construction peak | Gross B/cycle |
|---:|---:|---:|
| `2^26` | 26.29934 GiB | 420.7895 |
| `2^28` | **79.25003 GiB** | **317.0001** |

The adjacent `2^28` range transition is 78.79697 GiB. With a negacyclic-only
`compute_v` slot and chunked streamed `t_hat`, the large construction term
falls to about 27.22 GiB. The earlier root-preparation window still holds the
7.25 GiB packed rows, making the post-change evaluation-proof ceiling about
33.3 GiB.

The recursive tail uses at most a 262,144-ring NTT slot (0.625 GiB with both
transforms) at the first successor and becomes negligible thereafter.

## Capacity ceiling and required cuts

### Current `2^28`

| Window | B/cycle | GiB | Hard-cap reserve |
|---|---:|---:|---:|
| Commit | 362.08 | 90.52 | 4.48 GiB |
| Stage 5 transition | 324.60 | 81.15 | 13.85 GiB |
| Stage 6b transition | 348.73 | 87.18 | 7.82 GiB |
| Evaluation proof | 317.00 | 79.25 | 15.75 GiB |

The commit fails the 90 GiB working requirement and leaves too little hard-cap
reserve. Stage 6b also leaves too little reserve if allocator-retained bytes
scale with `T`.

### After the three structural cuts

| Window | Change | Projected peak |
|---|---|---:|
| Commit | negacyclic-only `digit_rows` | 67.00 GiB |
| Stage 6b transition | release trace after final reader | 71.18 GiB |
| Evaluation proof | negacyclic `compute_v` + streamed chunked `t_hat` | about 33.3 GiB |
| Stage 5 transition | unchanged conservative bound | at most 81.15 GiB |
| Stage 4, SHA-2 density | unchanged | about 74.58 GiB |

The resulting structural ceiling is at most 81.15 GiB for the current target,
leaving 13.85 GiB under 95 GiB and 8.85 GiB under the 90 GiB working target.

This does not prove a universal trace guarantee. Stage 4 exceeds 95 GiB when
`rho > 2.506`, so high-register-density workloads need a separate streamed or
more compact sparse-matrix representation.

## Why RSS remains above logical ownership

Stages 2–5 end with:

```rust
drop_in_background_thread(instances)
```

which uses fire-and-forget `rayon::spawn`. There is no handle or barrier
before the next phase allocates. This creates two distinct effects:

1. old instance buffers can remain genuinely owned while the next stage
   initializes;
2. after the destructor runs, the allocator can retain the dirty pages.

Allocative ownership snapshots show the large Stage-2/3/4 objects reach zero
logical ownership by their endpoints, so there is no evidence that the
prover struct accidentally retains them across all later stages. The missing
piece is timely destruction and page reclamation, not another protocol
polynomial.

The current `2^26` Stage-6 window is about 5 GiB above the conservative
source-level transition model. That observation is not extrapolated as a
capacity formula, but it establishes that the allocator reserve is already
material. After the structural cuts, the acceptance target is:

```text
background-owned + allocator-resident + unmodelled fixed state <= 8 GiB
```

at the Stage-5/6 boundary. A completion barrier followed by
`malloc_zone_pressure_relief` on macOS is the direct experiment. It is
accepted only if prover time does not regress.

## Ordered implementation plan

### A. Negacyclic-only `digit_rows` cache

Add a cache path whose key distinguishes a negacyclic-only slot from a
both-transform slot. Route `digit_rows` through it; retain both transforms for
cyclic and quotient consumers.

Expected exact `2^28` changes:

- commit NTT: 47.03125 -> 23.515625 GiB;
- root `compute_v`: 10 -> 5 GiB;
- no proof, transcript, or protocol change;
- less transform work and half the cache write/read traffic, so a runtime
  regression is not expected.

This is the first implementation target.

### B. Stream and chunk the root `t_hat` quotient

Extend the existing streamed quotient kernel so `t_hat` uses capacity-safe
chunks instead of requiring `t_safe == true` for the entire vector. This is
the direct analogue of the already chunked streamed `z` arm.

Expected exact `2^28` change:

- remove the 47.03125 GiB root B cache;
- avoid writing and rereading that transformed matrix;
- retain only bounded CRT accumulators and field-form setup reads.

This is likely a performance win at large `T`, but it needs a focused kernel
benchmark and a full `2^26` proof before promotion.

### C. Release trace at its actual final reader

Capture `trace_len`, initialize the RAM Hamming Booleanity source, then replace
and drop `self.trace` before Stage 6b allocates its large field tables. Stage 7
and reconstruction do not read the trace.

Expected exact `2^28` change:

- remove 16 GiB from Stage 6b and Stage 7;
- add no scan, conversion, or protocol work.

The implementation must account for any outstanding `Arc` held by background
drops. A debug ownership check or explicit drop barrier should make the
lifetime claim testable.

### D. Make phase drops observable and reclaimable

Replace fire-and-forget destruction at the large phase boundaries with a
completion mechanism. At the Stage-5/6 boundary, request allocator pressure
relief on macOS after all prior phase owners are gone.

Acceptance:

- at most 8 GiB non-owned reserve at the modeled `2^28` ceiling;
- no prover-time regression outside the established noise band;
- no process swaps and no system swapouts.

### E. Stage-4 universal-trace follow-up

If `2^28` must be guaranteed for arbitrary valid traces rather than the SHA-2
target, Stage 4 is next. Its `64N_reg` sparse entries should be streamed in
cycle windows or compacted while preserving the current locality. The hard
analytical target is `rho <= 2.506`; any representation intended to remove
that workload restriction must make the worst-case local term no larger than
`160.4T` under the 95 GiB cap, and preferably `140.4T` under the 90 GiB
working cap.

## Validation contract

Each optimization lands in its own commit.

1. Unit or small synthetic tests establish byte formulas and output parity.
2. `cargo nextest`, never `cargo test`.
3. Run the host Akita end-to-end tests and the required host/host+zk muldiv
   gates where the changed crate can affect them.
4. Screen at `2^22`; validate at `2^26` with forced K256.
5. Retain short, notable Perfetto traces and RSS logs in this benchmark run.
6. Reject any capacity change that causes a reproducible prover regression.
7. Attempt `2^28` only after the analytical structural ceiling plus measured
   non-owned reserve is below 90 GiB.

## Source map

The formulas above are derived from:

- packed pipeline lifetimes:
  `crates/jolt-prover-legacy/src/zkvm/packed.rs`;
- background destruction:
  `crates/jolt-prover-legacy/src/utils/thread.rs`;
- trace layout:
  `crates/jolt-riscv/src/trace_row.rs`;
- RA row layout:
  `crates/jolt-prover-legacy/src/poly/shared_ra_polys.rs`;
- Stage-4 sparse entries:
  `subprotocols/read_write_matrix/{registers,ram}.rs`;
- root packed commitment/opening:
  `crates/jolt-akita/src/trace_onehot.rs`;
- Akita CPU NTT cache sizing:
  `akita-prover/src/compute/cpu.rs` and
  `akita-types/src/ntt_cache.rs`;
- root ring quotient and fallback:
  `akita-prover/src/protocol/ring_relation/relation_quotient.rs` and
  `akita-prover/src/kernels/linear/fused_quotients.rs`;
- recursive witness schedule:
  `JoltD64OneHotK256::runtime_schedule`.

## Remaining uncertainty

The model intentionally leaves the following as explicit variables or
reserves:

- `mu`, `rho`, and `lambda` vary by guest trace;
- advice and committed-program objects are program/input sized;
- vector capacity rounding and Rayon stacks are not charged per object;
- macOS allocator residency is not derivable from Rust ownership;
- the Stage-5 transition bound is conservative because background destructor
  timing is not deterministic.

None of these uncertainties changes the ordering of the first three targets:
the current commit's unused cyclic NTT half, the root `t_hat` fallback, and
the trace's late owner are respectively 23.5 GiB, 47.0 GiB, and 16 GiB at
`2^28`.
