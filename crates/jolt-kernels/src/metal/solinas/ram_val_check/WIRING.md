# RAM value-check Metal contract

This directory contains the low-level Metal implementation for `RamValCheck`.
The target is the production Fibonacci fixture with `log_T = 26` and
`log_K = 13` on the measured M4 Max. Source and pipeline registration, the
resident sequence API, an independent parity test, and Criterion microbenchmarks
are wired. The proof-stage `SumcheckKernel` adapter and direct upstream row
producer are not.

## Exact boundary

The relation over the cycle domain is

```text
s(j) = inc(j) * ra(j) * (LT(j, r_cycle) + gamma)
ra(j) = 0                                      when cycle j has no RAM access
      = eq(r_address, address[j])              otherwise.
```

It has 26 low-to-high rounds and degree three. Each message is sampled at
`t = 0, 2, 3`; the sumcheck engine recovers `s(1)` from the previous claim.
The Metal prefix returns exactly three fields for each active round. The host
constructs `UnivariatePoly::from_evals_and_hint`, combines this member with
`RegistersReadWriteChecking`, absorbs the batch polynomial, and draws the next
challenge. All Fiat-Shamir work remains on the host.

The device owns:

- the first 11 messages (rounds 0 through 10);
- the first 10 binds, fused with messages 1 through 10;
- resident ping-pong state for the bound `inc` and `ra` tables;
- reduction of three column-major block sums after every message.

The host owns:

- the stage-4 gamma draw and every round challenge;
- the split `LT + gamma` state and its small low-table binds;
- polynomial construction, batch combination, and clear or ZK proof recording;
- the CPU tail from table length `2^16` in the retained configuration;
- the final `LtCyclePlusGamma` scalar check and output-claim assembly.

The advice and program-image output cells are copied from this member's input
claims, as in the optimized CPU kernel. The CPU tail returns the fully bound
`ram_inc` and `ram_ra` values. The microbenchmark includes an exact dense
continuation for cutoff selection; the production adapter will use the existing
optimized tail. No transcript field, output claim, or protocol ordering changes.

At `T = 2^26`, the device starts with a 1-GiB native row allocation. Each
16-byte row contains a signed 64-bit magnitude and a remapped address:

```text
(inc_magnitude: u64, address: u32, flags: u32)
```

Bit 0 of `flags` says the increment is nonnegative. `u32::MAX` is the no-access
address. A nonzero increment must have an address, and every non-sentinel address
must be below `K`. This packs the 16-byte field increment and four-byte address
of a two-plane design into one 16-byte row without losing any valid RAM delta:
the difference of two `u64` values has magnitude at most `u64::MAX`.

The primary PIOP boundary assumes these transcript-independent rows and the
sequence arenas were produced before `RamValCheck::prepare`, following the
resident-witness convention used by the existing Metal backend. The member
still pays row attachment, split-table construction, command submission, every
wait and 48-byte message read, the 2-MiB CPU-tail handoff, the CPU tail, and
output claims. A second diagnostic must charge native-row production and arena
initialization. A temporary packed vector followed by a full-domain copy is not
an admissible primary path.

The frozen optimized-CPU median is 234.656875 ms. The hard 5x complete-member
cap is 46.931375 ms. The existing Metal arm still runs this CPU kernel and has a
237.366627-ms median; it is not a GPU baseline.

## Lower bound

For `log_T = 26`, split the cycle point into 13 high and 13 low variables, so
`H = L = 8192`. Write `N = 2^26`. The factorization

```text
LT(j, r_cycle) + gamma
  = lt_hi[j_hi] + eq_hi[j_hi] * lt_lo[j_lo]
```

lets each block retain three sums of `inc * ra` and three sums of
`inc * ra * lt_lo`. Its epilogue combines those six sums with `lt_hi` and
`eq_hi`. This avoids a dense `LT` table.

The measured M4 Max controls are:

| Control | Retained value |
| --- | ---: |
| Large streaming copy | 420.68 GiB/s |
| Best relevant full-field product rate | 32.33 Gproduct/s |
| Six-accumulator dense transition | 18.10 Gproduct/s |
| Conservative compute-dense control | 16.42 Gproduct/s |

The six-accumulator rate is the matched register-pressure control: this design
keeps three `a` and three `b` field sums live, just like the measured
registers-value transition. Metal's public pipeline limits do not expose
allocated registers, spills, or resident SIMD groups, so this is not an
occupancy claim. Promotion still requires an Instruments capture.

The compulsory traffic figures below count the native or dense state, not
repeated reads of the 128-KiB address-equality table or the 128-KiB LT-low
table. Those tables should remain cached, but their cache traffic is a named
risk rather than free work.

| Phase | Useful products | Compulsory bytes | Traffic floor | 32.33-G/s floor | 18.10-G/s register floor |
| --- | ---: | ---: | ---: | ---: | ---: |
| First message | `3N + 6H = 201,375,744` | 1.000 GiB | 2.377 ms | 6.229 ms | 11.126 ms |
| Native bind + message | `2.5N + 6H = 167,821,312` | 2.000 GiB | 4.754 ms | 5.191 ms | 9.272 ms |
| Nine dense transitions | `167,886,848` | 2.994 GiB | 7.117 ms | 5.193 ms | 9.276 ms |
| GPU prefix | `537,083,904` | 5.994 GiB | 14.249 ms | 16.613 ms | 29.673 ms |

The register-constrained projection binds for every phase. Using the slower
16.42-Gproduct/s control gives a 32.709-ms arithmetic projection for the GPU
prefix. One retained optimized trace spends about 5.5 ms in the rounds after
the `2^16` handoff. Allowing for command latency, the tail handoff, host
polynomial work, and row attachment gives a predicted complete-member range of
36 to 44 ms. This range clears 5x but leaves little room for a slow native
gather phase.

The arithmetic/traffic crossover for a six-accumulator phase is
`18.10e9 / (420.68 * 2^30) = 0.0401 product/byte`. The first message,
native transition, and dense ladder have compulsory intensities of 0.1875,
0.0781, and 0.0522 product/byte, respectively. All three are compute or
register-pressure bound in the cache-optimistic model. Charging every address
and LT-low lookup as DRAM traffic would invalidate that model and requires an
architecture review.

## Resident bytes and cutoff

The target allocation plan is:

| Allocation | Bytes at `T = 2^26`, `K = 2^13` |
| --- | ---: |
| Native rows | 1,073,741,824 |
| Dense arena A (`T/2` two-field rows) | 1,073,741,824 |
| Dense arena B (`T/4` two-field rows) | 536,870,912 |
| Address equality table | 131,072 |
| LT-low, LT-high, and EQ-high tables | 393,216 |
| Two three-column partial buffers | 786,432 |
| Total sequence resident | 2,685,665,280 bytes (2.5012 GiB) |

The native transition writes the `T/2` state. Nine dense transitions then use
source lengths `2^25` through `2^17` and leave `2^16` two-field rows. The host
reads that 2-MiB state, uses the challenge drawn from round 10, and performs the
remaining 15 messages plus the final bind. The cutoff matches the existing
registers-value control and must be retuned by paired complete-member results;
GPU-active time alone cannot move it.

## Adjustment candidates

1. The retained native row saves 20% of the compulsory input bytes compared
   with separate field-increment and `u32` address planes. Correctness follows
   from the `u64` bound on RAM-delta magnitude. Direct production is required
   so packing does not add a full-domain copy.
2. The retained split-LT factorization replaces a `T`-field table with three
   8192-field tables. The identity above is exact and gamma belongs in
   `lt_hi`; moving it elsewhere changes every message.
3. Bind and next-message work is fused so the new dense row is consumed before
   it is stored. Separating them adds one full read of every destination row.
4. A sparse active-block prefix could reduce work when RAM accesses are rare,
   but its gain depends on the workload's exact access topology and it needs a
   different merge representation. It is not the first candidate because the
   dense design already projects below the hard cap. Reconsider it only if
   native gathers miss their bar or direct native-row production is too costly.
5. Splitting samples across SIMD lanes reduces live accumulators but adds
   shuffles and duplicated table work. The related five-factor experiment lost
   throughput. Admit this variant only after an Instruments capture identifies
   register residency as the limiting resource and both added costs are priced.

No protocol change is proposed by any retained step.

## Falsification bars

The 80%-of-matched-roof active limits, registered before shader work, are:

| Phase | GPU-active limit |
| --- | ---: |
| First message | 13.907 ms |
| Native bind + message | 11.590 ms |
| Nine dense transitions | 11.594 ms |
| Sum of phase limits | 37.091 ms |

Each phase must pass exact parity first. Missing any active limit rejects the
current schedule unless a resource capture demonstrates that the matched
18.10-Gproduct/s control was inapplicable and the analysis is revised. Passing
all phase bars is necessary but not sufficient: the complete member, including
the CPU tail and required host work, must be at most 46.931375 ms. If the
measured roof supports substantially more than 5x, optimization continues.

## Log-26 observations

The retained schedule uses 32 threads for all three phase types, a balanced
13/13 LT split, ten device binds, and a `2^16` CPU handoff. One warm observation
after shader and allocation warmup measured:

| Component | Wall or active time |
| --- | ---: |
| Resident Metal prefix wall, including 11 waits and 2-MiB readback | 26.580 ms |
| Metal prefix active | 24.127 ms |
| Exact dense CPU continuation, including dense-LT construction | 4.643 ms |
| Hybrid wall without host Fiat-Shamir | 31.223 ms |

The no-Fiat-Shamir diagnostic is 7.52x against the frozen 234.656875-ms CPU
member. A focused Criterion run over 110 iterations measured the same hybrid at
`[30.890, 31.106, 31.232] ms`, or 7.54x at the median. This is not the promoted
member metric: the proof adapter must add host polynomial/transcript work and
show that the native rows came directly from the upstream producer.

The measured warm active phases were 7.653 ms for the first message, 7.431 ms
for the native transition, and 9.042 ms for the nine-transition dense ladder.
All pass their registered limits. The resident sequence allocates no buffers
between rounds and reuses the exact native-row allocation identity.

The retained choices came from these bounded ablations:

- Threadgroup widths of 64, 128, and 512 did not improve the large phases;
  32 threads gave the lowest complete-prefix active time.
- A 14-bit LT low half raised warm prefix active time from about 24.1 ms to
  25.2 ms and made the first two phases slower. The balanced split remains.
- One-shot no-Fiat-Shamir hybrid times for 9, 10, 11, and 12 GPU binds were
  32.425, 31.223, 30.427, and 31.528 ms. A repeated 11-bind run regressed to a
  31.901-ms median versus 31.106 ms for ten binds, so ten remains the default.

Uploading a separately built 1-GiB native row vector took 98--114 ms in these
runs and is intentionally outside the resident metric. That path cannot meet
5x and is inadmissible for promotion; the stage-4 producer must populate the
retained allocation directly. Sequence setup was 0.8--1.3 ms when measured
separately and must either be reused or charged by the production adapter.

## Pipeline and host sequence

Source order is `fp128.metal`, `simd_reduce.metal`, then
`ram_val_check/shader.metal`. Register four entry points:

| Entry point | Purpose |
| --- | --- |
| `solinas_ram_val_check_first_message` | First factorized message from native rows |
| `solinas_ram_val_check_native_transition` | First bind into dense state and second message |
| `solinas_ram_val_check_dense_transition` | Dense bind and next message |
| `solinas_ram_val_check_reduce3` | Recursive reduction of three columns |

For each message, dispatch one threadgroup for every high block. Threads per
threadgroup must be a nonzero multiple of 32. Dynamic threadgroup memory is
`6 * (threads / 32) * sizeof(Fp128)`. The two partial buffers are column-major.
Reduction uses `ceil(input_count / 32)` outputs until one value per column
remains.

The host sequence is:

1. Attach the native rows and preallocated arenas. Build `eq_address`,
   `lt_lo`, `lt_hi + gamma`, and `eq_hi`.
2. Run the first message, reduce three columns, and return 48 bytes.
3. Let the batch engine construct and absorb the round polynomial, then draw
   the challenge. Bind the host LT-low table.
4. Run the native transition with that challenge and the bound LT-low table.
5. Repeat the host polynomial, transcript, LT bind, dense transition sequence
   for nine dense transitions.
6. Expose the `2^16` dense state to the CPU tail. Continue the same split-LT
   object, message formula, transcript, and final validation on the host.

## Promotion work outside this slice

- direct resident-row production by the shared RAM witness producer;
- the `SumcheckKernel` adapter and production optimized CPU-tail handoff;
- clear and ZK parity tests at the stage-4 batch boundary;
- a paired complete-member measurement including host Fiat-Shamir;
- Instruments register, spill, and occupancy capture.

The shader compiles on the measured M4 Max. The low-level test checks the first
message and every native/dense transition against the independent Rust oracle,
including signed increments, modulus-boundary values, no-access rows, and
zero/one/minus-one/random challenges. All 244 `jolt-kernels` Metal tests passed
after the target-scale run.
