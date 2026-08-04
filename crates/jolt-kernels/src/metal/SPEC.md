# Solinas Fp128 Metal kernels

| Field | Value |
|---|---|
| Created | 2026-08-03 |
| Status | Akita arithmetic, resident five-factor sequence, and first hybrid sumcheck slot implemented; occupancy capture pending |
| Scope | `jolt-kernels::metal` and its Criterion benchmarks |

This backend establishes the arithmetic and hardware limits of canonical 128-bit
Solinas-prime kernels on Apple GPUs, then applies those results to Jolt one sumcheck
slot at a time. It contains reusable field operations, a Rust dispatch layer,
correctness tests, controlled arithmetic probes, a resident five-factor round
sequence, and a hybrid `InstructionReadRaf` implementation. The stage driver and
Fiat-Shamir transcript remain on the host.

## Scope

The implemented field is `jolt_field::AkitaField`,
`p = 2^128 - 0xffffa7f7`. The development machine is an Apple M4 Max with 40 GPU
cores and unified memory. Device and compiled-pipeline limits are queried at runtime
rather than inferred from the model name.

Requirements:

- Values are canonical integers in `[0, p)`, encoded as four little-endian `u32`
  limbs. Host and device elements have a 16-byte stride and alignment.
- Arithmetic is exact for every canonical input. Entry kernels bounds-check their
  grid index.
- The offset `C` in `2^128 - C` is a nonzero compile-time shader constant smaller
  than `2^32`. Specializing it does not change the buffer ABI.
- Measurements keep command latency, dependency-chain latency, saturated arithmetic
  throughput, and streaming bandwidth separate.
- A dispatch rejects any field buffer larger than the device's runtime
  `maxBufferLength` before asking Metal to allocate it.
- Every retained result records the device, OS, probe, element and iteration counts,
  threadgroup width, and compiled-pipeline limits.

Non-goals:

- No device transcript, PCS operations, commitment kernels, or opening kernels. The
  first prover slot covers only the dense cycle tail of `InstructionReadRaf`.
- No attempt to preserve the structure of the previous experimental Metal branch.
- No claim that high occupancy alone means high field throughput.
- No stable public prover API. The current Rust surface exists to test and measure the
  kernels.

## Representation and host boundary

`metal::solinas::Fp128` is a buffer ABI, not a second host field implementation. It is
`#[repr(C, align(16))]` over `[u32; 4]` and provides only limb conversion and
canonicality checks. Runtime compilation specializes the MSL for `C`; dispatch checks
all inputs against `2^128 - C` before copying them into shared Metal buffers. Output
is checked for canonicality before it reaches Rust.

The Rust boundary converts between `AkitaField` and `Fp128` explicitly. Directly
casting an `AkitaField` slice remains disallowed until size, alignment, canonical
form, limb order, and aliasing are public invariants of the field type.

## Arithmetic

`SolinasFp128<C>` represents an integer in `[0, 2^128 - C)`. Addition and subtraction
fold a carry or borrow using `2^128 = C (mod p)`. Multiplication computes an
eight-limb schoolbook product, folds the upper four limbs by `C`, folds the remaining
high word a second time, and performs a branchless canonical correction.

The retained product schedule expresses each `u32 * u32` product as a widening
`ulong` operation. An alternative that explicitly requested low and high halves with
`*` and `mulhi` was removed after it was slower in every streaming and compute-dense
reconnaissance case. `fp128.metal` contains field functions only;
`probes.metal` contains entry points.

Any 128-by-128-bit schoolbook product needs 16 independent 32-by-32-bit limb
products. The retained schedule meets that lower bound. Reduction adds four
multiplications by `C`, one multiplication of the first-fold carry by `C`, carry
propagation, and a canonical correction. The compiler may strength-reduce constant
multiplication, so these are source operations rather than a claimed instruction
count.

### Reduction bound

Write a product `x < 2^256` as `x0 + 2^128*x1`. The first fold is
`x0 + C*x1 < (C + 1)2^128`, so its high word is at most `C`. The second fold adds at
most `C^2`. If it overflows 128 bits, adding one more `C` folds that carry. Otherwise
a trial addition of `C` detects values at least `p`. The condition `C(C + 1) < p`
makes one final correction sufficient; it holds for `C = 0xffffa7f7`. The reusable
implementation accepts any nonzero 32-bit offset satisfying that bound.

## Limit model

Three ceilings matter:

- Resident occupancy: active SIMD groups relative to the hardware's resident
  SIMD-group capacity.
- Arithmetic utilization: observed primitive integer rate relative to a same-binary
  raw-integer probe with a comparable dependency and ILP shape.
- Bandwidth utilization: logical bytes transferred relative to a same-run copy roof.

For threadgroup width `T`, execution width `W`, static threadgroup memory `S_tg`, and
per-thread register allocation `R_thread`, the usual residency bound is

```text
G_tg       = ceil(T / W)
N_resident = min(N_hw,
                 floor(S_core / S_tg),
                 floor(R_core / (T * R_thread)),
                 floor(G_core / G_tg))
occupancy  = N_resident * G_tg / G_core.
```

Terms for resources with zero allocation are omitted from the minimum.

Metal exposes `W`, the legal maximum `T`, and `S_tg` on the compiled pipeline. Public
runtime properties do not expose `R_thread`, `R_core`, `G_core`, or every hardware
residency cap. Those runtime values constrain occupancy but cannot produce an honest
percentage on their own. Xcode's GPU timeline supplies the missing compiled-resource
analysis and is the source of record for theoretical and measured occupancy.

For a kernel with `M` compiled arithmetic operations and `B` bytes of device-memory
traffic per field result, the roofline is

```text
R_field <= min(R_issue / M, bandwidth / B).
```

A streaming field operation reads two 16-byte operands and writes one result, so
`B = 48`. Copy moves 32 logical bytes per element. A chain of `k` multiplications
amortizes 48 bytes over `k` operations and has arithmetic intensity `k / 48` field
operations per byte. Logical byte counts exclude cache-line overfetch and command
traffic.

Command-buffer wall time is

```text
t_wall = t_encode + t_submit + t_queue + t_gpu + t_wait.
```

The no-op probe measures the fixed wall-time floor. `Invocation::execute_timed` also
reports `GPUEndTime - GPUStartTime` after completion, which isolates command-buffer
GPU execution from host encoding, queueing, and waiting. Every timed command buffer
for a primitive probe contains one compute dispatch; a product5 command contains its
main dispatch and recursive reductions. A one-SIMD-group, dependent-chain sweep over `k`
estimates dependency latency from the large-`k` slope; its intercept and timestamp
resolution still prevent calling that slope raw instruction latency without an
Instruments capture. Large-grid chains measure saturated dependency-bound throughput
instead. With `I` independent chains per thread, useful rate is bounded by

```text
R_chain(I) <= min(R_issue / M, W_active * I / L_dependency).
```

ILP values 1, 2, 4, and 8 show where latency becomes hidden and where added live state
reduces `W_active` through register pressure.

Apple does not publish enough execution-unit and register-file detail to derive an
absolute integer roof from core count and clock. The working ceiling is therefore the
highest stable same-binary primitive roof combined with the compiled-pipeline and
Instruments occupancy reports.

## Probe matrix

| Probe | Question | Timed work |
|---|---|---|
| `noop` | What is the command submission/completion floor? | One SIMD group, no memory |
| `copy` | What device-memory rate does this harness sustain? | 32 logical bytes/value |
| `add`, `sub` | What do carry chains cost under streaming traffic? | One field op/value |
| `mul_wide` | What pointwise multiplication rate includes streaming traffic? | One field op/value |
| dependency chain | What is the dependent-multiply wall-time slope? | One SIMD group, varying `k` |
| chain ILP 1/2/4/8 | How much ILP hides latency before register pressure wins? | Large grid, fixed `k` |
| `u32_mad_ilp8` | What same-binary integer recurrence rate is available? | Eight scalar chains/thread |
| product5 message | What rate can a five-factor relation reach with resident equality weights? | Five dense tables to five reduced message values |
| product5 transition | What does binding cost when the next message consumes bound values in registers? | Bind five tables, persist them, and emit the next message |
| product5 threadgroups | Which legal width best balances live state, reduction cost, and residency? | GPU-active sweep from 32 through 1024 threads |

Threadgroup widths are swept over every power-of-two multiple of
`threadExecutionWidth` through the pipeline maximum. A width loses if it lowers
repeatable throughput, lowers measured occupancy without compensating reuse, or
increases tail waste. The device's maximum threadgroup-memory length is only a
per-threadgroup legality bound; it is not resident memory available to one GPU core.

The fixed default workloads are:

- 1,024 through 4,194,304 elements for the copy working-set sweep;
- 1,048,576 elements for field streaming and threadgroup probes;
- 32,768 elements times 64 iterations for saturated chains;
- one SIMD group over 8 through 512 dependent iterations for the latency slope;
- 262,144 elements times 128 iterations for the raw-integer reference.
- 65,536, 1,048,576, and 4,194,304 elements per factor for product5 comparisons;
- 4,194,304 elements per factor for the product5 threadgroup sweep.

These values may change only before a comparison series begins. Within a series,
input generation, correctness checks, Criterion configuration, and metric calculation
remain fixed.

### CPU comparison contract

The CPU/GPU comparison applies the same map, `out[i] = lhs[i] * rhs[i]`, to canonical
16-byte values and includes the output write. Inputs and outputs are allocated before
timing. GPU wall time includes command encoding, submission, execution, and the
completion wait, but not pipeline compilation, allocation, input upload, validation,
or output readback. CPU wall time includes the loop or Rayon scheduling, arithmetic,
and output writes, but not allocation or validation.

The default sizes are 65,536, 1,048,576, and 4,194,304 elements. An exact-size stress
case is selected with `JOLT_SOLINAS_BENCH_ELEMENTS` plus one explicit benchmark
family; this keeps unrelated allocations and probes out of large runs. The GPU uses
the reconnaissance winner, `mul_wide` at the compiled maximum threadgroup width. CPU
cases include one sequential thread and an `available_parallelism()` Rayon pool. A
fixed pre-sweep selected 2,048-, 8,192-, and 16,384-element chunks for the default
sizes, so the small case does not activate only four workers. The candidate order is
reversed by setting `JOLT_SOLINAS_BENCH_ORDER=cpu-first` on the second run.

The in-tree CPU comparison uses the same `jolt_field::AkitaField` as the shader and
the optimized CPU dataflow, including `AkitaAccumulator` deferred reduction for the
five-factor message. The standalone pointwise controls still answer only the cost of
one streaming operation; the sumcheck comparison uses the complete resident round
sequence and the actual optimized CPU kernel shape.

### Five-factor ceiling contract

The relation shape comes from the dense cycle path of
`InstructionReadRafSumcheck` at `c9bdcc114a6fac96e2bb6ea87c9708b896979269`
(`origin/perf/kernels-optimized`). The selected case has one combined-value factor and
four virtual-RA factors. This is a representative instantiation of that prover DAG,
not a dependency on its types or an assertion that every Jolt sumcheck has five
factors.

For `N` values in each of five structure-of-arrays factor tables, the message probe
does the following:

1. Splits the equality table as `e_out[x_out] * e_in[x_in]`, with
   `len(e_in) * len(e_out) = N / 2`.
2. Assigns one threadgroup to each `x_out` block and streams adjacent endpoint pairs.
3. Multiplies both endpoints of the first factor by `e_in[x_in]`, then evaluates the
   five-factor product at `t = 1, 2, 3, 4, infinity`.
4. Reduces within SIMD groups, then across SIMD groups, applies `e_out` once per
   message lane, and recursively reduces the block outputs to five field values.

The fused transition consumes four old values per factor, binds both adjacent pairs
at one challenge, writes the two bound values to the next-round tables, and computes
the next message from those register values. The bound values are never reread by the
same transition. A command buffer includes the main kernel and all recursive message
reductions. It excludes allocation, host-to-buffer initialization, canonicality
validation, and host readback. Inputs and output buffers remain allocated across
Criterion iterations.

The dominant exact counts are:

| Probe | Useful field multiplications | Factor-table traffic |
|---|---:|---:|
| Message at source length `N` | `11N + 5 * len(e_out)` | `80N` bytes read |
| Fused transition at source length `N` | `8N + 5 * len(e_out)` | `80N` bytes read + `40N` bytes written |
| Full sequence, ignoring split-eq lower-order terms | `27(N - 1)` | `320N - 240` bytes |

“Useful” counts the relation multiplications, including binding and equality weights;
it excludes additions, reduction machinery, addressing, and the integer operations
inside one field multiplication. Factor traffic is deliberately optimistic. It
assumes the much smaller split equality tables remain cache-resident and excludes
partial-reduction scratch. Shader-level `e_in` loads total `O(N)`, but at `N = 2^28`
the unique inner table is only 128 KiB and is reused across every outer block. The
reported factor-byte rate is therefore a lower bound on logical traffic, not a claim
about physical DRAM transactions.

The full-sequence formula corresponds to one initial message, fused transitions at
source lengths `N, N/2, ..., 4`, and the final five binds. Criterion and the fixed
evaluator now run this sequence with the real Blake2b transcript on the host, a
measured GPU/CPU cutoff, one readback, and an optimized CPU tail. Primitive message
and transition groups remain useful ceiling diagnostics.

The CPU control uses the same five structure-of-arrays tables, split-equality shape,
sample points, and Rayon partition by outer block as the optimized cycle kernel. Its
buffers are allocated before timing, and its products accumulate through
`AkitaAccumulator`. Correctness compares every round message, transcript challenge,
final table, final claim, and transcript state exactly.

## Correctness and measurement procedure

Correctness runs before every timed probe. Vectors cover zero, one, `p - 1`, carry and
borrow boundaries, maximal limb products, products near `2^256`, and deterministic
pseudorandom canonical inputs. GPU results are checked against `BigUint`; a mismatch
aborts timing. The product5 benchmark also checks the matched CPU result and bound
tables against the same oracle. Unit tests separately pin the 16-byte ABI and limb
order.

The wall-time groups measure encode, submit, GPU execution, and completion wait
together. GPU-active groups sum Metal's command-buffer start/end interval instead.
Both use precompiled pipelines and preallocated shared buffers. General probes use 10
samples, one second of warmup, and two seconds of measurement. CPU/GPU comparisons
and GPU-active roofs use 20 samples, two seconds of warmup, and four seconds of
measurement. Theoretical occupancy, measured occupancy, ALU limiting, register
limiting, and cache-counter evidence still require an Instruments capture of the best
configuration in each family.

Retained comparisons use a quiet machine on AC power. Candidate order is inverted in
a second run. A result is promoted only when its confidence intervals separate in
both orders and the Instruments explanation agrees with the direction of the change.

Expected outcome: independent chains improve throughput until carry/multiply latency
is hidden, after which added ILP plateaus or reduces occupancy. Falsifying outcomes
include a streaming kernel already at the copy roof, a reduced pipeline thread limit
as ILP grows, or register spilling. Those outcomes reject an optimization; they do
not justify changing the evaluator.

## Commands

```bash
cargo nextest run -p jolt-kernels --features metal --cargo-quiet
cargo bench -p jolt-kernels --features metal --bench metal_solinas

# Criterion accepts a substring filter for a focused run.
cargo bench -p jolt-kernels --features metal --bench metal_solinas -- dependency_chain
cargo bench -p jolt-kernels --features metal --bench metal_solinas -- cpu_gpu_mul_wall
cargo bench -p jolt-kernels --features metal --bench metal_solinas -- gpu_active
JOLT_SOLINAS_BENCH_ORDER=cpu-first cargo bench -p jolt-kernels --features metal \
  --bench metal_solinas -- cpu_gpu_mul_wall

# Five-factor ceiling probes and threadgroup sweep.
JOLT_SOLINAS_BENCH_FAMILY=product5 cargo bench -p jolt-kernels --features metal \
  --bench metal_solinas -- --noplot
JOLT_SOLINAS_BENCH_FAMILY=product5-threadgroups \
  cargo bench -p jolt-kernels --features metal --bench metal_solinas -- --noplot
JOLT_SOLINAS_BENCH_ELEMENTS=4194304 JOLT_SOLINAS_PRODUCT5_THREADS=128 \
  JOLT_SOLINAS_BENCH_FAMILY=product5-message \
  cargo bench -p jolt-kernels --features metal --bench metal_solinas -- --noplot

# Complete transcript-driven dense cycle sequence against optimized Akita CPU.
JOLT_SOLINAS_BENCH_ELEMENTS=16777216 \
  JOLT_SOLINAS_BENCH_FAMILY=instruction-read-raf-cycle \
  JOLT_METAL_CUTOFF_LOG2=16 \
  cargo bench -p jolt-kernels --features metal --bench metal_solinas -- --noplot

# Exact-size GPU-only stress cases; each command initializes only that family.
JOLT_SOLINAS_BENCH_ELEMENTS=268435456 JOLT_SOLINAS_BENCH_FAMILY=gpu-active-copy \
  cargo bench -p jolt-kernels --features metal --bench metal_solinas -- --noplot
JOLT_SOLINAS_BENCH_ELEMENTS=268435456 JOLT_SOLINAS_BENCH_FAMILY=gpu-active-mul \
  cargo bench -p jolt-kernels --features metal --bench metal_solinas -- --noplot
JOLT_SOLINAS_BENCH_ELEMENTS=268435456 JOLT_SOLINAS_BENCH_FAMILY=gpu-wall \
  cargo bench -p jolt-kernels --features metal --bench metal_solinas -- --noplot

# Exact-size relation-shaped stress cases. Run separately to bound peak memory.
JOLT_SOLINAS_BENCH_ELEMENTS=268435456 JOLT_SOLINAS_BENCH_FAMILY=product5-message \
  cargo bench -p jolt-kernels --features metal --bench metal_solinas -- --noplot
JOLT_SOLINAS_BENCH_ELEMENTS=268435456 JOLT_SOLINAS_BENCH_FAMILY=product5-transition \
  cargo bench -p jolt-kernels --features metal --bench metal_solinas -- --noplot
```

The benchmark prints the Metal device, macOS version, offset, maximum buffer length,
maximum threadgroup memory, and each pipeline's execution width, maximum threads, and
static and dynamically selected threadgroup memory before timing.

## Current Akita cycle-sequence result

The complete Criterion workload uses five `AkitaField` tables, the optimized CPU
deferred-accumulation message, host Blake2b Fiat-Shamir, resident Metal ping-pong
tables, a `2^16` CPU cutoff at the retained large size, and one final readback. Setup,
pipeline compilation, and allocation are outside all cases. `metal_copied_handoff`
includes resetting the initial five tables into the shared Metal buffer;
`metal_direct_handoff` excludes that copy and represents a producer materializing
directly into the resident allocation, as the real stage-5 integration does.

| Elements per factor | Optimized CPU | Metal direct handoff | Direct speedup | Metal copied handoff | Copied speedup |
|---:|---:|---:|---:|---:|---:|
| `2^16` | 4.6745 ms | 4.1793 ms | 1.12x | 4.2382 ms | 1.10x |
| `2^24` | 130.85 ms | 26.898 ms | 4.87x | 50.277 ms | 2.60x |

At `2^24`, the same logical `27(N-1)` useful multiplication count gives 3.462
Gmul/s for CPU, 16.841 Gmul/s for the direct-handoff hybrid, and 9.010 Gmul/s when
the 1.25-GiB copied handoff is included. The fixed evaluator's retained run measured
about 5.4x CPU/GPU-active separation and 3.92x direct-handoff wall speedup; the fresh
Criterion process above measured 4.87x. The range is process-level wall variation,
not a correctness difference.

The production `PrepareKernel` path does not first build a dense host table and copy
it. Its first cycle bind computes each pending table value directly into the shared
Metal allocation. That materialization is common work with the CPU implementation,
but the full stage-level performance still needs a representative prover profile;
the direct-handoff number is therefore a cycle-tail result, not an end-to-end prover
claim.

## Historical pre-Akita reconnaissance

The tables below were collected earlier on the same M4 Max for the experimental
field `p = 2^128 - 275`. They document why pointwise maps were rejected as the main
architecture and why fusion was selected. They are not performance evidence for the
current Akita modulus or CPU field implementation.

| Elements | Optimized CPU, 1 thread | Optimized CPU, 16 threads | GPU wall, two process runs | GPU / 1 thread | GPU / 16 threads |
|---:|---:|---:|---:|---:|---:|
| 65,536 | 0.523 G/s | 0.968 G/s | 0.512–0.659 G/s | 0.98–1.26x | 0.53–0.68x |
| 1,048,576 | 0.521 G/s | 3.751 G/s | 2.295–4.405 G/s | 4.41–8.46x | 0.61–1.17x |
| 4,194,304 | 0.525 G/s | 4.550 G/s | 4.874–6.693 G/s | 9.28–12.74x | 1.07–1.47x |
| 268,435,456 | 0.491 G/s | 5.607–5.779 G/s | 9.477–9.501 G/s | 19.30–19.36x | 1.64–1.69x |

The 4,194,304-element GPU-active rate was 9.70 G/s. Its 48 logical bytes per
result correspond to 433.6 GiB/s, while the same-size active copy roof was
454.3 GiB/s. Pointwise multiplication therefore reaches about 95.5% of this harness's
logical copy roof. For this working set, occupancy work cannot materially improve the
streaming kernel unless it also changes traffic; the kernel is already bandwidth
bound. Occupancy remains important for compute-dense chains and future fused kernels.

The 1,048,576-element active rate was bimodal across runs, ranging from about 4.0 to
15.4 G/s, while the 4,194,304-element case was much steadier. The 48 MiB logical
working set sits on a cache-state boundary; it is not a sound saturation size. Wall
time also has distinct process-level states even at 4,194,304 elements. For that
reason this table is exploratory evidence, not a promoted performance baseline. The
safe claim within the default sweep is that Metal overtakes the tuned 16-thread CPU
only at its largest working set, by 1.07–1.47x wall time; it is not an unconditional
GPU win.

### 2^28-element stress case

The 268,435,456-element case was run separately after the exploratory series. Each
field buffer is exactly 4 GiB. The pointwise multiply uses three Metal buffers, or
12 GiB, and retains two 4-GiB Rust input vectors during measurement, for a nominal
20-GiB process footprint. The M4 Max reported a runtime `maxBufferLength` of
86,586,540,032 bytes (80.64 GiB), so this is a legal single-buffer dispatch rather
than a chunked or multi-buffer surrogate. After the runs, macOS reported 97% free
memory pressure, no compressed pages, and zero swap I/O.

| Metric | Process run 1 | Process run 2 |
|---|---:|---:|
| GPU-active multiply | 27.913 ms, 9.617 G/s | 28.012 ms, 9.583 G/s |
| GPU wall multiply | 28.325 ms, 9.477 G/s | 28.253 ms, 9.501 G/s |
| Optimized CPU, 16 threads | 46.451 ms, 5.779 G/s | 47.872 ms, 5.607 G/s |
| Optimized CPU, 1 thread | 546.85 ms, 0.491 G/s | not repeated |

One GPU-active copy run took 19.017 ms and reached 420.68 logical GiB/s.
Multiplication corresponded to 428.4–429.9 logical GiB/s at 48 bytes per result,
within 2.2% of the copy control. The fact that the logical multiply rate is slightly
higher does not mean it exceeded physical bandwidth: the byte counts describe
shader-level values, not hardware transactions or cache-line traffic. It does show
that field arithmetic does not measurably lower the large streaming rate. Pointwise
multiplication remains bandwidth-bound at this size.

GPU wall throughput varied by 0.25% between processes and GPU-active throughput by
0.4%. Host-visible overhead above GPU-active time was 0.24–0.41 ms, or 0.86–1.48%.
The CPU control varied by about 3%, so the all-core GPU advantage is retained as the
conservative 1.64–1.69x range rather than a point estimate. This stress case is
revalidated evidence for one large pointwise map; it does not resolve the cache-state
variance at the smaller default sizes.

### Five-factor ceiling at 2^28

Two quick Criterion process runs, with candidate order reversed, use `N = 268,435,456`
source values per factor. The message retains a 20-GiB Rust input plus a 20-GiB Metal
buffer. The transition also
allocates a 10-GiB Metal bound buffer and a 10-GiB CPU bound buffer, for a nominal
60-GiB comparison footprint. There is no per-iteration allocation or host readback.

| Phase | GPU wall | GPU active | Portable CPU, 16 threads | GPU wall / CPU |
|---|---:|---:|---:|---:|
| Five-factor message | 90.21–91.35 ms, 32.33–32.73 Gmul/s | 90.33–91.25 ms, 32.36–32.69 Gmul/s | 1.379–1.386 s, 2.131–2.142 Gmul/s | 15.09–15.36x |
| Fused bind + next message | 89.09–89.82 ms, 23.91–24.10 Gmul/s | 88.82–89.20 ms, 24.08–24.18 Gmul/s | 956.2–969.5 ms, 2.215–2.246 Gmul/s | 10.65–10.88x |

The message reads 20 GiB of factor data, corresponding to 219–222 GiB/s in the
optimistic factor-only model. The transition reads 20 GiB and writes 10 GiB,
corresponding to 334–337 GiB/s. The latter is about 79–80% of the measured
420.68-GiB/s copy roof; the message reaches only 53%. This is the intended result of
fusion: arithmetic intensity rises enough that pointwise bandwidth is no longer the
only constraint.

Assuming those large-size rates remain constant as tables shrink, one initial message
plus all fused transitions takes approximately

```text
90.21–91.35 ms + 2 * (89.09–89.82 ms) = 268.39–270.99 ms,
```

or 26.7–27.0 billion useful field multiplications per second for the `27(N - 1)`
core work. Applying the same geometric projection to the portable CPU control gives
3.291–3.325 seconds and a diagnostic 12.14–12.39x speedup. This is not an end-to-end
prover claim: the projection omits transcript work, the final short tail, CPU/GPU
handoff, and any other relation. It also does not use the pending optimized CPU field
implementation.

The absolute optimistic traffic floor is stronger than the measured kernel result.
Moving `320N - 240` factor bytes at 420.68 GiB/s takes 190.2 ms and permits 38.1
Gmul/s. The projected kernels take 268–271 ms, move factor bytes at an aggregate
295–298 GiB/s, and reach 70–71% of that roof. Closing the remaining 78–81 ms is the
concrete optimization target; a 5x claim against an optimized CPU is unsupported
until the matched `jolt-field` control is measured.

### Product5 occupancy and schedule selection

Both retained pipelines compile with execution width 32, a 1024-thread legal maximum,
and zero static threadgroup memory. Reduction scratch is allocated dynamically as
`80 * (threads / 32)` bytes: 320 bytes for the 128-thread message default and 160
bytes for the 64-thread transition default. Those allocations are 0.98% and 0.49% of
the 32-KiB per-threadgroup legal limit; even the 1024-thread sweep point requests only
2,560 bytes. This makes scratch capacity an unlikely limiter, but the public limit is
not a resident-memory budget and does not reveal allocation granularity. It cannot by
itself prove occupancy.

The `2^22` GPU-active sweep selected 128 threads for the message at 29.3–29.6 Gmul/s
and 64 threads for the transition at 21.0–21.1 Gmul/s. Widths of 512 and 1024 lose
materially, consistent with live-state or residency pressure rather than insufficient
grid parallelism. A SIMD-cohort candidate distributed the five sample evaluations
across neighboring lanes to reduce per-thread state, while loading every factor only
once. Its best rates were 14.05 Gmul/s for the message and 16.42 Gmul/s for the
transition. Shuffle and shared-reduction overhead dominated the register saving, so
that candidate was removed.

The runtime resource report cannot expose register allocation, resident threadgroup
memory, or actual active SIMD groups. The claim that the retained widths reach
theoretical occupancy therefore remains pending an Instruments capture. Throughput
is measured; occupancy is not inferred from it.

## Initial reconnaissance

A `--quick` run on 2026-08-03 checked the shape of the experiment but is not a
retained baseline. It used too few samples, was not order-inverted, and preceded the
copy working-set sweep.

| Observation | Quick result |
|---|---|
| Compiled limits, every probe | execution width 32, maximum threadgroup width 1024, static threadgroup memory 0 |
| No-op command wall time | about 98–141 us across warm and cold quick runs |
| Copy at 1,048,576 elements | about 112–115 GiB/s logical wall throughput |
| Streaming `wide`, width 256 | about 3.29–3.39 billion field multiplications/s |
| Best observed streaming `wide` | about 3.91–4.09 billion/s at width 1024 |
| Saturated `wide`, ILP 1 / 8 | about 5.13 / 5.86 billion field multiplications/s |

The removed split schedule reached only about 3.13–3.49 billion streaming and
4.10–4.73 billion saturated multiplications/s in the same quick runs, versus
3.29–4.09 and 5.13–5.86 billion/s for the widening schedule. This consistent loss is
enough to stop spending benchmark and maintenance budget on it; the values are
historical rejection evidence, not a retained baseline.

Three other conclusions are safe at this stage. Static threadgroup memory is not the
limiter. The non-monotone ILP and threadgroup results need a register/occupancy
capture; legal threadgroup width alone does not explain them. The command floor is
also a large part of short streaming measurements. The active copy and multiply
probes now separate it from the bandwidth ceiling.

This machine currently has the Command Line Tools but not `xctrace` or the offline
Metal compiler. Occupancy acceptance therefore remains pending an Xcode Instruments
capture; runtime properties cannot substitute for it.

## Acceptance and handoff

- [x] Metal unit and integration tests pass against the independent oracle on Apple
      Silicon; non-Metal feature combinations remain unchanged.
- [x] Criterion validates every timed kernel before measuring it.
- [x] Criterion reports the device and compiled pipeline limits.
- [x] Criterion distinguishes host wall time from Metal command-buffer GPU-active
      time.
- [x] Pointwise and five-factor workloads use `jolt_field::AkitaField` on the CPU.
- [x] The five-factor GPU and CPU paths agree with the same independent `BigUint`
      oracle before timing.
- [x] Product5 scratch is dynamically sized and remains below 1% of the legal
      per-threadgroup limit at each retained default.
- [x] The complete cycle evaluator uses the optimized Akita deferred-accumulation CPU
      dataflow and checks exact host-transcript parity.
- [x] The real `PrepareKernel` path matches the optimized CPU kernel round by round
      and passes the modular Akita end-to-end prover/verifier test.
- [ ] The retained widening multiplication has an Instruments capture explaining its
      occupancy and limiting resource.
- [ ] The compute-dense winner reaches the pipeline's theoretical occupancy, or the
      measured limiting resource defines and explains a lower reachable ceiling.
- [ ] A retained run explains and controls the observed process-level wall-time and
      cache-state variance.

`SUMCHECKS.md` defines how actual Jolt layouts, transcript flow, and round scheduling
consume the backend. Occupancy capture remains a hardware-analysis task; it does not
block exactness testing of additional slots.

## References

- [Apple: finding Metal GPU occupancy](https://developer.apple.com/documentation/xcode/finding-your-metal-apps-gpu-occupancy)
- [Apple: `GPUStartTime`](https://developer.apple.com/documentation/metal/mtlcommandbuffer/gpustarttime)
- [Apple: `threadExecutionWidth`](https://developer.apple.com/documentation/metal/mtlcomputepipelinestate/threadexecutionwidth)
- [Apple: `maxBufferLength`](https://developer.apple.com/documentation/metal/mtldevice/maxbufferlength)
- [Apple: Metal implementation limits](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf)
- [Apple: measuring memory bandwidth](https://developer.apple.com/documentation/xcode/measuring-the-gpu-s-use-of-memory-bandwidth)
