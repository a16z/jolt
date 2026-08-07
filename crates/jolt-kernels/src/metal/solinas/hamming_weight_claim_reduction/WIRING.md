# Hamming-weight claim-reduction successor

This directory contains one unregistered implementation slice for Akita's
stage-7 `HammingWeightClaimReduction`: a one-scan, fixed-29-selector resident
histogram. The Metal source has not been compiled or measured. Nothing here is
selected by the backend, source assembler, benches, or integration tests.

## Protocol boundary

The optimized CPU source of truth is
`src/optimized/hamming_weight_claim_reduction.rs`. At the production geometry:

```text
T = 2^26 cycles
K = 2^8 address lanes
P = 16 InstructionRa + 2 BytecodeRa + 2 RamRa
    + 8 UnsignedIncChunk + 1 UnsignedIncMsb
  = 29 pushforwards
```

For selector `i`, cycle `j`, and its hot address `h_i(j)`, preparation needs

```text
G_i(k) = sum_{j: h_i(j) = k} eq(r_cycle, j).
```

Akita separates the implicit default lane. `HammingWeightPreparePlan::finish`
sets every `G_i(0)` to zero and adds the logical default through one
delta-at-zero baseline. For RA column `i`, the retained weight is

```text
W_i(k) = gamma^(3i+1) * (eq(r_bool, k) - eq(r_bool, 0))
       + gamma^(3i+2) * (eq(r_virt_i, k) - eq(r_virt_i, 0)).
```

The baseline carries `gamma^(3i)`, both default EQ terms, and the RAM
activation where applicable. The eight increment digits and signed carry use
the same centered booleanity term plus the `gamma^78 * place_value *
balanced_value(k)` decode term. Thus the address summand is exactly

```text
baseline_delta(k) + sum_i G_i'(k) * W_i(k),
G_i'(0) = 0, G_i'(k) = G_i(k) for k != 0.
```

The GPU/host boundary is only the 29 recentered 256-lane tables. The existing
host plan constructs `W_i` and the baseline, then performs all eight
degree-two, low-to-high sumcheck rounds. It also performs Fiat--Shamir:

- Stage 7 draws Hamming's single batching challenge `gamma` before proving the
  batch; Hamming is the first batch member.
- The shader never absorbs a message, draws a challenge, or binds a table.
- Output values are ordered as 16 instruction, two bytecode, two RAM, eight
  increment chunks, then increment MSB.
- Every output point is
  `[reverse(the eight sumcheck challenges) || r_cycle]`.

Changing any of that is outside this slice.

## Dominant slice

The production shader uses one SIMD group per selector: 29 SIMD groups and 928
threads per threadgroup. At log 26 the checked split is

```text
I = 2^18 inner rows
O = 2^8 outer blocks
```

Each outer block is one histogram threadgroup. It stages 512 rows at a time:

1. Threads 0--511 load each 40-byte resident row and one `E_in` field once,
   decode all 29 hot bytes, and place hot bytes and weights in threadgroup
   memory.
2. SIMD group `i` scans selector `i`. Its 32 lanes own the 256 bins as eight
   fields per lane. Bin zero is skipped.
3. Each nonempty retained bin is multiplied by `E_out[outer]` and written as a
   compact partial.
4. One 256-thread finalizer group per selector sums the 256 outer partials and
   emits a zero in bin zero.

The dynamic scratch is checked at 23,232 bytes:

```text
29 * 512 hot bytes                  14,848
512 * 16-byte staged weights         8,192
3 * 16 SIMD audit words * 4 bytes      192
                                      -----
                                     23,232
```

The isolated compile probe admits this shape on an Apple M4 Max: SIMD width 32,
a 1,024-thread pipeline limit for both entry points, and 23,232 of 32,768 bytes
of threadgroup memory. It has eight live field accumulators per thread (118,784
bytes of aggregate accumulator payload before compiler temporaries), so
admission is not occupancy evidence. Promotion still requires a
compiler/counter report showing no thread-memory spill. If it spills, do not
silently change topology; measure a two-wave/two-scan fallback against this
candidate.

## Checked ABI and launch

`HammingWeightProtocolTopology::PRODUCTION` fails closed unless every family
count and the 256-bin domain match. `HammingWeightSlicePlan` derives all buffer
lengths, byte totals, parameters, dispatch widths, and compact indices with
checked arithmetic.

| Entry point | Buffers | Exact dispatch |
| --- | --- | --- |
| `solinas_hamming_weight_register_histogram` | 0 rows, 1 `E_in`, 2 `E_out`, 3 partials, 4 audit rows, 5 status, 6 params; threadgroup 0 scratch | `O` groups of 928 threads |
| `solinas_hamming_weight_register_finalize` | 0 partials, 1 output, 2 status, 3 params | 29 groups of 256 threads |

The partial index is

```text
((outer * 29 + selector) * 255) + (bin - 1),  bin in 1..256.
```

The output index is `selector * 256 + bin`. Both kernels are encoded in one
command buffer and followed by one readback. The status allocation is cleared
before dispatch and must report zero unsupported dispatches.

Audit values are one 32-byte row per outer block, not one global `u32`
counter. Each shard contains row, mapped-PC, mapped-RAM, retained-contribution,
and occupied-bin counts. This matters at log 28: `29 * 2^28 = 7,784,628,224`
does not fit a global `u32`, while one default shard contains at most
`29 * 2^18 = 7,602,176` contributions. `HammingWeightCensus::from_audit_rows`
checks every shard, reserved/status words, and the 64-bit aggregate.

The row is the existing five-word `BooleanityRow`/`InstructionCycleRow` view:
lookup low, lookup high, RAM address plus one, fused-inc magnitude, and packed
PC/flags. The fixed selector order is:

```text
0..7    lookup high bytes, shifts 56..0
8..15   lookup low bytes, shifts 56..0
16..17  mapped PC bytes, shifts 8,0; absent is cold
18..19  remapped RAM bytes, shifts 8,0; absent is cold
20..27  centered fused-inc bytes, shifts 0..56
28      signed carry: -1 -> 255, 0 -> 0, 1 -> 1
```

The Solinas offset must be `0xffff_a7f7`. All uploaded equality fields and all
read-back masses still require the existing canonical-field validation.

## Independent oracle and tests

`unfactored_recentered_pushforwards` is the primary fixture. For each row it
evaluates `eq(r_cycle, row)` directly from every cycle coordinate and then
applies the definition above. It never constructs or accepts `E_in`/`E_out`.
It is capped at `2^16` rows so it cannot become a production allocation.

`recentered_pushforwards` separately mirrors the split topology. Unit fixtures
compare every one of the 7,424 Akita-field masses across the algorithms, cover cold
PC/RAM rows, signed increment carry, bin-zero recentering, compact index
endpoints, exact log-26 buffer sizes, fail-closed topology/length checks, and
the log-28 audit overflow case. The CPU-only suite passes all ten tests.

Integration must additionally compare the input claim, every sampled round
evaluation and polynomial, all host challenges, final claim, output values,
output points, and transcript state against
`OptimizedHammingWeightClaimReduction` in clear Akita proving mode. Because
the current feature matrix rejects `akita` with `zk`, run the generic CPU ZK
regression separately; do not claim an Akita-Metal ZK parity run.

## Log-26 work and traffic

Let `B` be mapped-PC rows, `R` remapped-RAM rows, `C` retained nonzero
contributions, and `Q` occupied `(outer, selector, bin)` histograms after field
cancellation.

```text
raw selector opportunities = 25T + 2B + 2R <= 29T
                           <= 1,946,157,056
histogram field additions  = C
outer field products       = Q <= 29 * 255 * 256 = 1,893,120
finalizer field additions  = 1,893,120
host split-EQ products     = (I - 1) + (O - 1) = 262,398
```

The previous `2(I-1)+2(O-1)` product count was stale: each EQ child pair uses
one product and one subtraction.

At log 26, including the sharded audit/status handoff, the large-buffer ledger
is:

| Quantity | Bytes |
| --- | ---: |
| resident rows, one scan | 2,684,354,560 |
| logical `E_in` loads | 1,073,741,824 |
| logical `E_out` loads | 118,784 |
| partial write + read | 60,579,840 |
| output write + read | 237,568 |
| audit/status write + read | 16,416 |
| source-issued total | 3,819,048,992 (3.556767 GiB) |
| optimistic off-chip total | 2,749,386,784 (2.560566 GiB) |

The sequence owns 34,615,312 bytes (33.012 MiB); the 2.5-GiB row allocation is
borrowed from its producer. At the retained M4 Max copy control of
451,701,710,520 B/s, the source-issued floor is 8.455 ms and its 80%-of-copy
cap is 10.569 ms. Those are off-chip controls, not a kernel latency prediction.

On the frozen census `C = 1,588,505,707`, the shader also performs 28.343 GiB
of logical threadgroup traffic: staged hot-byte writes/reads, staged weight
writes, one 16-byte weight read per retained contribution, and audit scratch.
This internal stream plus SIMD lane utilization and field-add dependencies can
dominate after global row traffic is removed. Counters and the compiled
register report are mandatory; a copy roof cannot promote this kernel.

## Gates and why 8x remains credible

The frozen equal-input production control is
`benchmark-runs/metal-piop-eval/20260806-133709-697013` at revision
`5f520c21e338632aa0bf5936ceb02be6c22fa40f`, log 26, and 16 Rayon threads. Its
optimized-CPU member samples are:

```text
545.613583, 554.614169, 525.892210, 548.702500, 555.909956 ms
```

The median is 548.702500 ms, giving exact complete-member gates:

```text
hard 5x       <= 109.740500 ms
pursue 8x     <=  68.587812 ms
GPU-active aim <= 40.000000 ms (diagnostic, never sufficient alone)
```

The deployed five-scan Metal path has a 111.646165-ms median and 4.90155x
paired median speedup, so only a small complete-member improvement is needed
for the hard floor. Eight-times is also credible enough to pursue: the frozen
standalone census requires 39.713 billion retained additions/s to fit 40 ms,
while the accepted predecessor processed the same census at 47.405 billion
retained contributions/s during its 33.509-ms median GPU-active interval. That
comparison is directional, not a forecast: the predecessor used deferred
threadgroup atomics, this shader uses lane-owned reduced accumulators, and the
production run has different thermal and working-set conditions.

## Root integration steps

1. Declare this module in `src/metal/solinas/mod.rs`, export the plan/ABI needed
   by the host adapter, and add its `SOURCE` fragment to
   `src/metal/solinas/source.rs` after `fp128.metal`.
2. Add a `SolinasMetal` preparation object that compiles both named pipelines,
   validates SIMD width, thread limit, static + dynamic threadgroup memory,
   device registry, Solinas offset, and every `HammingWeightSlicePlan` buffer
   requirement before allocating or submitting.
3. In `src/metal/hamming_weight_claim_reduction.rs`, retain the existing
   `HammingWeightPreparePlan`, require its exact 29-selector production
   schedule, split `r_cycle` into the checked `E_out` prefix and `E_in` suffix,
   borrow the same `BooleanityRows` allocation, encode both dispatches in one
   command buffer, validate status/audit/masses, and call `finish_flat`.
4. Preserve optimized CPU fallback only before command submission. After
   submission, a Metal error aborts the proof. Preserve the terminal stage-7
   row-owner identity/lifetime checks already in the adapter.
5. Add low-level parity tests against the unfactored oracle, then full kernel
   lockstep and clear Akita end-to-end tests, with generic CPU ZK regression as
   a separate gate. Extend the standard Criterion family and the fixed
   alternating production evaluator without changing their CPU denominator or
   host-Fiat--Shamir boundary.

Do not enable this path until the 928-thread pipeline is admitted without
spill, all parity gates pass, and five alternating log-26 pairs clear 5x. Run a
separate log-27 transfer gate before promotion and a log-28 capacity/parity run
before claiming that scale. The principal honest blocker is compiled resource
behavior: this static packet cannot establish it without using the machine.
