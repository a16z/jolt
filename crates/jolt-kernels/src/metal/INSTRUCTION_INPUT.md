# Instruction-input Metal kernel

`InstructionInput` is the current Metal sumcheck slot. Witness preparation splits
each stage-1 row into a 48-byte instruction-input prefix and a 112-byte residual in
one traversal. Stage 1 consumes both buffers and drops the residual; InstructionInput
retains the original prefix, computes its native first message there, fuses the first
bind into eight dense tables, and then fuses every dense bind with the next degree-3
message. Fiat-Shamir remains on the host and the optimized CPU kernel owns the short
tail.

## Measured problem and requirements

The clean schema-6 `2^26` production holdout at `c750b0544` measured a 727.037 ms
optimized CPU service median and a 193.721 ms Metal service median, or 3.7469x. Both
proofs and every exactness/resource guard passed, but both the historical schema-1
4x gate and the optimized-first order stratum failed. The compact phase therefore
retains that revision only as infrastructure and freezes a new row-layout experiment
before any shader search.

The implementation must satisfy these requirements:

- Every round polynomial, host-transcript challenge, final table value, output
  claim, and transcript state must equal `OptimizedInstructionInputKernel`.
- One `RowsStore` traversal writes the final 48-byte compact allocation and the
  112-byte stage-1 residual directly. There is no full-domain projection, GPU copy,
  or host repack in PIOP. The compact words are `rs1`, unexpanded PC, effective
  `rs2`, immediate magnitude low/high, and the existing flags word.
- Round dispatches allocate no buffers. Transcript-independent ping-pong storage is
  allocated during `jolt_prover::backend_witness_prepare`, outside the primary PIOP
  metric and inside the evaluator's diagnostic preparation metric.
- The optimized control similarly collects its 48-byte native rows during backend
  witness preparation and reuses that exact allocation in stage 3. Production traces
  require matching preparation/use allocation identities for both backends.
- The standalone complete-member search metric includes command waits, host
  Fiat-Shamir, one dense-table readback, and the CPU tail. Shader candidates are
  ranked by complete-member Metal throughput; actual CPU speedup is decided only by
  the production PIOP gate. Fresh promotion requires 5x, and a measured 6-8x path
  remains in scope.
- Missing resident state, unsupported geometry, or a trace below the configured
  cutoff selects the optimized CPU slot before any transcript message is absorbed.

## Relation and schedule

For cycle `j`,

```text
q(j) = is_rs2(j) * rs2(j) + is_imm(j) * imm(j)
     + gamma * (is_rs1(j) * rs1(j) + is_pc(j) * upc(j))
s(j) = eq(r_product, j) * q(j).
```

The host keeps `GruenSplitEqPolynomial` and sends its current `e_in` and `e_out`
tables to shared buffers. Since `q` is quadratic, a command returns
`[q(0), q(1), q_lead]`. The host reconstructs `q(2)` and `q(3)`, applies Gruen's
linear factor, checks the previous claim, compresses the round polynomial, and draws
the next challenge. Keeping `q(1)` avoids the singular case in the two-coefficient
`gruen_poly_deg_3` reconstruction.

The device sequence has three operations:

1. The native message reads adjacent compact-row pairs. It evaluates flags and word
   values as exact signed integers, matching the CPU's `native_q_evals`, and reduces
   the constant, one, and quadratic-leading lanes.
2. The native transition reads four compact rows, binds two adjacent pairs directly
   into two values in each of eight dense tables, and forms the next message from
   those register values before storing them.
3. A dense transition reads four values from each table, binds them into two output
   values, forms the next message from the outputs, and writes the outputs to the
   alternate buffer.

Each operation and its recursive reduction share one command buffer. Once every
table has at most `cutoff_elements`, the host copies exactly
`8 * cutoff_elements * 16` bytes into the optimized kernel and finishes there.

## Capacity and ceiling

Let `T = 2^26` and let `n` denote one table's source length. One dense transition
unit consumes four positions from all eight tables. It reads 512 bytes, writes 256
bytes, performs 16 binding multiplications, and performs 18 relation/equality
multiplications for the next message. Its optimistic intensity is therefore 768
bytes per 34 useful field multiplications, or 22.6 B/mul.

| Work | Useful field multiplications | Optimistic main-state traffic |
|---|---:|---:|
| Native message | `3T + O(E_out)` | `48T` bytes read |
| Native bind plus message | `8.5T + O(E_out)` | `48T` read + `64T` write |
| All dense transitions | less than `8.5T + O(E_out)` | less than `192T` bytes |

The full prefix performs `20T - 17C` useful field multiplications when it hands off
at `C` elements per table. The compact layout moves `352T - 256C` bytes in the
conservative row-stride model, about 22 GiB at `C = 2^16`, before split-equality and
reduction scratch traffic. At the measured 420.68 GiB/s copy roof, the cache-optimistic
row floors are 7.13 ms for the native message and 16.64 ms for the native transition;
counting logical weight accesses gives 8.32 ms and 17.23 ms. The transition's
demonstrated Akita arithmetic rate instead implies about 33.9 ms. Keeping the measured
78.97 ms tail gives a roughly 124.8 ms, 5.8x planning point. A deliberately
conservative model changes only the first message by the 160/48 traffic ratio and
projects 4.19x, with both order strata above 4x. These are planning bounds, not
performance claims; the fixed target run decides promotion.

The two full dense allocations contain `8 * T/2` and `8 * T/4` field elements,
6 GiB total. Including weights and reduction scratch, the six sequence buffers use
6,443,433,984 bytes at `2^26`; the compact and residual buffers use 3 GiB and 7 GiB,
preserving the prior 10-GiB stage-1 total. Their persistent aggregate is
17,180,852,224 bytes. Admission also reserves the
1,573,024-byte stage-1 invocation peak and observes storage already retained by
Instruction RA before allocating or touching the row buffer. At `2^28`, the
persistent pair is 68,721,442,816 bytes and the stage-1 invocation peak is
3,145,888 bytes. At that scale the individual row buffers are 12 GiB and 28 GiB,
instead of one 40-GiB allocation. Each consumer is admitted independently: an
InstructionInput scratch rejection does not disable an admissible Metal stage 1,
and an unavailable stage-1 residual does not preclude the compact-only path. The
initial CPU tail is `2^16` elements per table (8 MiB readback), with neighboring
cutoffs included in the fixed search space.

## Criterion microbenchmarks

The `metal_solinas` Criterion bench has three explicit InstructionInput families:
`instruction-input-message`, `instruction-input-transition`, and
`instruction-input-service`; `instruction-input` runs all three. For example:

```sh
JOLT_SOLINAS_BENCH_FAMILY=instruction-input-transition \
JOLT_SOLINAS_BENCH_ELEMENTS=4194304 \
cargo bench -p jolt-kernels --bench metal_solinas --features metal,parallel -- --noplot
```

The phase benches construct rows, upload the compact allocation, allocate reusable
storage, validate exact output, and prime the native pipelines before timing. Metal
wall time includes weight writes, command submission, completion, reduction, and
descriptor readback. Metal-active time is a diagnostic that removes the host command
path. The native-transition benchmark establishes the preceding native message before
each sample but excludes it from that sample's timer. Its CPU control uses
preallocated table-major output, so it is an optimistic CPU phase comparison rather
than an allocation advantage for Metal.

The service benchmark compares the optimized-shape CPU mirror with the complete
resident hybrid sequence. It charges reset, all GPU waits, host Fiat-Shamir, one
dense readback, and the CPU tail while excluding common witness construction and
backend preparation. The phase benches report modeled useful field multiplications
using `3T + 3E_out` and `17T/2 + 3E_out`; complete service reports rows per second,
since both arms execute the whole service and the CPU tail makes a prefix-only
operation count misleading. Its readback scratch is allocated once outside the
Criterion loops. Default cases stop at `2^22`; `2^26` is opt-in because every
Criterion member repeats with multi-gigabyte resident state. These benches diagnose
the ceiling and individual losses. Only the alternating production evaluator can
promote a candidate.

## Validation and promotion

The fixed evaluator compares complete optimized and hybrid sequences with a Blake2b
host transcript. It checks native-message parity before timing, every round message
and challenge, the eight final values and typed output claims, transcript state,
resident-buffer identity, exact readback bytes, command completion, zero round
allocations, and finite wall/GPU-active timings. Schema 7 additionally requires
exactly `T` witness extractions, `T` compact and residual writes, one compact
allocation, zero copy/repack bytes or dispatches, a 48-byte ABI, and one compact ID
from production through stage 1, primer, and stage 3. Small adversarial rows cover
signed `i128` immediates, selector combinations, loads, simultaneous synthetic
load/store flags, and values that would expose a split-row mismatch.

The rejected five-pair holdout is tracked in
`autoresearch/evidence/instruction_input_c750b0544_rejected.json`: local service was
3.7469x, the order strata were 3.6971x and 3.7901x, PIOP was 2.2512x, and PIOP plus
backend preparation was 2.1548x. The compact candidate gets a fresh clean holdout;
the old result cannot promote it.

The immutable a2 baseline at `2^26`, `256/128/128` threadgroup widths, and a `2^16`
CPU cutoff measured a 157.908 ms controller-level complete-member Metal median and
424.988 million rows/s. Its 25 CPU controls had an 814.395125 ms pooled median but a
4.31% relative MAD, while the five process-level Metal throughputs had a 0.21%
relative MAD. A dense transition width of 64 reduced Metal wall time by less than 1%,
but CPU drift made the old paired score appear 7.09% better and falsely accepted it.

Schema v3 ranks candidates by `814395125 / complete_member_metal_ns`. The numerator
is the pooled a2 CPU median; the 25-sample vector's compact-JSON SHA256 is
`59f9946b7d1a3c05d3094528e853d2228ae5ec0d94a5dae2c63d5713a560a966`. This ratio
is a fixed normalization of Metal throughput, so it has the same candidate ordering
and relative changes as million rows/s. It is not a contemporaneous speedup claim.
Live CPU runs remain exactness and drift controls, and the separate-process
production gate remains the only speedup arbiter.

Schema v4 preserves that score and the frozen CPU reference while correcting the
resource model: only the 48-byte compact row is resident during trials; the standalone
evaluator's temporary 160-byte source row is charged only to sequence-setup peak.

A real `2^18` traced smoke run validates the serialized contract: the CPU preparation
and stage-3 identities match, the Metal preparation/stage-1/stage-3 identities match,
and cutoff `2^10` produces one readback followed by nine CPU round spans and one
CPU finish span. This is protocol and telemetry evidence, not target-scale performance
evidence.

The search uses matched CPU/Metal protocol tapes at `2^26`, a fixed empirical noise
gate, and one accepted parent. A baseline showed a repeatable 170--250 ms outside GPU
timestamps when a multi-gigabyte CPU trial immediately preceded a Metal trial. The
fixed schedule therefore runs all CPU controls first, performs one exact full-sequence
Metal residency warmup, and then runs the timed Metal trials back-to-back. Resident
rows and sequence storage are materialized only after the CPU batch. The warmup is
reported and charged to the GPU budget but excluded from the primary metric. This
single-process steady-state result ranks shader candidates; it is neither a first-use
latency claim nor a replacement for the separate-process production control.

The a2 width sweep rejected native-message 128 and native-transition 64. Dense 64's
sub-percent change did not clear the corrected 3% gate, so the v4 source phase resets
to production defaults `256/128/128`. The Rust allocation and telemetry wrapper is
frozen; an algebra or evaluator change starts another run. The normalized search
score has no absolute speedup threshold. Under this existing schema-1 contract, a
locally accepted parent proceeds to five alternating production PIOP pairs with both
proofs verified, where 4x is enforced on
the actual contemporaneous CPU/Metal measurements. Production reports kernel-service
spans separately; those spans omit shared sumcheck-driver Fiat-Shamir, so their metric
is explicitly named `instruction_input_kernel_service_speedup`. The PIOP span remains
the end-to-end arbiter. If the production control cannot clear 4x, the Metal slot is
removed while the negative result remains in the ledger.

That contract is resume-only. A fresh v2 successor must enforce the current 5x floor
and complete its sealed holdout and log-27 transfer before promotion.

## Implementation map

| Requirement | Code unit | Required check |
|---|---|---|
| Lossless direct row production | `InstructionInputRow` plus `SpartanOuterUniskipResidualRow` | 48 + 112 = 160 bytes, one extraction, split Stage-1 oracle parity |
| Native and dense device algebra | `InstructionInputSequence` and its Metal entry points | three descriptors and reconstructed `q(t)` values equal CPU at every round |
| Host protocol and CPU tail | `MetalInstructionInputKernel` plus optimized-kernel offload hooks | messages, challenges, final values, and claims equal |
| No PIOP allocation of full buffers | aggregate row/sequence preflight and `InstructionInputSequenceStorage` prepared by `MetalBackend` | exact byte geometry, admission boundary, and allocation count |
| Search and promotion | fixed example evaluator, autoresearch template, production PIOP evaluator | closed result schema, exact guards, alternating pairs |

The split preserves all original stage-1 algebra: two residual memory words plus the
compact effective `rs2` reconstruct load, store, and ordinary rows without assuming
protocol-valid flag combinations. Threadgroup widths and the CPU cutoff remain
search parameters after the row architecture freezes. Aggregate working-set
arithmetic and fallback selection are covered by deterministic tests; target-device
admission remains an observed production-run guard. A device failure after a message
has entered the transcript returns `SumcheckError::ComputeBackend` and does not retry.

## Alternatives

`BooleanityAddressPhase` has slightly more PIOP share but requires 29 exact 256-bin
histograms; its conservative ceiling is limited by threadgroup atomic traffic.
`OuterRemainder` has clean dense rounds, but its 2.55-billion field/scalar first
materialization and 188 ms CPU opening walk require a larger all-or-nothing port.
Instruction input has no terminal opening walk and provides the strongest current
case for a conservative speedup above the 5x promotion floor.
