# Instruction-input Metal kernel

`InstructionInput` is the current Metal sumcheck slot. The kernel reuses the
stage-1 packed row buffer, computes the native first message on that buffer, fuses the
first bind into eight dense tables, and then fuse every dense bind with the next
degree-3 message. Fiat-Shamir remains on the host and the optimized CPU kernel owns
the short tail.

## Measured problem and requirements

At a padded `2^26` Fibonacci trace, the pre-integration profile gave median times of
49.265 ms for native-row collection, 718.940 ms for the 26 rounds, and less than 1 us
for finish plus output claims. The first message cost 160.643 ms, the first bind and
next message cost 306.025 ms, and the remaining rounds cost about 248 ms. The fair
primary boundary now moves the 49.265 ms transcript-independent collection before
PIOP in both arms, so 718.940 ms is the provisional CPU denominator until the clean
target baseline replaces it.

The implementation must satisfy these requirements:

- Every round polynomial, host-transcript challenge, final table value, output
  claim, and transcript state must equal `OptimizedInstructionInputKernel`.
- The stage-1 `SpartanOuterUniskipRows` allocation is reused without uploading the
  trace again. Four unused bits in its flags word will hold the operand selectors.
  Its existing values are sufficient: loads canonically have `rs2 = 0`, and the
  packed row otherwise retains `rs2` in `slot1`.
- Round dispatches allocate no buffers. Transcript-independent ping-pong storage is
  allocated during `jolt_prover::backend_witness_prepare`, outside the primary PIOP
  metric and inside the evaluator's diagnostic preparation metric.
- The optimized control similarly collects its 48-byte native rows during backend
  witness preparation and reuses that exact allocation in stage 3. Production traces
  require matching preparation/use allocation identities for both backends.
- The standalone complete-member search metric includes command waits, host
  Fiat-Shamir, one dense-table readback, and the CPU tail. Shader candidates are
  ranked by complete-member Metal throughput; actual CPU speedup is decided only by
  the production PIOP gate. The minimum remains 4x, the working target is 5x, and a
  measured 6-8x path remains in scope.
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

1. The native message reads adjacent packed-row pairs. It evaluates flags and word
   values as exact signed integers, matching the CPU's `native_q_evals`, and reduces
   the constant, one, and quadratic-leading lanes.
2. The native transition reads four packed rows, binds two adjacent pairs directly
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
| Native message | `3T + O(E_out)` | `160T` bytes read |
| Native bind plus message | `8.5T + O(E_out)` | `160T` read + `64T` write |
| All dense transitions | less than `8.5T + O(E_out)` | less than `192T` bytes |

The full prefix performs `20T - 17C` useful field multiplications when it hands off
at `C` elements per table. It moves `576T - 256C` bytes in the conservative
full-row-stride model, about 36 GiB at the initial `C = 2^16`, before
split-equality and reduction scratch traffic. At the machine's 546 GB/s peak this is
a 70.8 ms physical lower bound. The more relevant retained controls sustain about
220 GiB/s for a message and 334-337 GiB/s for fused transitions, which projects a
130-160 ms prefix before the CPU tail. Against the fair provisional denominator, the
4x and 5x complete-member budgets are 179.7 ms and 143.8 ms respectively. These are
planning bounds, not performance
claims; target-size measurements decide whether the slot is retained.

The two full dense allocations contain `8 * T/2` and `8 * T/4` field elements,
6 GiB total. Including weights and reduction scratch, the six sequence buffers use
6,443,433,984 bytes at `2^26`; the 160-byte packed rows use 10,737,418,240 bytes.
Their persistent aggregate is 17,180,852,224 bytes. Admission also reserves the
1,573,024-byte stage-1 invocation peak and observes storage already retained by
Instruction RA before allocating or touching the row buffer. At `2^28`, the
persistent pair is 68,721,442,816 bytes and the stage-1 invocation peak is
3,145,888 bytes. The 40-GiB row allocation is also checked independently against
`maxBufferLength`; either limit selects CPU before row construction. Rejection
discards Metal row residency for both stage 1 and instruction input, then prepares
the CPU rows before PIOP. The initial CPU tail is `2^16` elements per table (8 MiB
readback), with neighboring cutoffs included in the fixed search space.

## Validation and promotion

The fixed evaluator compares complete optimized and hybrid sequences with a Blake2b
host transcript. It checks native-message parity before timing, every round message
and challenge, the eight final values and typed output claims, transcript state,
resident-buffer identity, exact readback bytes, command completion, zero round
allocations, and finite wall/GPU-active timings. Small adversarial rows cover signed
`i128` immediates, selector combinations, loads with canonical zero `rs2`, and values
that would expose a packed-row mismatch.

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
sub-percent change did not clear the corrected 3% gate, so the v3 source phase resets
to production defaults `256/128/128`. The Rust allocation and telemetry wrapper is
frozen; an algebra or evaluator change starts another run. The normalized search
score has no absolute speedup threshold. A locally accepted parent proceeds to five
alternating production PIOP pairs with both proofs verified, where 4x is enforced on
the actual contemporaneous CPU/Metal measurements. Production reports kernel-service
spans separately; those spans omit shared sumcheck-driver Fiat-Shamir, so their metric
is explicitly named `instruction_input_kernel_service_speedup`. The PIOP span remains
the end-to-end arbiter. If the production control cannot clear 4x, the Metal slot is
removed while the negative result remains in the ledger.

## Implementation map

| Requirement | Code unit | Required check |
|---|---|---|
| Lossless row reuse | `SpartanOuterUniskipRow` selector bits and instruction-input accessors | packed/native row parity, including load `rs2` |
| Native and dense device algebra | `InstructionInputSequence` and its Metal entry points | three descriptors and reconstructed `q(t)` values equal CPU at every round |
| Host protocol and CPU tail | `MetalInstructionInputKernel` plus optimized-kernel offload hooks | messages, challenges, final values, and claims equal |
| No PIOP allocation of full buffers | aggregate row/sequence preflight and `InstructionInputSequenceStorage` prepared by `MetalBackend` | exact byte geometry, admission boundary, and allocation count |
| Search and promotion | fixed example evaluator, autoresearch template, production PIOP evaluator | closed result schema, exact guards, alternating pairs |

The row ABI decision is resolved by `JoltTraceRow`'s canonical load invariant:
loads have no `rs2`, hence `rs2_value() == 0`; other rows retain `rs2` in the packed
slot. Threadgroup widths and the CPU cutoff are intentionally unresolved performance
parameters in the first immutable run. Aggregate working-set arithmetic and fallback
selection are covered by deterministic tests; target-device admission remains an
observed production-run guard. A device failure after a message has entered the
transcript returns `SumcheckError::ComputeBackend` and does not retry.

## Alternatives

`BooleanityAddressPhase` has slightly more PIOP share but requires 29 exact 256-bin
histograms; its conservative ceiling is limited by threadgroup atomic traffic.
`OuterRemainder` has clean dense rounds, but its 2.55-billion field/scalar first
materialization and 188 ms CPU opening walk require a larger all-or-nothing port.
Instruction input has no terminal opening walk and provides the strongest current
case for a conservative speedup above the 4x promotion floor.
