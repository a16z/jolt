# `JoltTraceRow` ownership experiment

Date: 2026-07-29 EDT

## Question

Can the prover replace its retained 96-byte tracer `Cycle` vector with the
existing 64-byte `JoltTraceRow` while preserving proof semantics and avoiding a
measurable Akita regression at K256?

This is an ownership and representation change, not a protocol change.

## Prediction and promotion gate

- Exact retained saving after padding: 32 B/cycle, or 2 GiB at `2^26` and
  8 GiB at `2^28`.
- Likely performance effect: neutral or faster in row-scanning phases because
  the permanent trace is one third smaller and final-row metadata is already
  cached.
- The initial adapter may briefly hold the source and destination vectors while
  converting, but the raw `Cycle` allocation must be released before commitment
  setup or proof working sets are allocated. The conversion window must remain
  below the later proof peak; otherwise this implementation is rejected in
  favor of a direct tracer sink.
- Reject on any proof/parity failure, a repeated focused-phase regression above
  2%, a full-prover regression above the established 0.48-second noise band, or
  a higher maximum RSS than the control.
- Promote only if the observed target-scale RSS reduction is consistent with
  the 2 GiB retained-byte prediction and the conversion itself is not a material
  end-to-end cost.

Frozen controls: K256 (`PERF_LOG_K_CHUNK=8`), the packed-one-hot protocol,
sumcheck scheduling, proof messages, guest workload, and benchmark harness.

## Claim-to-code map

| Spec claim / invariant | Code boundary | Validation |
|---|---|---|
| Every proof-facing row has a final Jolt instruction | `tracer::cycle_to_trace_row` at the trace/prover handoff | source-only rejection tests and real-trace construction |
| Logical proof columns equal the old `Cycle` derivation | `jolt_riscv::JoltTraceRow` accessors | existing real-trace parity test, expanded to all prover-used values |
| Load/store slot aliasing is valid | `CapturedState::{Load,Store}` conversion | focused LD/SD tests plus host and ZK e2e |
| Physical width is exactly 64 bytes | `JoltTraceRow` layout assertion | existing size test |
| Proof hot paths no longer retain or adapt `Cycle` | `JoltCpuProver::trace` and Stage 1–8 consumers | type-level cutover and `rg` audit |
| Raw trace ownership ends before large prover allocations | `JoltCpuProver::gen_from_trace` | construction span/RSS timeline and code review |
| Protocol and verifier inputs are unchanged | prover-only trace representation | host, host+ZK, and Akita committed-mode e2e verification |

## Ambiguity register

1. **Temporary dual residency.** The landed builder returns a second vector.
   Resolve empirically: add a named construction span, drop `Vec<Cycle>`
   immediately, and measure its local RSS. Do not infer the result from final
   vector sizes.
2. **Virtual-sequence metadata.** The 64-byte row caches the proof flags but not
   the exact `virtual_sequence_remaining` count. Consumers must use row
   accessors/cached flags and must not reconstruct a lossy
   `JoltInstructionRow`.
3. **Lookup routing.** Lookup inputs/outputs must be computed from
   `JoltTraceRow` through the canonical final instruction kind and logical
   values. A per-row bytecode-table lookup is outside this candidate.
4. **Optional register/RAM semantics.** Absent operands and non-memory rows must
   remain zero in logical polynomials while register addresses remain
   `Option<u8>` at construction boundaries.
5. **Padding.** Convert after the trace is padded so `Cycle::NoOp` becomes the
   canonical `JoltTraceRow::no_op`; include padded rows in parity coverage.
6. **Field-inline profile.** The cutover must compile for the supported feature
   matrix even though the target benchmark is RV64IMAC Akita.

## Minimal experiment

1. Add direct lookup/flag support for `JoltTraceRow`.
2. Port proof consumers to `&[JoltTraceRow]` / `Arc<Vec<JoltTraceRow>>`.
3. In `gen_from_trace`, compute values that still require raw cycles, pad the
   trace, build rows, and drop the source vector before allocating proof state.
4. Watch the parity test fail before the ownership switch, then pass after it.
5. Run focused correctness and clippy checks, screen at `2^22`, and run one
   target `2^26` candidate only if the screen passes.

## Runs

Append one line per run:

`run | revision | size | construction / prove / focused phases / max RSS | verdict`

`control | d4ec43f67 | 2^26 | prove 67.184 s; max RSS 46.457 GB; swaps 0 | baseline`
