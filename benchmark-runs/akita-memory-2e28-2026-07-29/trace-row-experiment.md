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
- A full-vector adapter is useful only as a diagnostic. Production tracing must
  feed bounded `Cycle` chunks into the row builder so the allocator never sees a
  trace-sized `Vec<Cycle>`.
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
| Raw trace ownership stays bounded | `trace_with_cycle_sink` and `JoltCpuProver::gen_from_elf` | chunk bound, construction span, RSS, and code review |
| Protocol and verifier inputs are unchanged | prover-only trace representation | host, host+ZK, and Akita committed-mode e2e verification |

## Ambiguity register

1. **Temporary dual residency.** The first adapter dropped `Vec<Cycle>` before
   proving, but allocator-retained pages kept maximum RSS high. The accepted
   path buffers at most `2^18` cycles, converts each chunk in parallel, and
   appends into a row vector whose final capacity is reserved before tracing.
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
5. **Padding.** Convert real rows as they arrive, then pad directly with
   `JoltTraceRow::default()`. Parity coverage proves that the default row equals
   the converted canonical no-op.
6. **Field-inline profile.** The cutover must compile for the supported feature
   matrix even though the target benchmark is RV64IMAC Akita.

## Minimal experiment

1. Add direct lookup/flag support for `JoltTraceRow`.
2. Port proof consumers to `&[JoltTraceRow]` / `Arc<Vec<JoltTraceRow>>`.
3. Add a bounded cycle sink for `gen_from_elf`; keep `gen_from_trace` as a
   compatibility/test adapter.
4. Watch the parity test fail before the ownership switch, then pass after it.
5. Run focused correctness and clippy checks, screen at `2^22`, and run one
   target `2^26` candidate only if the screen passes.

## Runs

Append one line per run:

`run | revision | size | construction / prove / focused phases / max RSS | verdict`

`control | d4ec43f67 | 2^26 | prove 67.184 s; max RSS 46.457 GB; swaps 0 | baseline`

`scan-probe-sequential | 17f605d04 + temporary test | 2^22 | build 931.3 ms; four Cycle scans 3.427 s; four row scans 107.0 ms | row scan passes; sequential build rejected`

`scan-probe-parallel | 17f605d04 + temporary test | 2^20 | build 25.7 ms; four Cycle scans 855.6 ms; four row scans 25.9 ms | promote to ownership implementation`

`full-vector-adapter | pre-937319abb | 2^22 | build 86.3 ms; prove 5.69-5.72 s; max RSS 14.785 GB | runtime passes; target RSS unresolved`

`full-vector-adapter | pre-937319abb | 2^26 | build 1.492 s; prove 54.65 s; max RSS 46.321 GB; swaps 0 | reject adapter: freed Cycle pages remain resident`

`bounded-cycle-sink | 937319abb | 2^22 | trace+conversion 515.8 ms; prove 5.95 s; max RSS 14.741 GB; swaps 0 | promote`

`bounded-cycle-sink | 937319abb | 2^26 | trace+conversion 7.293 s; prove 54.95 s; max RSS 44.244 GB; swaps 0 | accept`

The same-binary probe intentionally overrepresents bytecode-PC recovery in the
raw path, so its 33× scan ratio is not an end-to-end prediction. It establishes
the narrower claim needed here: the row accessors themselves are not the
bottleneck, and parallel construction has a plausible target-scale cost. The
temporary benchmark code was removed after recording these numbers.

## Outcome

The bounded sink meets both target gates. At `2^26`, maximum RSS falls by
2.213 GB relative to the 46.457 GB control. The exact retained representation
change predicts 2 GiB (2.147 GB), so the observed reduction is within 3.1% of
the byte count. The full-vector adapter saved only 0.136 GB because its
5.230 GB raw allocation remained resident after `drop`.

Trace production does not pay for the memory reduction. The adapter spent
5.781 s producing raw cycles and 1.492 s converting them, or 7.273 s total.
The bounded sink took 7.293 s, a 20 ms difference.

The proof itself improves from 67.184 s to 54.95 s (-18.2%). The largest
Perfetto deltas are:

| Span | Control | Compact rows | Change |
|---|---:|---:|---:|
| Akita commit | 24.730 s | 23.653 s | -4.4% |
| Stage 1 | 8.175 s | 3.057 s | -62.6% |
| Stage 3 | 3.588 s | 1.159 s | -67.7% |
| Stage 6a | 2.203 s | 0.969 s | -56.0% |
| Packed opening | 11.211 s | 10.961 s | -2.2% |

The improvement is an engineering consequence, not a protocol change:
proof-facing flags, values, addresses, and bytecode indexes are computed once
into a 64-byte row rather than repeatedly recovered from a 96-byte tracer enum.
Streaming witness commitment also scans the retained rows instead of replaying
the lazy emulator trace. Proof messages and verifier inputs are unchanged.

## Retained traces

- `benchmark-runs/perfetto_traces/mem-trace-row-2e22.json`
- `benchmark-runs/perfetto_traces/mem-trace-row-2e26.json`
- `benchmark-runs/perfetto_traces/mem-trace-row-adapter-2e22.json`
- `benchmark-runs/perfetto_traces/mem-trace-row-adapter-2e26.json`
