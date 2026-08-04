# Lane W1B — st6b CPU members (bytecode read+RAF cycle port, inc-claim prepare)

## Mission

Stage 6b runs at 25% GPU @2^27 (54% @2^25) and is the second-largest stage
(13.87 s canonical @2^27). Its CPU-only members and CPU prepares are the
biggest single-stage utilization prize: port them to Metal with byte-identical
proofs. Also kills the back half of the 9.16 s st5→st6b zero-GPU seam
(st6b's first ~5.4 s are CPU prepares).

## st6b composition @2^27 (instrumented, 15.76 s stage wall)

| member | time | status |
|---|---:|---|
| BytecodeReadRafCycle prepare + rounds | 2.24 + 4.82 = 7.06 s | **CPU-only slot — your main target** |
| IncClaimReduction::prepare | 2.58 s | Metal slot exists, prepare is CPU (and page-pressure-inflated at 2^27 — see note) |
| EqPolynomial::evals(_parallel) | 2.21 s | CPU eq tables feeding members |
| TraceBackend::oracle_table / materialize_cycle | 1.63 s | host witness materialization |
| InstructionRaVirtualization round(s) | 1.90 s | device member (leave alone) |
| RamHammingBooleanity rounds | 1.34 s | device member (leave alone) |

## Targets, in order

1. **`bytecode_read_raf_cycle` full port** (prepare + prove_round device path).
   CPU impl: `crates/jolt-kernels/src/optimized/bytecode_read_raf.rs`;
   reference: `reference/bytecode_read_raf.rs`. This is the only large
   cycle-phase member with NO Metal slot. Pattern to copy: the closest existing
   cycle-phase device slots — `metal/slots/instruction_read_raf.rs` (stage-5
   read+RAF cycle: same read+RAF shape over cycle domain!) and
   `metal/slots/ra_lazy.rs`. WARNING from M5 W3 T6: the stage-5 instruction
   read+RAF kernel hit a 32-bit word-offset overflow at 2^27 (fix
   `7449411b8` widened flat-table offsets to ulong) — your kernel has the same
   table geometry risk; write the 2^27-shape unit test up front (parity at a
   synthetic large offset, not a full 2^27 prove).
2. **`inc_claim_reduction` prepare device path** —
   `metal/slots/inc_claim_reduction.rs` exists; its prepare does CPU table
   builds before device rounds. Move the table build device-side (witness rows
   are already resident in the ProofSession for other slots — check what W8
   SoA lanes / parked columns exist from st4's joint-opening prefetch,
   `crates/jolt-prover/src/stages/stage4.rs` parking and
   `metal/slots/joint_opening.rs`).
3. **Eq-table + oracle_table feed** (2.2 + 1.6 s): only as far as they're
   inputs to (1)/(2) — e.g., if the ported kernels can consume eq tables
   produced on-device, take it; do NOT build a standalone split-eq project
   (refuted in M5 for standalone value).

## Coordination

- Lane W1D (memory-pressure lane) investigates WHY IncClaimReduction::prepare
  inflates 2.58→7.3 s at the ~90 GiB tier (stage-5 arena slab ownership, parked
  W4 U1 door). You own the code of that prepare; D owns allocator/lifetime
  changes. If D concludes a lifetime fix changes your prepare's memory source,
  coordinate through the orchestrator — do not both edit the slot.
- Lane W1A owns st6a + st7 slots (booleanity_address, bytecode_read_raf_address,
  HWCR tables). Shared shader utility edits: additive only (new functions),
  never rewrite shared primitives.

## Where things live

Backend registry `crates/jolt-kernels/src/metal/mod.rs` (`fn metal()`); slots
`crates/jolt-kernels/src/metal/slots/`; shaders
`crates/jolt-kernels/src/metal/shaders/*.metal`; stage driver
`crates/jolt-prover/src/stages/stage6b.rs`. Env knobs
`JOLT_METAL_MIN_TERMS[_<KIND>]`.

## Ground rules

- Worktree `~/dev/jolt/.worktrees/gpuutil-w1b`, branch `gpu/util-w1b`. Never
  push. Commit per unit.
- **Hard gate: proof bytes identical.** Parity test per slot (device threshold
  forced low, positive dispatch count, byte compare vs CPU path). Full gate
  matrix before any retained commit (see `.journals/gpu-util.md` Protocol).
- Timed runs only under `/tmp/jolt-gpu.lock.d`. Iterate 2^22-24, confirm 2^25
  cool. NO 2^27 runs from this lane (orchestrator certifies; your 2^27-geometry
  correctness risk is covered by the synthetic-offset unit test).
- GPU-util attribution via `-F jolt-profiling/monitor`; walls without monitor.
- Kill gates: bytecode_read_raf_cycle port at 2^24 — st6b wall −15% or
  member-attributed wall −40%, else stop and report. Inc prepare device path
  judged at 2^25 (−0.3 s st6b or clearly positive util shift).

## Baselines (canonical): st6b 1.680 s @2^25, 13.874 s @2^27

Measure your own ABBA pairs at 2^24/2^25 for attribution; the 2^27 pressure
component is expected to persist until D lands — quote 2^25 numbers.

## Reporting

To `.journals/lane-reports/w1b.md` (committed) + message_parent at:
(i) decomposition of BytecodeReadRafCycle prepare+rounds into device kernels vs
host glue, (ii) parity green + first 2^24 A/B, (iii) final with 2^25 cool
confirm + commits + binary sha + gate results.
