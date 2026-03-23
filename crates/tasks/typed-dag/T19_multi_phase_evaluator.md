# T19: Multi-Phase Address+Cycle Binding

**Status**: `[ ]` Not started
**Depends on**: T08 (S2), T11 (S5), T12 (S6)
**Blocks**: Full jolt-core parity
**Crate**: `jolt-zkvm`
**Estimated scope**: Large (~500 lines)

## Objective

Wire the existing `SegmentedEvaluator` and `SplitEqEvaluator` into stages
S2 (RamRW), S5 (RamRaCR, InstructionReadRaf), and S6 (Booleanity,
BytecodeReadRaf) to handle the `(address || cycle)` combined domain
correctly.

## Background

Currently these stages use cycle-only evaluation (`log_T` rounds) which
produces incorrect domain sizes for RA polynomials that live in the
combined `(address, cycle)` domain (`log_k + log_T` rounds).

jolt-core handles this with multi-phase binding:

| Instance | Phases | Existing Evaluator |
|---|---|---|
| RamRW | 3 phases (cycle→address→remaining) | `SegmentedEvaluator` chains |
| InstructionReadRaf | 2 phases (address→cycle) | `SegmentedEvaluator` |
| BytecodeReadRaf | 2 phases (address→cycle) | `SegmentedEvaluator` |
| Booleanity | 2 phases (address→cycle) | `SplitEqEvaluator` + `RaVirtualCompute` pattern |
| RamRaCR | address-only (`log_k` rounds) | `KernelEvaluator` with eq over address domain |

## Existing Infrastructure

Already built and available:

- **`SegmentedEvaluator`** (`evaluators/segmented.rs`): Chains multiple
  `KernelEvaluator` instances across phase boundaries. Transition callback
  materializes new buffers for the next phase.

- **`SplitEqEvaluator`** (`jolt-sumcheck/split_eq.rs`): Gruen-optimized
  eq polynomial that handles LowToHigh binding with √N memory.

- **`RaVirtualCompute`** (`evaluators/ra_virtual.rs`): `SumcheckCompute`
  impl for RA product-of-chunks, uses `SplitEqEvaluator` internally.

- **`RaPolynomial`** (`evaluators/ra_poly.rs`): Lazy one-hot polynomial
  representation using `SharedRaPolynomials` pattern.

## Deliverables

### 1. S2: RamReadWriteChecking (3-phase)

Wire `SegmentedEvaluator` with 3 segments:
- **Segment 1** (cycle, `phase1_num_rounds` rounds): Bind cycle variables
  with `SplitEqEvaluator`. Formula: `eq_cycle · ra · (val + γ · (inc + val))`.
- **Segment 2** (address, `phase2_num_rounds` rounds): Transition to
  address-major matrix. Bind address variables.
- **Segment 3** (remaining, `log_T + log_K - phase1 - phase2` rounds):
  Dense binding of remaining variables.

The `num_vars` for this instance becomes `log_K + log_T` (not just `log_T`).

### 2. S5: RamRaCR (address-domain only)

Currently skipped because RA polys don't match cycle-only eq table.

Fix: The eq point for RamRaCR should span the address domain. The RA polys
need to be evaluated at a `log_k`-length point, not `log_T`. The sumcheck
should have `num_vars = log_k` and operate over the address portion only.

### 3. S5: InstructionReadRaf (2-phase)

Wire `SegmentedEvaluator` or `RaVirtualCompute`:
- **Phase 1** (address, `LOG_K` rounds): RA product over address chunks.
  Uses Toom-Cook grid via `RaVirtualCompute` or `KernelEvaluator::with_toom_cook_eq`.
- **Phase 2** (cycle, `log_T` rounds): Standard eq-weighted evaluation.

Degree: `n_virtual_ra_polys + 2`.

### 4. S6: Booleanity (2-phase, address+cycle)

Currently uses cycle-only domain. Fix to use `(log_k_chunk + log_T)` rounds:
- **Phase 1** (address, `log_k_chunk` rounds): Bind address variables via
  `SplitEqEvaluator`. RA polys are bound per-chunk.
- **Phase 2** (cycle, `log_T` rounds): Bind cycle variables. Use
  `SharedRaPolynomials` for memory-efficient RA access.

Formula: `Σ_i γ^i · eq(r, x) · (ra_i(x)² - ra_i(x)) = 0`.

### 5. S6: BytecodeReadRaf (2-phase)

Wire `SegmentedEvaluator`:
- **Phase 1** (address, `log_K` rounds): Bytecode RA product over chunks.
- **Phase 2** (cycle, `log_T` rounds): Per-stage eq-weighted value lookups.

Multi-stage folding: 5 stages' bytecode lookups + 2 RAF evaluations batched
with γ-powers.

## Key Pattern

All multi-phase instances follow:
1. Build `KernelEvaluator` for Phase 1 (address or cycle domain)
2. Define transition callback that takes Phase 1 challenges and produces
   Phase 2 `KernelEvaluator`
3. Wrap in `SegmentedEvaluator`
4. Pass to `BatchedSumcheckProver` as normal `Box<dyn SumcheckCompute>`

The `SegmentedEvaluator` handles the phase switching transparently — the
`BatchedSumcheckProver` sees it as one continuous sumcheck instance.

## Reference

- RamRW 3-phase: `jolt-core/src/zkvm/ram/read_write_checking.rs:309-638`
- InstrReadRaf 2-phase: `jolt-core/src/zkvm/instruction_lookups/read_raf_checking.rs:853-917`
- BytecodeReadRaf 2-phase: `jolt-core/src/zkvm/bytecode/read_raf_checking.rs:439-662`
- Booleanity 2-phase: `jolt-core/src/subprotocols/booleanity.rs:292-540`
- SegmentedEvaluator: `crates/jolt-zkvm/src/evaluators/segmented.rs`
- SplitEqEvaluator: `crates/jolt-sumcheck/src/split_eq.rs`
- RaVirtualCompute: `crates/jolt-zkvm/src/evaluators/ra_virtual.rs`

## Acceptance Criteria

- [ ] S2 RamRW uses `log_K + log_T` rounds (3-phase SegmentedEvaluator)
- [ ] S5 RamRaCR operates over address-only domain
- [ ] S5 InstructionReadRaf uses 2-phase with RA products
- [ ] S6 Booleanity uses `log_k_chunk + log_T` rounds (2-phase)
- [ ] S6 BytecodeReadRaf uses 2-phase with multi-stage folding
- [ ] All RA polynomial evaluations use correct (addr, cycle) points
- [ ] E2E smoke test still passes
