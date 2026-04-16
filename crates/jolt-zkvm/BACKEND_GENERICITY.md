# Backend Genericity Harness

**Goal**: Fully eliminate protocol awareness from ComputeBackend trait and all backend implementations. The compiler encodes ALL protocol knowledge; backends execute generic polynomial operations.

**Analogy**: XLA/TVM — compiler lowers attention/Conv2D into matmul/reduce. Backend knows matmul/reduce but not attention/Conv2D.

**Termination condition**: Zero protocol-specific types in ComputeBackend trait, zero state machine modules in jolt-cpu, zero InstanceConfig/CpuInstanceState/LookupTraceData references in any backend or runtime code.

## Test Gate

```bash
cargo nextest run -p jolt-equivalence transcript_divergence --cargo-quiet
cargo nextest run -p jolt-equivalence zkvm_proof_accepted --cargo-quiet
cargo fmt --check -q
cargo clippy -p jolt-compiler -p jolt-compute -p jolt-cpu -p jolt-zkvm -p jolt-dory -p jolt-openings -p jolt-verifier --message-format=short -q --all-targets -- -D warnings
```

## Loop Protocol

```
1. Read this file → find CURRENT TARGET
2. Analyze the target's code (state machine, caller sites, data flow)
3. Design the increment: express as kernel(s) + materialize ops
4. Implement the change
5. Run test gate
6. Pass → mark DONE, advance to next target
7. Fail → diagnose, fix, goto 5
```

---

## Protocol Leakage Inventory

### HARD leakage (must eliminate)

| # | Item | Location | Status |
|---|------|----------|--------|
| H1 | `InstanceConfig` enum (3 protocol variants) | module.rs:230-272 | DONE — converted to plain struct, no enum dispatch |
| H2 | `CpuInstanceState` enum (3 protocol variants) | backend.rs:13-21 | DONE — deleted from backend |
| H3 | `instance_init/bind/reduce/finalize` on ComputeBackend | traits.rs:352-376 | DONE — removed from trait |
| H4 | `CpuHwReductionState` state machine | hw_reduction.rs (175 LOC) | DONE — compiled into Formula kernel |
| H5 | `CpuBooleanityState` state machine | booleanity.rs (446 LOC) | DONE — compiled into Formula kernel |
| H6 | `CpuPrefixSuffixState` state machine | prefix_suffix.rs (659 LOC) | DONE — moved to runtime, deleted from jolt-cpu |
| H7 | `LookupTraceData` in ComputeBackend | traits.rs:16-23 | DONE — moved to jolt-zkvm, jolt-instructions dep removed from jolt-compute |
| H8 | `Op::UnifiedInstance{Init,Bind,Reduce,Finalize}` | module.rs:608-635 | IN PROGRESS — runtime handles directly, pending decomposition into generic ops |
| H9 | `instance_states` map in RuntimeState | runtime/mod.rs | IN PROGRESS — typed as PrefixSuffixState, pending generic ops decomposition |
| H10 | `lookup_trace`/`bytecode_data` params in handlers | handlers.rs | DONE — moved into RuntimeState, removed from dispatch_op/execute signatures |
| H11 | `InputBinding::BytecodeVal` (protocol materialization) | module.rs:421-438, helpers.rs | TODO |

### SOFT leakage (rename pass at end)

| # | Item | Location | Status |
|---|------|----------|--------|
| S1 | Protocol-named PolynomialId variants | polynomial_id.rs | TODO |
| S2 | `DomainSeparator::UniskipPoly/RamValCheckGamma` | module.rs | TODO |
| S3 | Protocol-named Op comments | module.rs | TODO |

---

## Execution Plan

### Target 1: HwReduction → Kernel (CURRENT TARGET)

**Why first**: Simplest state machine (175 LOC). Formula is a weighted sum — directly expressible as KernelSpec Formula with Challenge factors. Proof of concept for the pattern.

**Analysis**:
- Init: Compute G[d] via eq-project, eq_bool via eq_table, eq_virt[d] via eq_table
- Reduce: `Σ_d (γ_hw·G_d + γ_bool·G_d·eq_bool + γ_virt·G_d·eq_virt_d)` — degree-2, expressible as Formula
- Bind: Standard LowToHigh bind on all buffers
- Finalize: Extract G[d][0] as evaluations

**Steps**:
1. Define KernelDef for HwReduction with appropriate Formula
2. Emit Materialize ops for G[d], eq_bool, eq_virt[d] initialization in jolt_core_module.rs
3. Replace UnifiedInstance{Init,Bind,Reduce,Finalize} with standard Kernel{Reduce,Bind} + Evaluate ops
4. Verify transcript equivalence
5. Delete CpuHwReductionState, remove HwReduction variant from InstanceConfig/CpuInstanceState

**Marks**: H4 DONE, partial H1/H2/H8

### Target 2: Booleanity → Kernel(s)

**Why second**: Medium complexity (446 LOC). Two phases with different formulas. Gruen polynomial is a mathematical primitive (degree-3 reconstruction), not protocol-specific.

**Analysis**:
- Phase 1: Gruen degree-3 over G with expanding F_table and eq_rest
- Phase 2: Gruen degree-3 over H with eq_rest
- Phase transition: Materialize H from F_table × ra_data, reinit eq_rest

**Challenge**: Expanding tables and Gruen polynomial are not standard kernel patterns. Options:
A. Add Gruen as a new Iteration variant (mathematical, not protocol)
B. Express Gruen as a Formula with auxiliary challenge slots
C. Keep custom eval function but route through kernel compilation

**Marks**: H5 DONE, partial H1/H2/H8

### Target 3: PrefixSuffix → Kernel(s)

**Why last**: Most complex (659 LOC). Deep coupling to jolt_instructions (ALL_PREFIXES, LookupTableKind, Suffixes). Lazy materialization per phase.

**Analysis**:
- RAF component: Products of p and q polynomial pairs — expressible as Formula
- Read-checking component: Prefix×suffix MLE product sums — requires lookup table evaluation
- Multi-phase: Expanding tables, prefix checkpoints, suffix re-materialization

**Challenge**: Prefix/suffix MLE evaluation uses jolt_instructions. Options:
A. Add generic "table eval" primitive to backend (parametric, not protocol-specific)
B. Pre-materialize suffix polynomials in compiler schedule
C. Keep prefix_suffix as a kernel type with table-generic compilation

**Marks**: H6 DONE, partial H1/H2/H8

### Target 4: Trait Cleanup

After all three state machines are compiled away:
1. Remove `instance_init/bind/reduce/finalize` from ComputeBackend trait (H3)
2. Delete `InstanceConfig` enum (H1)
3. Delete `CpuInstanceState` enum (H2)
4. Delete `Op::UnifiedInstance*` variants (H8)
5. Remove `instance_states` from RuntimeState (H9)
6. Remove `LookupTraceData` from traits (H7)
7. Remove `lookup_trace`/`bytecode_data` params (H10)
8. Handle `InputBinding::BytecodeVal` (H11)

### Target 5: Soft Cleanup

Rename pass for S1-S3.

---

## Progress

- Target 1 (HwReduction): DONE — hw_reduction.rs deleted, Formula kernel replaces state machine
- Target 2 (Booleanity): DONE — compiled into Formula kernel
- Target 3 (PrefixSuffix): DONE (Step 1) — moved to runtime, deleted from jolt-cpu. Pending: decompose into generic ops (Step 2)
- Target 4 (Trait Cleanup): DONE — instance_* removed from ComputeBackend, CpuInstanceState deleted, dead InstanceConfig variants deleted
- Target 5 (Soft Cleanup): NOT STARTED
