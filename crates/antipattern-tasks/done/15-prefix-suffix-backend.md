# Task 15: Make PrefixSuffix Backend-Agnostic

## Status: TODO

## Anti-Pattern
PrefixSuffixState is a CPU-only state machine that completely bypasses the ComputeBackend trait:

1. **PrefixSuffixInit**: Creates a `PrefixSuffixState<F>` on the CPU. Builds eq tables, expands lookup tables, groups cycles by table kind — all CPU-side `Vec<F>` operations.

2. **PrefixSuffixBind**: Calls `ps.ingest_challenge()` which updates CPU-side expanding tables and suffix accumulators. No backend involvement.

3. **PrefixSuffixReduce**: Calls `ps.compute_address_round()` which does the full prefix×suffix dot products on the CPU. Returns `[eval_0, eval_2]`. No backend.reduce() call.

4. **PrefixSuffixMaterialize**: Calls `ps.materialize_outputs()` which builds RA poly vectors on the CPU, then uploads to device via `backend.upload()`.

The only backend interaction is the final upload. All computation is CPU-bound. This means:
- GPU/Metal backends get zero acceleration for PrefixSuffix phases
- The entire address phase of InstructionReadRaf (the most expensive sumcheck) stays on CPU
- PrefixSuffixState holds ~O(T × num_tables × 2^chunk_bits) data that could live on-device

## Fix

### Phase 1: Extract computation into backend trait methods
Add PrefixSuffix-specific methods to ComputeBackend:

```rust
trait ComputeBackend {
    // Existing methods...
    
    /// Build expanding eq table from challenges.
    fn ps_expand_eq(&self, table: &mut DeviceBuffer, challenge: F);
    
    /// Compute address round: Σ P×Q products over all table kinds.
    fn ps_address_round(
        &self,
        prefix_polys: &[DeviceBuffer],
        suffix_polys: &[DeviceBuffer],
        eq_table: &DeviceBuffer,
    ) -> [F; 2];  // [eval_0, eval_2]
    
    /// Materialize RA polynomials from accumulated state.
    fn ps_materialize(&self, ...) -> Vec<(PolynomialId, DeviceBuffer)>;
}
```

### Phase 2: Replace PrefixSuffixState with device buffers
Instead of a CPU-side state struct, PrefixSuffix data lives in device buffers:
- Expanding tables → device buffers, updated via `ps_expand_eq`
- Suffix polys → device buffers, built once per phase
- P/Q arrays → device buffers, computed per round

The runtime ops just orchestrate backend calls, same as Dense/Segmented.

### Phase 3: Kernel-based reduce
Ultimately, `PrefixSuffixReduce` should call `backend.reduce()` with a PrefixSuffix-specific compiled kernel, just like Dense and Segmented phases do. The kernel formula (Σ P×Q) is expressible in the existing KernelSpec system.

## Current Impact
- InstructionReadRaf address phase (LOG_K=128 bits, num_phases=16, chunk_bits=8) is ~16 sub-phases × 8 rounds = 128 address rounds, all CPU-only
- This is likely the performance bottleneck for GPU backends
- The cycle phase (after materialize) already goes through backend.reduce() correctly

## Test
```bash
cargo nextest run -p jolt-equivalence transcript_divergence --cargo-quiet
```

## Risk: Medium-High
PrefixSuffix has complex state (expanding tables, prefix checkpoints, suffix accumulators). The phase-transition logic (condensing u_evals, resetting expanding tables) needs careful translation. Incremental approach: start with Phase 1, benchmark, then Phase 2.

## Dependencies: Task 12 (resolve_inputs cleanup makes the materialization boundary cleaner)
