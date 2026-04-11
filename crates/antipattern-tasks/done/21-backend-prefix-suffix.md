# Task 21: Route PrefixSuffix Through Backend

## Status: TODO

## Anti-Pattern
The entire `prefix_suffix.rs` (834 lines) is a CPU-only state machine. All 4 lifecycle ops bypass the backend completely:

- **PrefixSuffixInit**: Builds eq tables, expanding tables, suffix accumulators, groups cycles by table kind — all `Vec<F>` on CPU
- **PrefixSuffixBind**: Updates expanding tables, binds suffix/Q/P polys via `bind_in_place()` — CPU halving loops
- **PrefixSuffixReduce**: Computes `[eval_0, eval_2]` from read-checking + RAF prefix×suffix dot products — CPU field arithmetic
- **PrefixSuffixMaterialize**: Builds RA poly vectors and combined_val from accumulated state — CPU loop over all cycles

This is the instruction lookup address phase: 128 rounds, each calling Bind + Reduce. It's likely the single biggest prover bottleneck and gets zero GPU acceleration.

## Fix
Add an opaque `PrefixSuffixState` associated type to `ComputeBackend`:

```rust
trait ComputeBackend {
    type PrefixSuffixState<F: Field>: Send + Sync;

    fn ps_init<F: Field>(
        &self,
        config: &PrefixSuffixConfig,
        challenges: &[F],
        trace: &LookupTraceData,
    ) -> Self::PrefixSuffixState<F>;

    fn ps_bind<F: Field>(
        &self,
        state: &mut Self::PrefixSuffixState<F>,
        challenge: F,
        round: usize,
    );

    fn ps_reduce<F: Field>(
        &self,
        state: &Self::PrefixSuffixState<F>,
        previous_claim: F,
    ) -> Vec<F>;  // [eval_0, eval_1, eval_2]

    fn ps_materialize<F: Field>(
        &self,
        state: Self::PrefixSuffixState<F>,
    ) -> Vec<(PolynomialId, Vec<F>)>;
}
```

### Why opaque state (not decomposed methods)
The internal operations have deeply interleaved data dependencies:
- Expanding tables feed condensation which feeds P construction
- Prefix checkpoints update every 2 rounds and feed into read-checking
- Suffix accumulators bind each round and feed into the dot products
- Phase transitions reset/rebuild multiple data structures simultaneously

A GPU implementation would want to keep all buffers on-device and batch operations within each round. Decomposing into 10+ fine-grained methods would force unnecessary synchronization points.

### Migration steps
1. **D1a**: Define `PrefixSuffixConfig` (extracted from `Iteration::PrefixSuffix` fields — compiler-side). Move `LookupTraceData` to `jolt-compute`.
2. **D1b**: Add the 4 trait methods + associated type to `ComputeBackend`.
3. **D1c**: Move current `PrefixSuffixState<F>` to `jolt-cpu` as the CPU implementation of `CpuBackend::PrefixSuffixState<F>`.
4. **D1d**: Update runtime to call `backend.ps_init()` / `backend.ps_bind()` / `backend.ps_reduce()` / `backend.ps_materialize()`.
5. **D1e**: Add comprehensive round-by-round parity tests.

### Metal/GPU path
The CPU backend is a direct lift. Metal initially delegates to CPU. Later, the key GPU kernels are:
- `ps_bind`: parallel bind of suffix polys (O(num_tables × 2^chunk_bits) per round)
- `ps_reduce`: parallel prefix×suffix dot products (O(2^chunk_bits × num_tables))
- `ps_materialize`: parallel RA poly construction (O(T × num_phases))

## Risk: High
834 lines of CPU code with instruction-table-specific logic. The state machine runs for 128 rounds with phase transitions every `chunk_bits` rounds. Needs careful round-by-round parity testing.

## Dependencies: Should be done LAST — all other backend tasks provide practice with the migration pattern.

## Test
```bash
cargo nextest run -p jolt-equivalence transcript_divergence --cargo-quiet
```
Plus: dedicated parity test with known trace data exercising all 128 rounds and phase transitions.
