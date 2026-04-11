# Task 17: Route Input Materializations Through Backend

## Status: TODO

## Anti-Pattern
`resolve_inputs()` (~230 lines) constructs kernel input buffers via CPU field arithmetic for 6 `InputBinding` variants, then uploads the result. The backend only sees the upload — all construction logic is invisible to non-CPU backends.

Variants and their weight:
- `ScaleByChallenge`: element-wise scalar multiply, ~10 lines
- `Transpose`: matrix transpose, ~15 lines
- `EqGather`: eq table + indexed gather, ~15 lines
- `EqPushforward`: eq table + scatter-add, ~15 lines
- `EqProject`: eq table + matrix-vector multiply (two code paths), ~40 lines
- `BytecodeVal`: multi-step gamma/RAF/register computation, ~50 lines

(`EqTable`, `LtTable`, `EqPlusOneTable`, `Provided` already go through the backend or are trivial.)

## Fix
Add per-variant backend methods:

```rust
fn scale_buffer<F: Field>(&self, source: &Self::Buffer<F>, scale: F) -> Self::Buffer<F>;
fn transpose<F: Field>(&self, source: &Self::Buffer<F>, rows: usize, cols: usize) -> Self::Buffer<F>;
fn eq_gather<F: Field>(&self, eq_point: &[F], indices: &Self::Buffer<F>, output_len: usize) -> Self::Buffer<F>;
fn eq_pushforward<F: Field>(&self, eq_point: &[F], indices: &Self::Buffer<F>, output_size: usize) -> Self::Buffer<F>;
fn eq_project<F: Field>(&self, source: &Self::Buffer<F>, eq_point: &[F], inner_size: usize, outer_size: usize) -> Self::Buffer<F>;
fn bytecode_val<F: Field>(&self, /* bytecode params */) -> Self::Buffer<F>;
```

The runtime calls `backend.eq_project(...)` instead of building a Vec<F> and uploading. The buffer stays on-device for GPU backends.

## Incremental Approach
Migrate one variant at a time. Keep `resolve_inputs` as a fallback for unconverted variants. Order by simplicity:
1. ScaleByChallenge
2. Transpose
3. EqGather
4. EqPushforward
5. EqProject
6. BytecodeVal (most complex, depends on `bytecode_raf` module)

## Risk: Medium
Many variants but each is independent. BytecodeVal is the hardest due to external module dependency.

## Dependencies: None (subsumes Task 12)
