# Task 20: Fused Segmented Reduce in Backend

## Status: TODO

## Anti-Pattern
`segmented_reduce()` (runtime.rs ~70 lines) downloads ALL input data from device, then re-uploads column slices in a loop over outer positions:

```rust
let input_data: Vec<Vec<F>> = kdef.inputs.iter()
    .map(|b| backend.download(buf.as_field()))  // download everything
    .collect();

for (a, &weight) in outer_eq.iter().enumerate() {
    // extract column, upload, run kernel, accumulate
    col_buf.copy_from_slice(&data[start..start + inner_size]);
    col_bufs.push(DeviceBuffer::Field(backend.upload(&col_buf)));
    let evals = backend.reduce(compiled_kernel, &col_refs, challenges);
    // accumulate with weight
}
```

This is O(outer_size) round trips between host and device. For GPU backends, the entire segmented reduce should be a single fused kernel dispatch.

## Fix
Add to `ComputeBackend`:

```rust
fn segmented_reduce<F: Field>(
    &self,
    kernel: &Self::CompiledKernel<F>,
    inputs: &[&Buf<Self, F>],
    outer_eq: &Self::Buffer<F>,
    inner_only: &[bool],
    inner_size: usize,
    challenges: &[F],
) -> Vec<F>;
```

CPU backend: lift existing loop (download + column extraction + reduce). GPU backend: single fused kernel over all outer positions.

Also requires `build_outer_eq` to keep the eq table on-device (currently downloads it to `Vec<F>` in `segmented_outer_eqs`). Change `segmented_outer_eqs` storage from `Vec<F>` to `Buf<B, F>`.

## Risk: Medium-High
Complex interaction between inner-only vs mixed inputs, column extraction logic, and per-position accumulation. The fused kernel specification for GPU is non-trivial.

## Dependencies: Task 17 (input materializations keep buffers on-device)
