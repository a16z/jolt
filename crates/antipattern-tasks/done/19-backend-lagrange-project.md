# Task 19: Route LagrangeProject Through Backend

## Status: TODO

## Anti-Pattern
`Op::LagrangeProject` (runtime.rs ~60 lines) does a download-compute-upload round trip:
1. Computes Lagrange basis on CPU: `lagrange_evals(domain_start, domain_size, r)` — O(k^2)
2. Downloads the device buffer to host
3. Triple nested loop: `for c in cycles { for g in groups { for k in domain { projected[c*groups+g] += basis[k] * data[...] } } }` — O(cycles × groups × domain_size)
4. Optionally scales by kernel tau factor
5. Uploads result back to device

This is the Spartan constraint regrouping projection. The download/upload round trip is exactly what GPU backends should eliminate.

## Fix
Add to `ComputeBackend`:

```rust
fn lagrange_project<F: Field>(
    &self,
    buf: &Self::Buffer<F>,
    challenge: F,
    domain_start: i64,
    domain_size: usize,
    stride: usize,
    num_groups: usize,
    kernel_tau: Option<(F, usize, i64)>,  // (tau, domain_size, domain_start)
) -> Self::Buffer<F>;
```

CPU backend: lift existing code. GPU backend: single kernel dispatch, no round trip.

## Risk: Medium
The stride/group layout and optional kernel tau scaling have non-trivial semantics. Need careful parity testing.

## Dependencies: None
