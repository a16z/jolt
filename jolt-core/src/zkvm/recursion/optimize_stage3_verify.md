# Stage 3 Verification Performance Optimization

## Problem Analysis

The `verify_recursion_stage3` span is taking 5,758 ms when it should be in the milliseconds range. The bottleneck is in the `expected_output_claim` function of `JaggedSumcheckVerifier`.

### Root Causes

1. **Repeated Point Object Creation**: Inside the polynomial loop (which can iterate thousands of times), we create new `Point` objects for every polynomial:
   ```rust
   let zc = Point::from_usize(t_prev, num_bits);
   let zd = Point::from_usize(t_curr, num_bits);
   ```
   Each `Point::from_usize` allocates a new vector and computes binary representation.

2. **Redundant Branching Program Evaluations**: Many polynomials share the same `t_prev` and `t_curr` values (cumulative sizes), but we recompute `branching_program.eval_multilinear` for each one.

3. **Expensive eval_multilinear**: Each call to `eval_multilinear`:
   - Loops over `num_bits` (potentially 20+ iterations)
   - For each bit, computes equality values for 16 combinations
   - Performs state transitions for all 16 bit combinations
   - Total complexity: O(num_bits * 16 * 4) operations per polynomial

### Scale of the Problem

- Number of polynomials = Number of constraints * 15 (for each PolyType)
- For fibonacci_18, this could be thousands of polynomials
- Total operations: thousands * O(num_bits * 64)

## Optimization Strategy

### 1. Cache Point Objects (Immediate Fix)

Instead of creating Point objects inside the loop, pre-create and cache them:

```rust
// Before the loop
let mut point_cache: HashMap<usize, Point<F>> = HashMap::new();

// Inside the loop
let zc = point_cache.entry(t_prev).or_insert_with(|| {
    Point::from_usize(t_prev, num_bits)
});
let zd = point_cache.entry(t_curr).or_insert_with(|| {
    Point::from_usize(t_curr, num_bits)
});
```

**Expected savings**: ~20% of the 5.7s (avoiding thousands of allocations)

### 2. Cache Branching Program Evaluations (Major Fix)

Since many polynomials share the same `(t_prev, t_curr)` pairs:

```rust
// Before the loop
let mut bp_eval_cache: HashMap<(usize, usize), F> = HashMap::new();

// Inside the loop
let g_mle = bp_eval_cache.entry((t_prev, t_curr)).or_insert_with(|| {
    let zc = Point::from_usize(t_prev, num_bits);
    let zd = Point::from_usize(t_curr, num_bits);
    branching_program.eval_multilinear(&za, &zb, &zc, &zd)
});
```

**Expected savings**: ~70% of the 5.7s (most polynomials will hit the cache)

### 3. Pre-compute Common t_prev/t_curr Pairs (Optional)

Analyze the bijection to identify all unique `(t_prev, t_curr)` pairs upfront:

```rust
// In JaggedSumcheckVerifier::new
let unique_size_pairs: HashSet<(usize, usize)> = (0..bijection.num_polynomials())
    .map(|poly_idx| {
        let t_prev = bijection.cumulative_size_before(poly_idx);
        let t_curr = bijection.cumulative_size(poly_idx);
        (t_prev, t_curr)
    })
    .collect();

// Pre-compute all branching program evaluations
let bp_evals: HashMap<(usize, usize), F> = unique_size_pairs.into_iter()
    .map(|(t_prev, t_curr)| {
        let zc = Point::from_usize(t_prev, num_bits);
        let zd = Point::from_usize(t_curr, num_bits);
        let eval = branching_program.eval_multilinear(&za, &zb, &zc, &zd);
        ((t_prev, t_curr), eval)
    })
    .collect();
```

### 4. Alternative: Batch Evaluation

Instead of computing g_mle for each polynomial separately, batch the computation:

1. Identify all unique `(t_prev, t_curr)` pairs
2. Compute all necessary Point objects once
3. Batch evaluate the branching program
4. Look up results in the main loop

## Implementation Plan

1. **Quick Win**: Implement Point caching (#1) - Expected to reduce time by ~1 second
2. **Major Fix**: Implement branching program evaluation caching (#2) - Expected to reduce time by ~4 seconds
3. **Measure**: Profile to confirm improvements
4. **Optional**: If still slow, implement pre-computation (#3)

## Expected Result

Current: 5,758 ms
Target: < 1,000 ms (5x+ improvement)

The key insight is that the jagged bijection creates patterns where many polynomials share cumulative size boundaries, making caching extremely effective.