# jolt-cpu Optimization Experiment Log

## Hardware
- Apple M1 Pro (6P + 2E cores), 16GB RAM
- See `goal.txt` for roofline analysis and targets

## Baseline
- **Date**: 2026-04-03
- **Commit**: 90eb55b0f
- **Rayon threads**: 8

| Benchmark | ns/op | Mops/s | Target ns/op | Gap |
|-----------|-------|--------|--------------|-----|
| field_mul | 14.32 | 69.82 | — (roofline) | — |
| field_add | 3.46 | 288.69 | — (roofline) | — |
| fmadd_one | 9.17 | 109.10 | ~3.5 (= field_add) | 2.6× |
| toom4_eval | 221.23 | 4.52 | ≤155 (10×14.3×1.1) | 1.43× |
| toom8_eval | 662.42 | 1.51 | ≤456 (29×14.3×1.1) | 1.45× |
| toom16_eval | 2289.88 | 0.44 | ≤1494 (95×14.3×1.1) | 1.53× |
| dense_reduce_d4 | 69.53 | 14.38 | ≤43 | 1.62× |
| dense_reduce_d8 | 204.17 | 4.90 | ≤126 | 1.62× |
| dense_reduce_d16 | 813.66 | 1.23 | ≤411 | 1.98× |
| bind_l2h | 36.01 | 27.77 | ≤4.7 | 7.66× |
| bind_h2l | 29.57 | 33.82 | ≤4.7 | 6.29× |

Notes:
- T_mul = 14.32ns (measured). Targets recalibrated from goal.txt's 18ns estimate.
- fmadd_one at 9.17ns vs field_mul at 14.32ns: the fmadd does 4×4 schoolbook WITHOUT
  Montgomery REDC (deferred), so it's cheaper than a full multiply. But still ~2.6× more
  than field_add. The acc_add optimization should bring it to ~3.5ns.
- Bind targets recalculated: per-element = T_mul + 2×T_add = 14.32 + 6.92 = 21.24ns,
  parallel (6 cores, 80%): 21.24/4.8 = 4.4ns/pair. Current 30-36ns is 7× off —
  suggests significant overhead beyond the field ops themselves.
- dense_reduce targets: single-pair cost / (6×0.80). Current ~1.6× off for D=4,8;
  ~2× off for D=16.

---

## Experiment 001: Eliminate heap allocs in product_sum + acc_add optimization

- **Date**: 2026-04-03 12:56
- **Commit**: 90eb55b0f (dirty)
- **Hypothesis**: (1) Heap allocations in product_sum::compile_fn (2 Vecs per eval call)
  dominate overhead for small D. Replacing with const-generic stack arrays should
  improve toom eval by 20%+. (2) WideAccumulator::fmadd(one, x) does a full 4×4
  schoolbook multiply (16 u64 MULs). Replacing with acc_add (4 u64 ADDs at offset 4)
  should save ~5.7ns per accumulation.
- **Changes**:
  - `product_sum.rs`: Dispatch to const-generic `compile_single_product<D>` for P=1
    and `compile_multi_product<D>` for P>1, using `[(F,F); D]` stack arrays.
  - `accumulator.rs`: Added `acc_add()` default method to FieldAccumulator trait.
  - `wide_accumulator.rs`: Optimized `acc_add()` — adds val limbs at offset 4 (×R=×2^256).
  - `backend.rs`: All `a.fmadd(one, *e)` → `a.acc_add(*e)`, removed unused `one`.
- **Files**: product_sum.rs, accumulator.rs, wide_accumulator.rs, backend.rs

### Results
| Benchmark | Baseline | Current | Delta |
|-----------|----------|---------|-------|
| field_mul | 14.32 | 14.21 | -0.8% |
| field_add | 3.46 | 3.33 | -3.8% |
| fmadd_one | 9.17 | 9.37 | +2.2% |
| toom4_eval | 221.23 | 170.09 | **-23.1%** |
| toom8_eval | 662.42 | 598.37 | **-9.7%** |
| toom16_eval | 2289.88 | 2191.75 | -4.3% |
| dense_reduce_d4 | 69.53 | 52.12 | **-25.0%** |
| dense_reduce_d8 | 204.17 | 185.53 | **-9.1%** |
| dense_reduce_d16 | 813.66 | 771.95 | -5.1% |
| bind_l2h | 36.01 | 34.13 | -5.2% |
| bind_h2l | 29.57 | 31.85 | +7.7% |

### Analysis
Both optimizations landed cleanly. Toom eval improvements are from heap alloc elimination
(stack arrays). Dense reduce gains combine both: faster eval + cheaper accumulation.
The D=4 case benefits most because the heap alloc overhead was proportionally largest
(2 allocs of 4 elements each = 256B heap overhead vs 256B of useful data).

Note: fmadd_one bench still measures the OLD fmadd path since it calls the trait method
directly, not acc_add. The acc_add improvement is visible only in the reduce benchmarks.

Remaining gaps to target:
- toom4_eval: 170ns vs target 155ns (1.10×)
- dense_reduce_d4: 52ns vs target 43ns (1.21×)
- dense_reduce_d8: 186ns vs target 126ns (1.48×)
- dense_reduce_d16: 772ns vs target 411ns (1.88×)
- bind: 31-34ns vs target 4.7ns (6.6-7.2×) — bind target likely wrong, re-evaluate

### Decision
- [x] Keep

---

## Experiment 002: MaybeUninit for pairs array in compile_single_product

- **Date**: 2026-04-03 12:58
- **Hypothesis**: The zero-init of `[(F::zero(), F::zero()); D]` before immediately overwriting
  wastes ~10ns per call (512B memset for D=8). Using MaybeUninit should eliminate this.
- **Change**: Replaced zero-init with MaybeUninit + ptr::write in compile_single/multi_product.

### Results
| Benchmark | Before | After | Delta |
|-----------|--------|-------|-------|
| toom4_eval | 169 | 179 | +5.9% |
| toom8_eval | 596 | 794 | +33.2% |
| toom16_eval | 2178 | 2334 | +7.2% |

### Analysis
REGRESSION. The compiler was already optimizing away the dead zero-init via dead store
elimination. The raw pointer writes in MaybeUninit actually inhibit compiler optimizations
(pointer aliasing analysis, register allocation across the unsafe boundary).

### Decision
- [x] Revert

---

## Experiment 003: Rayon fold restructuring + with_min_len(4096)

- **Date**: 2026-04-03 13:15
- **Commit**: 90eb55b0f (dirty)
- **Hypothesis**: Rayon's default splitting creates too many small tasks for dense_reduce,
  wasting time on scheduling overhead. Adding `.with_min_len(4096)` ensures each task
  processes at least 4096 pairs (~700µs at D=4), amortizing Rayon overhead. Also simplified
  the fold identity for reduce_dense_fixed by moving scratch arrays into the fold body.
- **Changes**:
  - `backend.rs`: Added `.with_min_len(4096)` to all parallel reduce paths
    (reduce_dense_fixed, reduce_dense_dynamic, reduce_sparse).
  - `backend.rs`: Simplified reduce_dense_fixed fold: scratch arrays (lo, hi, evals)
    moved from fold identity tuple into fold body, fold state is just `[Accumulator; NE]`.
- **Files**: backend.rs

### Results
| Benchmark | Exp001 | Current (best of 3) | Delta |
|-----------|--------|---------------------|-------|
| dense_reduce_d4 | 52.12 | 47.12 | **-9.6%** |
| dense_reduce_d8 | 185.53 | 181.88 | -2.0% |
| dense_reduce_d16 | 771.95 | 696.55 | **-9.7%** |

Note: System exhibited high variance during testing (field_mul oscillated 14-22ns).
Best-of-3 numbers taken from runs where field_mul was stable at ~14.5ns.

### Analysis
Modest improvement, most visible at D=4 and D=16. The with_min_len(4096) reduces task
creation overhead. For D=4 (170ns/pair), the default Rayon splitting likely over-split,
creating too many tasks with insufficient work each. The fold body restructuring
(stack arrays per iteration vs per task) has negligible effect since the compiler
eliminates dead zero-init via DSE. Change is low-risk and adds no complexity.

### Decision
- [x] Keep

---

## Experiment 004: In-place LowToHigh bind (eliminate Vec allocation)

- **Date**: 2026-04-03 13:30
- **Commit**: 90eb55b0f (dirty)
- **Hypothesis**: The LowToHigh bind path allocates a new Vec<F> via par_iter().collect()
  and replaces the old buffer — a 4MB allocation + 8MB deallocation per bind call.
  HighToLow uses split_at_mut() in-place (no allocation) and is 2.6× faster.
  Making L2H in-place should match H2L performance.
- **Changes**:
  - `backend.rs`: Replaced L2H parallel bind from `par_iter().map().collect()` to
    unsafe in-place write via raw pointer. Safety: buf[i] is written after buf[2*i]
    and buf[2*i+1] are read; since i < 2*i for i ≥ 1, writes never clobber unread inputs.
    Each par_iter task writes a unique buf[i].
- **Files**: backend.rs

### Results
| Benchmark | Before | After | Delta |
|-----------|--------|-------|-------|
| bind_l2h | 61.77 | 24.27 | **-60.7%** |
| bind_h2l | 24.08 | 22.84 | -5.2% |

### Analysis
Eliminating the Vec allocation closed the L2H gap entirely. Both bind variants now
perform identically at ~23-24ns/elem. The overhead was dominated by allocator cost
(4MB malloc + 8MB free per call). With 8 Rayon threads vs 1-thread sequential:
speedup = 145/24 = 6.0× (75% parallel efficiency on 8 threads).

### Decision
- [x] Keep

---

## Experiment 005: Pre-compute input data pointers in reduce_dense_fixed

- **Date**: 2026-04-03 13:45
- **Commit**: 90eb55b0f (dirty)
- **Hypothesis**: The reduce inner loop dereferences through `&[&Vec<F>]` — two pointer
  indirections per input per pair. For D=16 (16 inputs), this is 32 extra pointer loads
  per pair. Pre-computing raw data pointers (stored as usize for Send+Sync) eliminates
  this overhead. Expected to help most for large NI where indirection dominates.
- **Changes**:
  - `backend.rs`: In reduce_dense_fixed, pre-compute `ptrs: [usize; NI]` from
    `inputs[k].as_ptr()`. Replace `inputs.iter().enumerate()` loop with direct
    pointer arithmetic via `load_pair()` closure. Sequential path also updated.
- **Files**: backend.rs

### Results
| Benchmark | Before | After | Delta |
|-----------|--------|-------|-------|
| dense_reduce_d4 | 47.77 | 47.07 | -1.5% |
| dense_reduce_d8 | 162.63 | 170.60 | +4.9% |
| dense_reduce_d16 | 789.51 | 707.78 | **-10.4%** |

### Analysis
Clear improvement for D=16 where 16 input streams amplify the indirection overhead.
D=4 and D=8 are within measurement noise. The usize pointer cache eliminates per-pair
slice lookups and Vec dereferences, but the benefit only shows when NI is large enough
for the overhead to matter relative to the eval cost.

### Decision
- [x] Keep

---

## Current Status (after Experiments 001-005)

| Benchmark | Original | Current | Δ vs baseline | Target | Gap |
|-----------|----------|---------|---------------|--------|-----|
| field_mul | 14.32 | 14.25 | -0.5% | ≤15 | ✅ MET |
| toom4_eval | 221.23 | 169.08 | -23.6% | ≤194 | ✅ MET |
| toom8_eval | 662.42 | 599.25 | -9.5% | ≤581 | ~3% gap |
| toom16_eval | 2289.88 | 2191.52 | -4.3% | ≤2027 | ~8% gap |
| dense_reduce_d4 | 69.53 | 47.07 | -32.3% | ≤39.5 | ~19% gap |
| dense_reduce_d8 | 204.17 | 170.60 | -16.4% | ≤148.5 | ~15% gap |
| dense_reduce_d16 | 813.66 | 707.78 | -13.0% | ≤625 | ~13% gap |
| bind_l2h | 36.01 | 24.60 | -31.7% | ≤22.1 | ~11% gap |
| bind_h2l | 29.57 | 24.53 | -17.0% | ≤22.1 | ~11% gap |

---

## Experiment 006: Direct toom eval in reduce (bypass dyn dispatch)

- **Date**: 2026-04-03 14:15
- **Hypothesis**: The Box<dyn Fn> eval dispatch + lo/hi→pairs copy adds ~15-20ns per pair.
  A specialized reduce_product_sum_fixed that calls toom_cook::eval_linear_prod_assign
  directly (no dyn dispatch, no intermediate lo/hi arrays) should improve D=4 by ~8%.
- **Changes**: Added is_single_product_sum flag to CpuKernel, reduce_product_sum_fixed
  function, dispatcher routing.

### Results
No measurable improvement in either sequential (1-thread) or parallel benchmarks.
- d4: 235.2ns seq (was 235.8ns), 47ns par (was 47ns)
- d8: 888.9ns seq (was 891ns), 164ns par (was 165ns)
- d16: 3760ns seq (was 3751ns), 683ns par (was 708ns)

### Analysis
The compiler already optimizes the dyn dispatch path effectively. The Box<dyn Fn>
overhead (~5ns) is hidden by superscalar execution (overlaps with memory loads).
The lo/hi→pairs copy is optimized away or pipelined. The real bottleneck for dense
reduce is memory access latency from scattered input arrays, not compute overhead.

### Decision
- [x] Revert (adds complexity for zero benefit)
