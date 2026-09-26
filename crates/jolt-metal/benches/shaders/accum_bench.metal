// Deferred multiply-accumulate benchmarks, generic over the field type.
//
// Every kernel writes a value that depends on all of its work, and the
// benchmark checks it against the CPU, so the compiler cannot drop work.

// Terms per accumulator in the fmadd kernels: 4 per round.
constant constexpr uint FMADD_ROUNDS = 64;

// Four fmadds per round into one accumulator, then y <- y + x0, so no product
// is loop-invariant. The field add is a quarter of an add per fmadd.
template <typename F>
kernel void jolt_bench_accum_fmadd(device const F* a [[buffer(0)]],
                                   device const F* b [[buffer(1)]],
                                   device F* out [[buffer(2)]],
                                   uint i [[thread_position_in_grid]]) {
    using Acc = jolt::Accumulator<F>;
    static_assert(4 * FMADD_ROUNDS <= Acc::CAPACITY, "reduce before CAPACITY terms");
    F y = b[i];
    F x0 = a[i];
    F x1 = x0 + y;
    F x2 = x1 + y;
    F x3 = x2 + y;
    Acc acc = Acc::zero();
    for (uint r = 0; r < FMADD_ROUNDS; r++) {
        acc.fmadd(x0, y);
        acc.fmadd(x1, y);
        acc.fmadd(x2, y);
        acc.fmadd(x3, y);
        y = y + x0;
    }
    out[i] = acc.reduce();
}

// The same products into four accumulators, one per x: four times the live
// accumulator state, as in a kernel that accumulates several outputs.
template <typename F>
kernel void jolt_bench_accum_fmadd4(device const F* a [[buffer(0)]],
                                    device const F* b [[buffer(1)]],
                                    device F* out [[buffer(2)]],
                                    uint i [[thread_position_in_grid]]) {
    using Acc = jolt::Accumulator<F>;
    static_assert(FMADD_ROUNDS <= Acc::CAPACITY, "reduce before CAPACITY terms");
    F y = b[i];
    F x0 = a[i];
    F x1 = x0 + y;
    F x2 = x1 + y;
    F x3 = x2 + y;
    Acc acc0 = Acc::zero();
    Acc acc1 = Acc::zero();
    Acc acc2 = Acc::zero();
    Acc acc3 = Acc::zero();
    for (uint r = 0; r < FMADD_ROUNDS; r++) {
        acc0.fmadd(x0, y);
        acc1.fmadd(x1, y);
        acc2.fmadd(x2, y);
        acc3.fmadd(x3, y);
        y = y + x0;
    }
    out[i] = (acc0.reduce() + acc1.reduce()) + (acc2.reduce() + acc3.reduce());
}

// Four fmadd_i64 per round into one small-scalar accumulator. Scalar k of
// round r is s[i] + (4 r + k) * 0x9E3779B97F4A7C15 modulo 2^64, read as
// signed, so signs are mixed and no product is loop-invariant.
template <typename F>
kernel void jolt_bench_accum_fmadd_i64(device const F* a [[buffer(0)]],
                                       device const ulong* s [[buffer(1)]],
                                       device F* out [[buffer(2)]],
                                       uint i [[thread_position_in_grid]]) {
    using Acc = jolt::SmallScalarAccumulator<F>;
    static_assert(4 * FMADD_ROUNDS <= Acc::CAPACITY, "reduce before CAPACITY terms");
    F x0 = a[i];
    F x1 = x0 + x0;
    F x2 = x1 + x0;
    F x3 = x2 + x0;
    ulong z = s[i];
    Acc acc = Acc::zero();
    for (uint r = 0; r < FMADD_ROUNDS; r++) {
        acc.fmadd_i64(x0, long(z));
        z += 0x9E3779B97F4A7C15ul;
        acc.fmadd_i64(x1, long(z));
        z += 0x9E3779B97F4A7C15ul;
        acc.fmadd_i64(x2, long(z));
        z += 0x9E3779B97F4A7C15ul;
        acc.fmadd_i64(x3, long(z));
        z += 0x9E3779B97F4A7C15ul;
    }
    out[i] = acc.reduce();
}

// jolt_bench_field_inner_product with the products accumulated unreduced and
// each threadgroup's accumulators merged by threadgroup_merge. Launched in
// whole threadgroups (reduce.h). A threadgroup accumulates distinct indices
// below n, fewer than 2^32 terms, so any n is within CAPACITY.
template <typename F>
kernel void jolt_bench_accum_inner_product(device const F* a [[buffer(0)]],
                                           device const F* b [[buffer(1)]],
                                           constant uint& n [[buffer(2)]],
                                           device F* out [[buffer(3)]],
                                           uint i [[thread_position_in_grid]],
                                           uint threads [[threads_per_grid]],
                                           uint group [[threadgroup_position_in_grid]],
                                           ushort lane [[thread_index_in_simdgroup]],
                                           ushort simdgroup [[simdgroup_index_in_threadgroup]],
                                           ushort simdgroups [[simdgroups_per_threadgroup]],
                                           ushort width [[threads_per_simdgroup]]) {
    using Acc = jolt::Accumulator<F>;
    static_assert(Acc::CAPACITY >= 0xFFFFFFFFul, "a threadgroup may read every index below n");
    threadgroup Acc scratch[32];
    Acc acc = Acc::zero();
    for (uint j = i; j < n; j += threads) {
        acc.fmadd(a[j], b[j]);
    }
    Acc total = jolt::threadgroup_merge(acc, scratch, lane, simdgroup, simdgroups, width);
    if (simdgroup == 0 && lane == 0) {
        out[group] = total.reduce();
    }
}
