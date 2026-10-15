// Accumulator and reduction kernels, generic over the field type.
//
// The conformance kernels give each thread `terms` consecutive terms, sum
// them with threadgroup_merge, and write the threadgroup's sum from every
// thread, so the test also checks that every lane receives the total. Grids
// are whole threadgroups (reduce.h); threads past the inputs contribute the
// empty sum.

// Term t is a[t] * b[t], or a[t] alone when t = 3 (mod 4).
template <typename F>
kernel void jolt_test_accum_fmadd(device const F* a [[buffer(0)]],
                                  device const F* b [[buffer(1)]],
                                  constant uint& n [[buffer(2)]],
                                  constant uint& terms [[buffer(3)]],
                                  device F* out [[buffer(4)]],
                                  uint i [[thread_position_in_grid]],
                                  ushort lane [[thread_index_in_simdgroup]],
                                  ushort simdgroup [[simdgroup_index_in_threadgroup]],
                                  ushort simdgroups [[simdgroups_per_threadgroup]],
                                  ushort width [[threads_per_simdgroup]]) {
    using Acc = jolt::Accumulator<F>;
    threadgroup Acc scratch[32];
    Acc acc = Acc::zero();
    for (uint k = 0; k < terms; k++) {
        uint t = i * terms + k;
        if (t < n) {
            if (t % 4 == 3) {
                acc.add(a[t]);
            } else {
                acc.fmadd(a[t], b[t]);
            }
        }
    }
    Acc total = jolt::threadgroup_merge(acc, scratch, lane, simdgroup, simdgroups, width);
    out[i] = total.reduce();
}

// Term t is, by t mod 4: fmadd_i64(a[t], s[t]), fmadd_u64(a[t], s[t]),
// fmadd_signed_u64(a[t], s[t], t mod 8 < 4), or add(a[t]).
template <typename F>
kernel void jolt_test_accum_small_scalar(device const F* a [[buffer(0)]],
                                         device const ulong* s [[buffer(1)]],
                                         constant uint& n [[buffer(2)]],
                                         constant uint& terms [[buffer(3)]],
                                         device F* out [[buffer(4)]],
                                         uint i [[thread_position_in_grid]],
                                         ushort lane [[thread_index_in_simdgroup]],
                                         ushort simdgroup [[simdgroup_index_in_threadgroup]],
                                         ushort simdgroups [[simdgroups_per_threadgroup]],
                                         ushort width [[threads_per_simdgroup]]) {
    using Acc = jolt::SmallScalarAccumulator<F>;
    threadgroup Acc scratch[32];
    Acc acc = Acc::zero();
    for (uint k = 0; k < terms; k++) {
        uint t = i * terms + k;
        if (t < n) {
            switch (t % 4) {
            case 0:
                acc.fmadd_i64(a[t], long(s[t]));
                break;
            case 1:
                acc.fmadd_u64(a[t], s[t]);
                break;
            case 2:
                acc.fmadd_signed_u64(a[t], s[t], t % 8 < 4);
                break;
            default:
                acc.add(a[t]);
                break;
            }
        }
    }
    Acc total = jolt::threadgroup_merge(acc, scratch, lane, simdgroup, simdgroups, width);
    out[i] = total.reduce();
}

// Capacity tests. Fill kernels write one unreduced accumulator per thread,
// each holding `terms` worst-case terms. The merge kernel runs as a single
// threadgroup: it merges every accumulator into one and reduces it, after
// one more term when `extra` is set. With `pre_reduce`, that term goes into
// Acc::zero().add(partial), the documented path past CAPACITY; without it,
// it goes into the full accumulator.

// `terms` products x * x.
template <typename F>
kernel void jolt_test_accum_fill(device const F* x [[buffer(0)]],
                                 constant uint& terms [[buffer(1)]],
                                 device jolt::Accumulator<F>* out [[buffer(2)]],
                                 uint i [[thread_position_in_grid]]) {
    using Acc = jolt::Accumulator<F>;
    F v = x[0];
    Acc acc = Acc::zero();
    for (uint k = 0; k < terms; k++) {
        acc.fmadd(v, v);
    }
    out[i] = acc;
}

// `terms` products x * (2^64 - 1), all positive or all negative.
template <typename F>
kernel void jolt_test_accum_small_scalar_fill(device const F* x [[buffer(0)]],
                                              constant uint& terms [[buffer(1)]],
                                              constant bool& negative [[buffer(2)]],
                                              device jolt::SmallScalarAccumulator<F>* out
                                              [[buffer(3)]],
                                              uint i [[thread_position_in_grid]]) {
    using Acc = jolt::SmallScalarAccumulator<F>;
    F v = x[0];
    Acc acc = Acc::zero();
    for (uint k = 0; k < terms; k++) {
        if (negative) {
            acc.fmadd_signed_u64(v, ~0ul, false);
        } else {
            acc.fmadd_u64(v, ~0ul);
        }
    }
    out[i] = acc;
}

// The sum of `count` accumulators, merged by one threadgroup.
template <typename Acc>
Acc merge_partials(device const Acc* partials,
                   uint count,
                   threadgroup Acc* scratch,
                   uint i,
                   uint threads,
                   ushort lane,
                   ushort simdgroup,
                   ushort simdgroups,
                   ushort width) {
    Acc acc = Acc::zero();
    for (uint j = i; j < count; j += threads) {
        acc.merge(partials[j]);
    }
    return jolt::threadgroup_merge(acc, scratch, lane, simdgroup, simdgroups, width);
}

template <typename F>
kernel void jolt_test_accum_merge(device const jolt::Accumulator<F>* partials [[buffer(0)]],
                                  constant uint& count [[buffer(1)]],
                                  device const F* x [[buffer(2)]],
                                  constant bool& extra [[buffer(3)]],
                                  constant bool& pre_reduce [[buffer(4)]],
                                  device F* out [[buffer(5)]],
                                  uint i [[thread_position_in_grid]],
                                  uint threads [[threads_per_grid]],
                                  ushort lane [[thread_index_in_simdgroup]],
                                  ushort simdgroup [[simdgroup_index_in_threadgroup]],
                                  ushort simdgroups [[simdgroups_per_threadgroup]],
                                  ushort width [[threads_per_simdgroup]]) {
    using Acc = jolt::Accumulator<F>;
    threadgroup Acc scratch[32];
    Acc total =
        merge_partials(partials, count, scratch, i, threads, lane, simdgroup, simdgroups, width);
    if (extra) {
        if (pre_reduce) {
            F partial = total.reduce();
            total = Acc::zero();
            total.add(partial);
        }
        total.fmadd(x[0], x[0]);
    }
    if (i == 0) {
        out[0] = total.reduce();
    }
}

template <typename F>
kernel void jolt_test_accum_small_scalar_merge(
    device const jolt::SmallScalarAccumulator<F>* partials [[buffer(0)]],
    constant uint& count [[buffer(1)]],
    device const F* x [[buffer(2)]],
    constant bool& extra [[buffer(3)]],
    constant bool& pre_reduce [[buffer(4)]],
    constant bool& negative [[buffer(5)]],
    device F* out [[buffer(6)]],
    uint i [[thread_position_in_grid]],
    uint threads [[threads_per_grid]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort simdgroup [[simdgroup_index_in_threadgroup]],
    ushort simdgroups [[simdgroups_per_threadgroup]],
    ushort width [[threads_per_simdgroup]]) {
    using Acc = jolt::SmallScalarAccumulator<F>;
    threadgroup Acc scratch[32];
    Acc total =
        merge_partials(partials, count, scratch, i, threads, lane, simdgroup, simdgroups, width);
    if (extra) {
        if (pre_reduce) {
            F partial = total.reduce();
            total = Acc::zero();
            total.add(partial);
        }
        total.fmadd_signed_u64(x[0], ~0ul, !negative);
    }
    if (i == 0) {
        out[0] = total.reduce();
    }
}
