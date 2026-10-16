// Simdgroup and threadgroup sums of accumulators, generic over any type that
// meets the accumulator contract in accum.h.
//
// Field arithmetic is exact, so the order of merges does not change any
// result. Capacity still applies: the merged sum counts every term of every
// thread.
//
// Both functions need every lane of each simdgroup to be active, since a
// shuffle from an inactive lane is undefined. Kernels that use them dispatch
// whole threadgroups whose size is a multiple of the execution width, and
// threads with no work contribute Acc::zero().

#ifndef JOLT_FIELD_REDUCE_H
#define JOLT_FIELD_REDUCE_H

#include <metal_stdlib>

namespace jolt {

// The sum over the simdgroup, returned to every lane: a butterfly of
// log2(width) exchanges, so no lane waits for a broadcast. `width` is
// [[threads_per_simdgroup]], a power of two.
template <typename Acc>
Acc simd_merge(Acc acc, ushort width) {
    for (ushort mask = 1; mask < width; mask <<= 1) {
        acc.merge(simd_shuffle_xor(acc, mask));
    }
    return acc;
}

// The sum over the threadgroup, returned to every thread. Each simdgroup sums
// its lanes, lane 0 of each writes its sum to `scratch`, and after a barrier
// every simdgroup sums those partial sums.
//
// `scratch` holds at least `simdgroups` accumulators, and `simdgroups` is at
// most `width`: 32 simdgroups of 32 lanes cover the largest threadgroup,
// 1024 threads. A kernel that calls this again with the same scratch must
// place a threadgroup barrier between the calls.
template <typename Acc>
Acc threadgroup_merge(Acc acc,
                      threadgroup Acc* scratch,
                      ushort lane,
                      ushort simdgroup,
                      ushort simdgroups,
                      ushort width) {
    acc = simd_merge(acc, width);
    if (lane == 0) {
        scratch[simdgroup] = acc;
    }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    Acc partial = lane < simdgroups ? scratch[lane] : Acc::zero();
    return simd_merge(partial, width);
}

} // namespace jolt

#endif // JOLT_FIELD_REDUCE_H
