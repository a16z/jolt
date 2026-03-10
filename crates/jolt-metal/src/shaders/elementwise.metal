#include "wide_accumulator.metal"

// buf[i] *= scalar
kernel void fr_scale_kernel(
    device Fr*       buf    [[buffer(0)]],
    device const Fr* scalar [[buffer(1)]],
    uint tid                [[thread_position_in_grid]]
) {
    buf[tid] = fr_mul(buf[tid], scalar[0]);
}

// out[i] = a[i] + b[i]
kernel void fr_add_buf_kernel(
    device const Fr* a      [[buffer(0)]],
    device const Fr* b      [[buffer(1)]],
    device Fr*       result [[buffer(2)]],
    uint tid                [[thread_position_in_grid]]
) {
    result[tid] = fr_add(a[tid], b[tid]);
}

// out[i] = a[i] - b[i]
kernel void fr_sub_buf_kernel(
    device const Fr* a      [[buffer(0)]],
    device const Fr* b      [[buffer(1)]],
    device Fr*       result [[buffer(2)]],
    uint tid                [[thread_position_in_grid]]
) {
    result[tid] = fr_sub(a[tid], b[tid]);
}

// buf[i] += scalar * other[i]
kernel void fr_accumulate_kernel(
    device Fr*       buf    [[buffer(0)]],
    device const Fr* scalar [[buffer(1)]],
    device const Fr* other  [[buffer(2)]],
    uint tid                [[thread_position_in_grid]]
) {
    buf[tid] = fr_add(buf[tid], fr_mul(scalar[0], other[tid]));
}

// Parallel reduction: each threadgroup reduces a chunk to a single partial sum.
// The host reads back partial sums and finishes on CPU.
constant uint SUM_GROUP_SIZE = 256;

kernel void fr_sum_kernel(
    device const Fr*  buf         [[buffer(0)]],
    device Fr*        partials    [[buffer(1)]],
    device const uint* params     [[buffer(2)]],
    uint tid                      [[thread_position_in_grid]],
    uint lid                      [[thread_position_in_threadgroup]],
    uint gid                      [[threadgroup_position_in_grid]]
) {
    uint n = params[0];

    // Per-thread accumulation: acc_add_fr avoids all multiplications.
    // Accumulated value is Σ a_i_mont as a raw wide integer.
    WideAcc local_acc = acc_zero();
    for (uint i = tid; i < n; i += SUM_GROUP_SIZE * params[1]) {
        acc_add_fr(local_acc, buf[i]);
    }

    // acc_reduce gives Σ a_i (standard form); fr_to_mont restores Montgomery.
    Fr local_fr = fr_to_mont(acc_reduce(local_acc));

    // Tree reduction with Fr values (8 KB shared vs 18 KB for WideAcc).
    threadgroup Fr shared[SUM_GROUP_SIZE];
    shared[lid] = local_fr;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = SUM_GROUP_SIZE / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            shared[lid] = fr_add(shared[lid], shared[lid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0) {
        partials[gid] = shared[0];
    }
}

// Dot product: each threadgroup reduces a chunk of a[i]*b[i] to a partial sum.
kernel void fr_dot_product_kernel(
    device const Fr*  a           [[buffer(0)]],
    device const Fr*  b           [[buffer(1)]],
    device Fr*        partials    [[buffer(2)]],
    device const uint* params     [[buffer(3)]],
    uint tid                      [[thread_position_in_grid]],
    uint lid                      [[thread_position_in_threadgroup]],
    uint gid                      [[threadgroup_position_in_grid]]
) {
    uint n = params[0];

    // Per-thread: acc_fmadd defers Montgomery reduction to the end.
    WideAcc local_acc = acc_zero();
    for (uint i = tid; i < n; i += SUM_GROUP_SIZE * params[1]) {
        acc_fmadd(local_acc, a[i], b[i]);
    }

    Fr local_fr = acc_reduce(local_acc);

    // Tree reduction with Fr values.
    threadgroup Fr shared[SUM_GROUP_SIZE];
    shared[lid] = local_fr;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = SUM_GROUP_SIZE / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            shared[lid] = fr_add(shared[lid], shared[lid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0) {
        partials[gid] = shared[0];
    }
}
