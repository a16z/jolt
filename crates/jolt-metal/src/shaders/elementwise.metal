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
    threadgroup WideAcc shared_acc[SUM_GROUP_SIZE];

    uint n = params[0];
    WideAcc local_acc = acc_zero();

    // Each thread accumulates its stripe: fmadd(buf[i], 1)
    // Since we want sum, we use the identity: sum = Σ buf[i] * R_mont * R_mont_inv
    // Simpler: just add elements directly using field addition, and use wide
    // accumulator with fmadd(element, ONE_MONT).
    Fr one = fr_one();
    for (uint i = tid; i < n; i += SUM_GROUP_SIZE * params[1]) {
        acc_fmadd(local_acc, buf[i], one);
    }
    shared_acc[lid] = local_acc;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction within threadgroup
    for (uint stride = SUM_GROUP_SIZE / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            acc_merge(shared_acc[lid], shared_acc[lid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0) {
        partials[gid] = acc_reduce_tg(shared_acc[0]);
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
    threadgroup WideAcc shared_acc[SUM_GROUP_SIZE];

    uint n = params[0];
    WideAcc local_acc = acc_zero();

    for (uint i = tid; i < n; i += SUM_GROUP_SIZE * params[1]) {
        acc_fmadd(local_acc, a[i], b[i]);
    }
    shared_acc[lid] = local_acc;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = SUM_GROUP_SIZE / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            acc_merge(shared_acc[lid], shared_acc[lid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0) {
        partials[gid] = acc_reduce_tg(shared_acc[0]);
    }
}
