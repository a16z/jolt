// Test and benchmark kernels for BN254 Fr field arithmetic.
//
// Each kernel operates element-wise on input arrays. Thread index
// maps 1:1 to array position. All inputs and outputs are in
// Montgomery form (matching arkworks' internal representation).

#include "wide_accumulator.metal"

kernel void fr_mul_kernel(
    device const Fr* a       [[buffer(0)]],
    device const Fr* b       [[buffer(1)]],
    device Fr*       result  [[buffer(2)]],
    uint tid                 [[thread_position_in_grid]]
) {
    result[tid] = fr_mul(a[tid], b[tid]);
}

kernel void fr_add_kernel(
    device const Fr* a       [[buffer(0)]],
    device const Fr* b       [[buffer(1)]],
    device Fr*       result  [[buffer(2)]],
    uint tid                 [[thread_position_in_grid]]
) {
    result[tid] = fr_add(a[tid], b[tid]);
}

kernel void fr_sub_kernel(
    device const Fr* a       [[buffer(0)]],
    device const Fr* b       [[buffer(1)]],
    device Fr*       result  [[buffer(2)]],
    uint tid                 [[thread_position_in_grid]]
) {
    result[tid] = fr_sub(a[tid], b[tid]);
}

kernel void fr_sqr_kernel(
    device const Fr* a       [[buffer(0)]],
    device Fr*       result  [[buffer(1)]],
    uint tid                 [[thread_position_in_grid]]
) {
    result[tid] = fr_sqr(a[tid]);
}

kernel void fr_neg_kernel(
    device const Fr* a       [[buffer(0)]],
    device Fr*       result  [[buffer(1)]],
    uint tid                 [[thread_position_in_grid]]
) {
    result[tid] = fr_neg(a[tid]);
}

// Each thread accumulates N_FMADD products via the wide accumulator.
constant uint N_FMADD = 256;

kernel void fr_fmadd_kernel(
    device const Fr* a       [[buffer(0)]],
    device const Fr* b       [[buffer(1)]],
    device Fr*       result  [[buffer(2)]],
    device const uint* params [[buffer(3)]],
    uint tid                 [[thread_position_in_grid]]
) {
    uint stride = params[0]; // number of elements per input array
    WideAcc acc = acc_zero();
    uint base = tid * N_FMADD;
    for (uint i = 0; i < N_FMADD; i++) {
        uint idx = (base + i) % stride;
        acc_fmadd(acc, a[idx], b[idx]);
    }
    result[tid] = acc_reduce(acc);
}

kernel void fr_from_u64_kernel(
    device const ulong* vals  [[buffer(0)]],
    device Fr*          result [[buffer(1)]],
    uint tid                   [[thread_position_in_grid]]
) {
    result[tid] = fr_from_u64(vals[tid]);
}
