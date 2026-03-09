#include "bn254_fr.metal"

// Pairwise interpolation (LowToHigh): out[i] = buf[2i] + s * (buf[2i+1] - buf[2i])
// Dispatched with n/2 threads.
kernel void fr_interpolate_low_kernel(
    device const Fr* buf    [[buffer(0)]],
    device const Fr* scalar [[buffer(1)]],
    device Fr*       out    [[buffer(2)]],
    uint tid                [[thread_position_in_grid]]
) {
    Fr s = scalar[0];
    Fr lo = buf[2 * tid];
    Fr hi = buf[2 * tid + 1];
    out[tid] = fr_add(lo, fr_mul(s, fr_sub(hi, lo)));
}

// In-place interpolation (HighToLow): buf[i] = buf[i] + s * (buf[i+half_n] - buf[i])
// Safe in-place: each thread reads buf[tid] and buf[tid+half_n], writes buf[tid].
// No cross-thread aliasing. Dispatched with n/2 threads. params[0] = half_n.
kernel void fr_interpolate_inplace_high_kernel(
    device Fr*         buf    [[buffer(0)]],
    device const Fr*   scalar [[buffer(1)]],
    device const uint* params [[buffer(2)]],
    uint tid                  [[thread_position_in_grid]]
) {
    uint half_n = params[0];
    Fr s = scalar[0];
    Fr lo = buf[tid];
    Fr hi = buf[tid + half_n];
    buf[tid] = fr_add(lo, fr_mul(s, fr_sub(hi, lo)));
}

// Product table round (split-half): table[j] *= (1-r), table[j+prev_len] = table[j] * r.
// Matches the CPU parallel path. Safe in-place: thread j writes table[j] and
// table[j+prev_len], no cross-thread aliasing. params[0] = prev_len.
kernel void fr_product_table_round_kernel(
    device Fr*         table       [[buffer(0)]],
    device const Fr*   r_val       [[buffer(1)]],
    device const Fr*   one_minus_r [[buffer(2)]],
    device const uint* params      [[buffer(3)]],
    uint tid                       [[thread_position_in_grid]]
) {
    uint prev_len = params[0];
    Fr base = table[tid];
    table[tid + prev_len] = fr_mul(base, r_val[0]);
    table[tid]            = fr_mul(base, one_minus_r[0]);
}
