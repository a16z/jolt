extern "C" __global__ void gruen_round_poly_pairs(
    u64 *__restrict__ out,
    const u64 *__restrict__ factors,
    const u64 *__restrict__ term_coeffs,
    const unsigned int *__restrict__ term_offsets,
    const unsigned int *__restrict__ term_indices,
    const u64 *__restrict__ e_in,
    const u64 *__restrict__ e_out,
    unsigned long pair_stride,
    unsigned int num_terms,
    unsigned long half,
    unsigned long in_pairs,
    unsigned int low_round
) {
    extern __shared__ u64 sdata[];
    u64 *acc = sdata + threadIdx.x * 8;
    for (int k = 0; k < 8; k++) acc[k] = 0;

    unsigned long row = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    if (row < half) {
        u64 weight[4];
        if (low_round) {
            unsigned long x_out = row / in_pairs;
            unsigned long p = row % in_pairs;
            u64 ein_sum[4], eo[4];
            fr_add(e_in + (2 * p) * 4, e_in + (2 * p + 1) * 4, ein_sum);
            load4(e_out + x_out * 4, eo);
            fr_mul(eo, ein_sum, weight);
        } else {
            u64 eout_sum[4], ei[4];
            fr_add(e_out + (2 * row) * 4, e_out + (2 * row + 1) * 4, eout_sum);
            load4(e_in, ei);
            fr_mul(ei, eout_sum, weight);
        }

        u64 q_constant[4], q_top[4];
        for (int k = 0; k < 4; k++) { q_constant[k] = 0; q_top[k] = 0; }
        for (unsigned int t = 0; t < num_terms; t++) {
            u64 lo_prod[4], delta_prod[4];
            for (int k = 0; k < 4; k++) {
                lo_prod[k] = term_coeffs[t * 4 + k];
                delta_prod[k] = term_coeffs[t * 4 + k];
            }
            unsigned int start = term_offsets[t];
            unsigned int end = term_offsets[t + 1];
            for (unsigned int s = start; s < end; s++) {
                const u64 *factor = factors + term_indices[s] * pair_stride * 2 * 4;
                u64 lo[4], hi[4], delta[4], tmp[4];
                load4(factor + (row * 2) * 4, lo);
                load4(factor + (row * 2 + 1) * 4, hi);
                fr_sub(hi, lo, delta);
                fr_mul(lo_prod, lo, tmp);
                for (int k = 0; k < 4; k++) lo_prod[k] = tmp[k];
                fr_mul(delta_prod, delta, tmp);
                for (int k = 0; k < 4; k++) delta_prod[k] = tmp[k];
            }
            u64 sum[4];
            fr_add(q_constant, lo_prod, sum);
            for (int k = 0; k < 4; k++) q_constant[k] = sum[k];
            fr_add(q_top, delta_prod, sum);
            for (int k = 0; k < 4; k++) q_top[k] = sum[k];
        }
        fr_mul(q_constant, weight, acc + 0);
        fr_mul(q_top, weight, acc + 4);
    }
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            u64 *other = sdata + (threadIdx.x + stride) * 8;
            u64 t[4];
            fr_add(acc + 0, other + 0, t);
            for (int k = 0; k < 4; k++) acc[k] = t[k];
            fr_add(acc + 4, other + 4, t);
            for (int k = 0; k < 4; k++) acc[4 + k] = t[k];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        for (int k = 0; k < 8; k++) out[blockIdx.x * 8 + k] = acc[k];
    }
}
