extern "C" __global__ void round_poly_pairs(
    u64 *__restrict__ out,
    const u64 *const *__restrict__ factor_ptrs,
    const u64 *__restrict__ points,
    const u64 *__restrict__ term_coeffs,
    const unsigned int *__restrict__ term_offsets,
    const unsigned int *__restrict__ term_indices,
    unsigned long pair_stride,
    unsigned int num_terms,
    unsigned int degree,
    unsigned long half
) {
    extern __shared__ u64 sdata[];
    u64 *acc = sdata + threadIdx.x * (degree * 4);
    for (unsigned int e = 0; e < degree; e++) {
        for (int k = 0; k < 4; k++) acc[e * 4 + k] = 0;
    }

    unsigned long row = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    if (row < half) {
        for (unsigned int e = 0; e < degree; e++) {
            u64 x[4];
            load4(points + e * 4, x);
            u64 eval[4];
            for (int k = 0; k < 4; k++) eval[k] = 0;
            for (unsigned int t = 0; t < num_terms; t++) {
                u64 prod[4];
                load4(term_coeffs + t * 4, prod);
                unsigned int start = term_offsets[t];
                unsigned int end = term_offsets[t + 1];
                for (unsigned int s = start; s < end; s++) {
                    const u64 *factor = factor_ptrs[term_indices[s]];
                    u64 lo[4], hi[4];
                    load4(factor + (row * 2) * 4, lo);
                    load4(factor + (row * 2 + 1) * 4, hi);
                    u64 diff[4], linear[4];
                    fr_sub(hi, lo, diff);
                    fr_mul(diff, x, linear);
                    fr_add(lo, linear, linear);
                    fr_mul(prod, linear, prod);
                }
                fr_add(eval, prod, eval);
            }
            for (int k = 0; k < 4; k++) acc[e * 4 + k] = eval[k];
        }
    }
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            u64 *other = sdata + (threadIdx.x + stride) * (degree * 4);
            for (unsigned int e = 0; e < degree; e++) {
                u64 t[4];
                fr_add(acc + e * 4, other + e * 4, t);
                for (int k = 0; k < 4; k++) acc[e * 4 + k] = t[k];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        for (unsigned int k = 0; k < degree * 4; k++) {
            out[blockIdx.x * (degree * 4) + k] = acc[k];
        }
    }
}

extern "C" __global__ void eq_round_poly_pairs(
    u64 *__restrict__ out,
    const u64 *const *__restrict__ factor_ptrs,
    const u64 *__restrict__ points,
    const u64 *__restrict__ term_coeffs,
    const unsigned int *__restrict__ term_offsets,
    const unsigned int *__restrict__ term_indices,
    const u64 *__restrict__ e_in,
    const u64 *__restrict__ e_out,
    unsigned long pair_stride,
    unsigned int num_terms,
    unsigned int degree,
    unsigned long half,
    unsigned int in_bits
) {
    extern __shared__ u64 sdata[];
    u64 *acc = sdata + threadIdx.x * (degree * 4);
    for (unsigned int e = 0; e < degree; e++) {
        for (int k = 0; k < 4; k++) acc[e * 4 + k] = 0;
    }

    unsigned long row = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    if (row < half) {
        unsigned long in_mask = ((unsigned long)1 << in_bits) - 1;
        u64 weight[4], ein[4], eout[4];
        load4(e_in + (row & in_mask) * 4, ein);
        load4(e_out + (row >> in_bits) * 4, eout);
        fr_mul(ein, eout, weight);
        for (unsigned int e = 0; e < degree; e++) {
            u64 x[4];
            load4(points + e * 4, x);
            u64 eval[4];
            for (int k = 0; k < 4; k++) eval[k] = 0;
            for (unsigned int t = 0; t < num_terms; t++) {
                u64 prod[4];
                load4(term_coeffs + t * 4, prod);
                unsigned int start = term_offsets[t];
                unsigned int end = term_offsets[t + 1];
                for (unsigned int s = start; s < end; s++) {
                    const u64 *factor = factor_ptrs[term_indices[s]];
                    u64 lo[4], hi[4];
                    load4(factor + (row * 2) * 4, lo);
                    load4(factor + (row * 2 + 1) * 4, hi);
                    u64 diff[4], linear[4];
                    fr_sub(hi, lo, diff);
                    fr_mul(diff, x, linear);
                    fr_add(lo, linear, linear);
                    fr_mul(prod, linear, prod);
                }
                fr_add(eval, prod, eval);
            }
            fr_mul(eval, weight, eval);
            for (int k = 0; k < 4; k++) acc[e * 4 + k] = eval[k];
        }
    }
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            u64 *other = sdata + (threadIdx.x + stride) * (degree * 4);
            for (unsigned int e = 0; e < degree; e++) {
                u64 t[4];
                fr_add(acc + e * 4, other + e * 4, t);
                for (int k = 0; k < 4; k++) acc[e * 4 + k] = t[k];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        for (unsigned int k = 0; k < degree * 4; k++) {
            out[blockIdx.x * (degree * 4) + k] = acc[k];
        }
    }
}

extern "C" __global__ void round_poly_reduce(
    u64 *__restrict__ out,
    const u64 *__restrict__ in,
    unsigned int degree,
    unsigned long n
) {
    extern __shared__ u64 sdata[];
    u64 *acc = sdata + threadIdx.x * (degree * 4);

    unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        for (unsigned int k = 0; k < degree * 4; k++) acc[k] = in[i * (degree * 4) + k];
    } else {
        for (unsigned int k = 0; k < degree * 4; k++) acc[k] = 0;
    }
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            u64 *other = sdata + (threadIdx.x + stride) * (degree * 4);
            for (unsigned int e = 0; e < degree; e++) {
                u64 t[4];
                fr_add(acc + e * 4, other + e * 4, t);
                for (int k = 0; k < 4; k++) acc[e * 4 + k] = t[k];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        for (unsigned int k = 0; k < degree * 4; k++) {
            out[blockIdx.x * (degree * 4) + k] = acc[k];
        }
    }
}
