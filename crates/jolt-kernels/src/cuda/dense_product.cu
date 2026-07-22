#define DENSE_PRODUCT_MAX_COEFFS 16

extern "C" __global__ void dense_product_pairs(
    u64 *__restrict__ out,
    const u64 *const *__restrict__ factor_ptrs,
    const u64 *__restrict__ term_coeffs,
    const unsigned int *__restrict__ term_offsets,
    const unsigned int *__restrict__ term_indices,
    unsigned long pair_stride,
    unsigned int num_terms,
    unsigned int degree,
    unsigned long half
) {
    unsigned int width = degree + 1;
    extern __shared__ u64 sdata[];
    u64 *acc = sdata + threadIdx.x * (width * 4);
    for (unsigned int e = 0; e < width; e++) {
        for (int k = 0; k < 4; k++) acc[e * 4 + k] = 0;
    }

    unsigned long row = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    if (row < half) {
        u64 poly[DENSE_PRODUCT_MAX_COEFFS * 4];
        for (unsigned int t = 0; t < num_terms; t++) {
            for (int k = 0; k < 4; k++) {
                poly[k] = term_coeffs[t * 4 + k];
            }
            for (unsigned int e = 1; e < width; e++) {
                for (int k = 0; k < 4; k++) poly[e * 4 + k] = 0;
            }
            unsigned int start = term_offsets[t];
            unsigned int end = term_offsets[t + 1];
            unsigned int cur = 0;
            for (unsigned int s = start; s < end; s++) {
                const u64 *factor = factor_ptrs[term_indices[s]];
                u64 lo[4], hi[4], delta[4];
                load4(factor + (row * 2) * 4, lo);
                load4(factor + (row * 2 + 1) * 4, hi);
                fr_sub(hi, lo, delta);
                for (int index = (int)cur; index >= 0; index--) {
                    u64 prod[4], sum[4];
                    fr_mul(poly + index * 4, delta, prod);
                    fr_add(poly + (index + 1) * 4, prod, sum);
                    for (int k = 0; k < 4; k++) poly[(index + 1) * 4 + k] = sum[k];
                    fr_mul(poly + index * 4, lo, prod);
                    for (int k = 0; k < 4; k++) poly[index * 4 + k] = prod[k];
                }
                cur++;
            }
            for (unsigned int e = 0; e <= cur; e++) {
                u64 sum[4];
                fr_add(acc + e * 4, poly + e * 4, sum);
                for (int k = 0; k < 4; k++) acc[e * 4 + k] = sum[k];
            }
        }
    }
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            u64 *other = sdata + (threadIdx.x + stride) * (width * 4);
            for (unsigned int e = 0; e < width; e++) {
                u64 t[4];
                fr_add(acc + e * 4, other + e * 4, t);
                for (int k = 0; k < 4; k++) acc[e * 4 + k] = t[k];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        for (unsigned int k = 0; k < width * 4; k++) {
            out[blockIdx.x * (width * 4) + k] = acc[k];
        }
    }
}
