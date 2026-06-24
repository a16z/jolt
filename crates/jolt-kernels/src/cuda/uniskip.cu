__device__ __forceinline__ void uniskip_group_matvec(
    const u64 *__restrict__ dots,
    const u64 *__restrict__ coeffs,
    const unsigned int *__restrict__ rows,
    unsigned int group_len,
    unsigned int target,
    u64 *acc
) {
    for (int k = 0; k < 4; k++) acc[k] = 0;
    for (unsigned int pos = 0; pos < group_len; pos++) {
        u64 d[4], c[4], p[4];
        load4(dots + rows[pos] * 4, d);
        load4(coeffs + (target * group_len + pos) * 4, c);
        fr_mul(c, d, p);
        u64 t[4];
        fr_add(acc, p, t);
        for (int k = 0; k < 4; k++) acc[k] = t[k];
    }
}

extern "C" __global__ void uniskip_pairs(
    u64 *__restrict__ out,
    const u64 *__restrict__ row_dots_a,
    const u64 *__restrict__ row_dots_b,
    const u64 *__restrict__ eq_evals,
    const u64 *__restrict__ first_coeffs,
    const u64 *__restrict__ second_coeffs,
    const unsigned int *__restrict__ first_rows,
    const unsigned int *__restrict__ second_rows,
    unsigned int first_len,
    unsigned int second_len,
    unsigned long row_count,
    unsigned int degree,
    unsigned long cycles
) {
    extern __shared__ u64 sdata[];
    u64 *acc = sdata + threadIdx.x * (degree * 4);
    for (unsigned int t = 0; t < degree; t++) {
        for (int k = 0; k < 4; k++) acc[t * 4 + k] = 0;
    }

    unsigned long cycle = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    if (cycle < cycles) {
        const u64 *a = row_dots_a + cycle * row_count * 4;
        const u64 *b = row_dots_b + cycle * row_count * 4;
        u64 eq0[4], eq1[4];
        load4(eq_evals + (cycle * 2) * 4, eq0);
        load4(eq_evals + (cycle * 2 + 1) * 4, eq1);
        for (unsigned int t = 0; t < degree; t++) {
            u64 az0[4], bz0[4], az1[4], bz1[4];
            uniskip_group_matvec(a, first_coeffs, first_rows, first_len, t, az0);
            uniskip_group_matvec(b, first_coeffs, first_rows, first_len, t, bz0);
            uniskip_group_matvec(a, second_coeffs, second_rows, second_len, t, az1);
            uniskip_group_matvec(b, second_coeffs, second_rows, second_len, t, bz1);
            u64 prod0[4], prod1[4], term0[4], term1[4];
            fr_mul(az0, bz0, prod0);
            fr_mul(az1, bz1, prod1);
            fr_mul(eq0, prod0, term0);
            fr_mul(eq1, prod1, term1);
            fr_add(term0, term1, acc + t * 4);
        }
    }
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            u64 *other = sdata + (threadIdx.x + stride) * (degree * 4);
            for (unsigned int t = 0; t < degree; t++) {
                u64 sum[4];
                fr_add(acc + t * 4, other + t * 4, sum);
                for (int k = 0; k < 4; k++) acc[t * 4 + k] = sum[k];
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
