__device__ __forceinline__ void group_matvec(
    const u64 *__restrict__ dots,
    const u64 *__restrict__ weights,
    const unsigned int *__restrict__ rows,
    unsigned int group_len,
    u64 *acc
) {
    for (int k = 0; k < 4; k++) acc[k] = 0;
    for (unsigned int j = 0; j < group_len; j++) {
        u64 d[4], w[4], p[4];
        load4(dots + rows[j] * 4, d);
        load4(weights + j * 4, w);
        fr_mul(w, d, p);
        u64 t[4];
        fr_add(acc, p, t);
        for (int k = 0; k < 4; k++) acc[k] = t[k];
    }
}

extern "C" __global__ void dense_outer(
    u64 *__restrict__ eq_out,
    u64 *__restrict__ az_out,
    u64 *__restrict__ bz_out,
    const u64 *__restrict__ eq_evals,
    const u64 *__restrict__ scale,
    const u64 *__restrict__ weights,
    const u64 *__restrict__ row_dots_a,
    const u64 *__restrict__ row_dots_b,
    const unsigned int *__restrict__ first_rows,
    const unsigned int *__restrict__ second_rows,
    unsigned int first_len,
    unsigned int second_len,
    unsigned long row_count,
    unsigned long cycles
) {
    unsigned long cycle = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    if (cycle < cycles) {
        unsigned long index = cycle * 2;
        const u64 *a = row_dots_a + cycle * row_count * 4;
        const u64 *b = row_dots_b + cycle * row_count * 4;

        u64 s[4];
        load4(scale, s);
        u64 e0[4], e1[4], r[4];
        load4(eq_evals + index * 4, e0);
        load4(eq_evals + (index + 1) * 4, e1);
        fr_mul(e0, s, r);
        store4(eq_out + index * 4, r);
        fr_mul(e1, s, r);
        store4(eq_out + (index + 1) * 4, r);

        u64 az0[4], bz0[4], az1[4], bz1[4];
        group_matvec(a, weights, first_rows, first_len, az0);
        group_matvec(b, weights, first_rows, first_len, bz0);
        group_matvec(a, weights, second_rows, second_len, az1);
        group_matvec(b, weights, second_rows, second_len, bz1);
        store4(az_out + index * 4, az0);
        store4(bz_out + index * 4, bz0);
        store4(az_out + (index + 1) * 4, az1);
        store4(bz_out + (index + 1) * 4, bz1);
    }
}
