__device__ __forceinline__ void csr_row_dot(
    const u64 *__restrict__ coeffs,
    const unsigned int *__restrict__ vars,
    unsigned int start,
    unsigned int end,
    const u64 *__restrict__ witness_row,
    u64 *acc
) {
    for (int k = 0; k < 4; k++) acc[k] = 0;
    for (unsigned int k = start; k < end; k++) {
        u64 coeff[4], w[4], p[4];
        load4(coeffs + k * 4, coeff);
        load4(witness_row + vars[k] * 4, w);
        fr_mul(coeff, w, p);
        u64 t[4];
        fr_add(acc, p, t);
        for (int i = 0; i < 4; i++) acc[i] = t[i];
    }
}

extern "C" __global__ void row_dots_kernel(
    u64 *__restrict__ a_out,
    u64 *__restrict__ b_out,
    const u64 *__restrict__ witness,
    const unsigned int *__restrict__ a_offsets,
    const unsigned int *__restrict__ a_vars,
    const u64 *__restrict__ a_coeffs,
    const unsigned int *__restrict__ b_offsets,
    const unsigned int *__restrict__ b_vars,
    const u64 *__restrict__ b_coeffs,
    unsigned long row_count,
    unsigned long num_vars_padded,
    unsigned long total
) {
    unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) {
        unsigned long cycle = i / row_count;
        unsigned long row = i % row_count;
        const u64 *witness_row = witness + cycle * num_vars_padded * 4;

        u64 acc[4];
        csr_row_dot(a_coeffs, a_vars, a_offsets[row], a_offsets[row + 1], witness_row, acc);
        store4(a_out + i * 4, acc);
        csr_row_dot(b_coeffs, b_vars, b_offsets[row], b_offsets[row + 1], witness_row, acc);
        store4(b_out + i * 4, acc);
    }
}
