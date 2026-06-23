__device__ __forceinline__ void csr_group_dot(
    const u64 *__restrict__ coeffs,
    const unsigned int *__restrict__ vars,
    const unsigned int *__restrict__ offsets,
    unsigned int entry_start,
    unsigned int entry_end,
    const u64 *__restrict__ witness_row,
    u64 *acc
) {
    for (int k = 0; k < 4; k++) acc[k] = 0;
    for (unsigned int e = entry_start; e < entry_end; e++) {
        unsigned int start = offsets[e];
        unsigned int end = offsets[e + 1];
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
}

extern "C" __global__ void dense_outer_fused_kernel(
    u64 *__restrict__ eq_out,
    u64 *__restrict__ az_out,
    u64 *__restrict__ bz_out,
    const u64 *__restrict__ eq_evals,
    const u64 *__restrict__ scale,
    const u64 *__restrict__ witness,
    const unsigned int *__restrict__ a_offsets,
    const unsigned int *__restrict__ a_vars,
    const u64 *__restrict__ a_coeffs,
    const unsigned int *__restrict__ b_offsets,
    const unsigned int *__restrict__ b_vars,
    const u64 *__restrict__ b_coeffs,
    unsigned int split,
    unsigned int total_entries,
    unsigned long num_vars_padded,
    unsigned long cycles
) {
    unsigned long cycle = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    if (cycle < cycles) {
        unsigned long index = cycle * 2;
        const u64 *witness_row = witness + cycle * num_vars_padded * 4;

        u64 s[4];
        load4(scale, s);
        u64 e0[4], e1[4], r[4];
        load4(eq_evals + index * 4, e0);
        load4(eq_evals + (index + 1) * 4, e1);
        fr_mul(e0, s, r);
        store4(eq_out + index * 4, r);
        fr_mul(e1, s, r);
        store4(eq_out + (index + 1) * 4, r);

        u64 az0[4], az1[4], bz0[4], bz1[4];
        csr_group_dot(a_coeffs, a_vars, a_offsets, 0, split, witness_row, az0);
        csr_group_dot(a_coeffs, a_vars, a_offsets, split, total_entries, witness_row, az1);
        csr_group_dot(b_coeffs, b_vars, b_offsets, 0, split, witness_row, bz0);
        csr_group_dot(b_coeffs, b_vars, b_offsets, split, total_entries, witness_row, bz1);
        store4(az_out + index * 4, az0);
        store4(az_out + (index + 1) * 4, az1);
        store4(bz_out + index * 4, bz0);
        store4(bz_out + (index + 1) * 4, bz1);
    }
}
