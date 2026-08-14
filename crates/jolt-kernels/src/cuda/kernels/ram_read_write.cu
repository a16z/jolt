#define RRW_COLD 0xFFFFFFFFu

extern "C" __global__ void rrw_flags_kernel(const unsigned int *__restrict__ address,
                                           unsigned int cycles,
                                           unsigned int *__restrict__ flags) {
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= cycles) return;
    flags[j] = (address[j] == RRW_COLD) ? 0u : 1u;
}

extern "C" __global__ void rrw_inc_kernel(const u64 *__restrict__ read_value,
                                         const u64 *__restrict__ write_value,
                                         unsigned int cycles, u64 *__restrict__ out) {
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= cycles) return;
    u64 pre_raw[LIMBS] = {read_value[j], 0, 0, 0};
    u64 post_raw[LIMBS] = {write_value[j], 0, 0, 0};
    u64 pre[LIMBS], post[LIMBS], delta[LIMBS];
    fr_to_mont(pre_raw, pre);
    fr_to_mont(post_raw, post);
    fr_sub(post, pre, delta);
    store4(out + (unsigned long long)j * LIMBS, delta);
}

extern "C" __global__ void rrw_scatter_kernel(
    const unsigned int *__restrict__ address, const u64 *__restrict__ read_value,
    const u64 *__restrict__ write_value, const unsigned int *__restrict__ offsets,
    unsigned int cycles, unsigned int *__restrict__ out_rows,
    unsigned int *__restrict__ out_cols, u64 *__restrict__ out_val,
    u64 *__restrict__ out_prev, u64 *__restrict__ out_next, u64 *__restrict__ out_coeffs) {
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= cycles) return;
    unsigned int col = address[j];
    if (col == RRW_COLD) return;

    unsigned int k = offsets[j];
    out_rows[k] = j;
    out_cols[k] = col;
    out_prev[k] = read_value[j];
    out_next[k] = write_value[j];

    u64 raw[LIMBS] = {read_value[j], 0, 0, 0};
    u64 val[LIMBS];
    fr_to_mont(raw, val);
    store4(out_val + (unsigned long long)k * LIMBS, val);

    u64 one[LIMBS];
    load4(FR_ONE, one);
    store4(out_coeffs + (unsigned long long)k * LIMBS, one);
}
