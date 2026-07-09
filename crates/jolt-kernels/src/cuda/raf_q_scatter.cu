__device__ __constant__ u64 RAF_R2[4] = {
    0x1bb8e645ae216da7ULL, 0x53fe3ab1e35c59e3ULL,
    0x8c49833d53bb8085ULL, 0x0216d0b17f4e44a5ULL
};

__device__ __forceinline__ void uninterleave_u128(u128 val, u64 *x_out, u64 *y_out) {
    u128 m1 = ((u128)0x5555555555555555ULL << 64) | 0x5555555555555555ULL;
    u128 m2 = ((u128)0x3333333333333333ULL << 64) | 0x3333333333333333ULL;
    u128 m3 = ((u128)0x0F0F0F0F0F0F0F0FULL << 64) | 0x0F0F0F0F0F0F0F0FULL;
    u128 m4 = ((u128)0x00FF00FF00FF00FFULL << 64) | 0x00FF00FF00FF00FFULL;
    u128 m5 = ((u128)0x0000FFFF0000FFFFULL << 64) | 0x0000FFFF0000FFFFULL;
    u128 m6 = ((u128)0x00000000FFFFFFFFULL << 64) | 0x00000000FFFFFFFFULL;
    u128 lo = ((u128)0ULL << 64) | 0xFFFFFFFFFFFFFFFFULL;

    u128 x = (val >> 1) & m1;
    u128 y = val & m1;
    x = (x | (x >> 1)) & m2;
    x = (x | (x >> 2)) & m3;
    x = (x | (x >> 4)) & m4;
    x = (x | (x >> 8)) & m5;
    x = (x | (x >> 16)) & m6;
    x = (x | (x >> 32)) & lo;
    y = (y | (y >> 1)) & m2;
    y = (y | (y >> 2)) & m3;
    y = (y | (y >> 4)) & m4;
    y = (y | (y >> 8)) & m5;
    y = (y | (y >> 16)) & m6;
    y = (y | (y >> 32)) & lo;
    *x_out = (u64)x;
    *y_out = (u64)y;
}

__device__ __forceinline__ void raf_to_mont(const u64 *raw, u64 *out) {
    fr_mul(raw, RAF_R2, out);
}

__device__ __forceinline__ void raf_add_inplace(u64 *acc, const u64 *v) {
    u64 t[4];
    fr_add(acc, v, t);
    for (int k = 0; k < 4; k++) acc[k] = t[k];
}

extern "C" __global__ void raf_q_scatter(
    u64 *__restrict__ worker_banks,
    const u64 *__restrict__ weight,
    const u64 *__restrict__ lookup_index_lo,
    const u64 *__restrict__ lookup_index_hi,
    const unsigned char *__restrict__ is_interleaved,
    unsigned long suffix_len,
    unsigned long poly_len,
    unsigned long trace_len,
    unsigned long num_workers
) {
    unsigned long worker = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    if (worker >= num_workers) return;

    unsigned long bank_stride = 5UL * poly_len * 4UL;
    u64 *bank = worker_banks + worker * bank_stride;
    unsigned long shift_half_off = 0;
    unsigned long left_off = poly_len * 4UL;
    unsigned long right_off = 2UL * poly_len * 4UL;
    unsigned long shift_full_off = 3UL * poly_len * 4UL;
    unsigned long identity_off = 4UL * poly_len * 4UL;

    u128 suffix_mask = (suffix_len >= 128)
        ? (u128)-1
        : (((u128)1 << suffix_len) - 1);
    unsigned long index_mask = poly_len - 1UL;

    for (unsigned long c = worker; c < trace_len; c += num_workers) {
        u128 lookup_index = ((u128)lookup_index_hi[c] << 64) | (u128)lookup_index_lo[c];
        unsigned long index = ((unsigned long)(lookup_index >> suffix_len)) & index_mask;
        u128 suffix_bits = lookup_index & suffix_mask;

        u64 w[4];
        load4(weight + c * 4, w);

        if (is_interleaved[c]) {
            raf_add_inplace(bank + shift_half_off + index * 4, w);
            u64 left_suffix, right_suffix;
            uninterleave_u128(suffix_bits, &left_suffix, &right_suffix);
            if (left_suffix != 0) {
                u64 s_raw[4] = {left_suffix, 0, 0, 0};
                u64 s_mont[4], contrib[4];
                raf_to_mont(s_raw, s_mont);
                fr_mul(w, s_mont, contrib);
                raf_add_inplace(bank + left_off + index * 4, contrib);
            }
            if (right_suffix != 0) {
                u64 s_raw[4] = {right_suffix, 0, 0, 0};
                u64 s_mont[4], contrib[4];
                raf_to_mont(s_raw, s_mont);
                fr_mul(w, s_mont, contrib);
                raf_add_inplace(bank + right_off + index * 4, contrib);
            }
        } else {
            raf_add_inplace(bank + shift_full_off + index * 4, w);
            if (suffix_bits != 0) {
                u64 s_raw[4] = {
                    (u64)suffix_bits,
                    (u64)(suffix_bits >> 64),
                    0, 0
                };
                u64 s_mont[4], contrib[4];
                raf_to_mont(s_raw, s_mont);
                fr_mul(w, s_mont, contrib);
                raf_add_inplace(bank + identity_off + index * 4, contrib);
            }
        }
    }
}

extern "C" __global__ void raf_q_scatter_reduce(
    u64 *__restrict__ banks,
    const u64 *__restrict__ worker_banks,
    unsigned long slots,
    unsigned long num_workers
) {
    unsigned long slot = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= slots) return;

    unsigned long bank_stride = slots * 4UL;
    u64 acc[4];
    load4(worker_banks + slot * 4, acc);
    for (unsigned long w = 1; w < num_workers; w++) {
        u64 v[4];
        load4(worker_banks + w * bank_stride + slot * 4, v);
        raf_add_inplace(acc, v);
    }
    store4(banks + slot * 4, acc);
}
