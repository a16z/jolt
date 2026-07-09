__device__ __forceinline__ void uninterleave_u128(u128 val, u64 *x_out, u64 *y_out) {
    u128 m1 = ((u128)0x5555555555555555ULL << 64) | 0x5555555555555555ULL;
    u128 m2 = ((u128)0x3333333333333333ULL << 64) | 0x3333333333333333ULL;
    u128 m3 = ((u128)0x0F0F0F0F0F0F0F0FULL << 64) | 0x0F0F0F0F0F0F0F0FULL;
    u128 m4 = ((u128)0x00FF00FF00FF00FFULL << 64) | 0x00FF00FF00FF00FFULL;
    u128 m5 = ((u128)0x0000FFFF0000FFFFULL << 64) | 0x0000FFFF0000FFFFULL;
    u128 m6 = ((u128)0x00000000FFFFFFFFULL << 64) | 0x00000000FFFFFFFFULL;

    u128 x = (val >> 1) & m1;
    u128 y = val & m1;
    x = (x | (x >> 1)) & m2;
    x = (x | (x >> 2)) & m3;
    x = (x | (x >> 4)) & m4;
    x = (x | (x >> 8)) & m5;
    x = (x | (x >> 16)) & m6;
    x = (x | (x >> 32)) & (((u128)0ULL << 64) | 0xFFFFFFFFFFFFFFFFULL);
    y = (y | (y >> 1)) & m2;
    y = (y | (y >> 2)) & m3;
    y = (y | (y >> 4)) & m4;
    y = (y | (y >> 8)) & m5;
    y = (y | (y >> 16)) & m6;
    y = (y | (y >> 32)) & (((u128)0ULL << 64) | 0xFFFFFFFFFFFFFFFFULL);
    *x_out = (u64)x;
    *y_out = (u64)y;
}

extern "C" __global__ void raf_q_scatter(
    u64 *__restrict__ banks,
    const u64 *__restrict__ weight,
    const u64 *__restrict__ lookup_index_lo,
    const u64 *__restrict__ lookup_index_hi,
    const unsigned char *__restrict__ is_interleaved,
    unsigned long suffix_len,
    unsigned long poly_len,
    unsigned long trace_len
) {
    (void)banks;
    (void)weight;
    (void)lookup_index_lo;
    (void)lookup_index_hi;
    (void)is_interleaved;
    (void)suffix_len;
    (void)poly_len;
    (void)trace_len;
    (void)uninterleave_u128;
}
