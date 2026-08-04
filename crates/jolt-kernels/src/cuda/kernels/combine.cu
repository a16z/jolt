#define CMB_SCALE_ONE 0
#define CMB_SCALE_NEG_ONE 1
#define CMB_SCALE_TWO_POW_XLEN 2
#define CMB_SCALE_XLEN_ONES 3

#define CMB_NO_PREFIX 0xFFFFFFFFu

__device__ __forceinline__ void cmb_scale(unsigned int scale, u64 *out) {
    switch (scale) {
        case CMB_SCALE_ONE:
            load4(FR_ONE, out);
            return;
        case CMB_SCALE_NEG_ONE: {
            u64 zero[LIMBS] = {0, 0, 0, 0}, one[LIMBS];
            load4(FR_ONE, one);
            fr_sub(zero, one, out);
            return;
        }
        case CMB_SCALE_TWO_POW_XLEN: {
            u64 raw[LIMBS] = {0, 1, 0, 0};
            fr_to_mont(raw, out);
            return;
        }
        default: {
            u64 raw[LIMBS] = {0xFFFFFFFFFFFFFFFFULL, 0, 0, 0};
            fr_to_mont(raw, out);
            return;
        }
    }
}

extern "C" __global__ void cmb_combine_kernel(const unsigned int *__restrict__ scales,
                                              const unsigned int *__restrict__ prefix_ids,
                                              const unsigned int *__restrict__ suffix_slots,
                                              unsigned int term_count,
                                              const u64 *__restrict__ prefixes,
                                              const u64 *const *__restrict__ suffix_columns,
                                              u64 *__restrict__ out,
                                              unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    u64 acc[LIMBS] = {0, 0, 0, 0};
    for (unsigned int t = 0; t < term_count; t++) {
        u64 value[LIMBS];
        load4(suffix_columns[suffix_slots[t]] + (unsigned long long)i * LIMBS, value);

        unsigned int prefix = prefix_ids[t];
        if (prefix != CMB_NO_PREFIX) {
            u64 p[LIMBS], product[LIMBS];
            load4(prefixes + (unsigned long long)prefix * LIMBS, p);
            fr_mul(value, p, product);
            store4(value, product);
        }

        if (scales[t] != CMB_SCALE_ONE) {
            u64 s[LIMBS], scaled[LIMBS];
            cmb_scale(scales[t], s);
            fr_mul(value, s, scaled);
            store4(value, scaled);
        }

        u64 sum[LIMBS];
        fr_add(acc, value, sum);
        store4(acc, sum);
    }
    store4(out + (unsigned long long)i * LIMBS, acc);
}
