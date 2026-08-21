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
