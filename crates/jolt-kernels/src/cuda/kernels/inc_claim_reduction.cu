template <bool Split>
__device__ __forceinline__ void inc_weight_pair(
    const u64 *lo, const u64 *hi, unsigned int lo_len, unsigned int hi_len,
    unsigned int column, unsigned long long row, u64 *even, u64 *odd) {
    if constexpr (!Split) {
        load4(hi + ((unsigned long long)column * hi_len + 2 * row) * LIMBS, even);
        load4(hi + ((unsigned long long)column * hi_len + 2 * row + 1) * LIMBS, odd);
    } else {
        for (int l = 0; l < LIMBS; l++) even[l] = odd[l] = 0;
        unsigned int inner = (2 * row) % lo_len;
        unsigned long long outer = (2 * row) / lo_len;
        #pragma unroll
        for (unsigned int k = 2 * column; k < 2 * column + 2; k++) {
            u64 a[LIMBS], b[LIMBS], high[LIMBS], term[LIMBS], sum[LIMBS];
            load4(lo + ((unsigned long long)k * lo_len + inner) * LIMBS, a);
            load4(lo + ((unsigned long long)k * lo_len + inner + 1) * LIMBS, b);
            load4(hi + ((unsigned long long)k * hi_len + outer) * LIMBS, high);
            fr_mul(high, a, term);
            fr_add(even, term, sum); load4(sum, even);
            fr_mul(high, b, term);
            fr_add(odd, term, sum); load4(sum, odd);
        }
    }
}

template <bool Native, bool Split>
__device__ __forceinline__ void inc_paired_round(
    const u64 *ram, const u64 *rd, const u64 *dense,
    unsigned int len, const u64 *lo, const u64 *hi,
    unsigned int lo_len, unsigned int hi_len, u64 *partials, u64 *scratch) {
    unsigned long long row = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    u64 acc[2][LIMBS] = {};
    if (row < len / 2) {
        // Each endpoint accumulates at most four signed native products per thread.
        u64 folded[2][2 * UNR_SLOTS] = {};
        #pragma unroll
        for (unsigned int c = 0; c < 2; c++) {
            u64 w0[LIMBS], w1[LIMBS], w2[LIMBS], twice[LIMBS];
            inc_weight_pair<Split>(lo, hi, lo_len, hi_len, c, row, w0, w1);
            fr_add(w1, w1, twice); fr_sub(twice, w0, w2);
            if constexpr (Native) {
                const u64 *column = c == 0 ? ram : rd;
                const u64 *a = column + 4 * row;
                const u64 *b = a + 2;
                u64 neg[LIMBS], zero[LIMBS] = {};
                hf_accumulate_native(a, w0, HF_KIND_TWOS_I128, 0, folded[0]);
                fr_add(w2, w2, twice);
                fr_sub(zero, w2, neg);
                hf_accumulate_native(a, neg, HF_KIND_TWOS_I128, 0, folded[1]);
                hf_accumulate_native(b, twice, HF_KIND_TWOS_I128, 0, folded[1]);
            } else {
                u64 a[LIMBS], b[LIMBS], v2[LIMBS], term[LIMBS], sum[LIMBS];
                load4(dense + ((unsigned long long)c * len + 2 * row) * LIMBS, a);
                load4(dense + ((unsigned long long)c * len + 2 * row + 1) * LIMBS, b);
                fr_mul(w0, a, term); fr_add(acc[0], term, sum); load4(sum, acc[0]);
                fr_add(b, b, twice); fr_sub(twice, a, v2);
                fr_mul(w2, v2, term); fr_add(acc[1], term, sum); load4(sum, acc[1]);
            }
        }
        if constexpr (Native) {
            unr_finalize(folded[0], acc[0]);
            unr_finalize(folded[1], acc[1]);
        }
    }
    lane_block_reduce(scratch, 2, acc, partials);
}

extern "C" __global__ void inc_paired_native_round_kernel(
    const u64 *ram, const u64 *rd, unsigned int len,
    const u64 *lo, const u64 *hi, unsigned int lo_len, unsigned int hi_len,
    u64 *partials) {
    extern __shared__ u64 scratch[];
    inc_paired_round<true, true>(ram, rd, nullptr, len, lo, hi, lo_len, hi_len, partials, scratch);
}

extern "C" __global__ void inc_paired_split_round_kernel(
    const u64 *dense, unsigned int len, const u64 *lo, const u64 *hi,
    unsigned int lo_len, unsigned int hi_len, u64 *partials) {
    extern __shared__ u64 scratch[];
    inc_paired_round<false, true>(nullptr, nullptr, dense, len, lo, hi, lo_len, hi_len, partials, scratch);
}

extern "C" __global__ void inc_paired_dense_round_kernel(
    const u64 *dense, unsigned int len, const u64 *weights, u64 *partials) {
    extern __shared__ u64 scratch[];
    inc_paired_round<false, false>(nullptr, nullptr, dense, len, nullptr, weights, 1, len, partials, scratch);
}

extern "C" __global__ void inc_paired_bind_kernel(
    const u64 *ram, const u64 *rd, unsigned int half,
    u64 r0, u64 r1, u64 r2, u64 r3, u64 *dense) {
    unsigned long long row = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= half) return;
    const u64 challenge[LIMBS] = {r0,r1,r2,r3};
    #pragma unroll
    for (unsigned int c = 0; c < 2; c++) {
        const u64 *column = c == 0 ? ram : rd;
        u64 a[LIMBS], b[LIMBS], diff[LIMBS], term[LIMBS], out[LIMBS];
        hf_promote_narrow(column + 4 * row, HF_KIND_TWOS_I128, 0, a);
        hf_promote_narrow(column + 4 * row + 2, HF_KIND_TWOS_I128, 0, b);
        fr_sub(b, a, diff); fr_mul(diff, challenge, term); fr_add(a, term, out);
        store4(dense + ((unsigned long long)c * half + row) * LIMBS, out);
    }
}

extern "C" __global__ void inc_paired_weights_kernel(
    const u64 *low_scalars, const u64 *hi, unsigned int hi_len, u64 *out) {
    unsigned long long row = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= hi_len) return;
    #pragma unroll
    for (unsigned int c = 0; c < 2; c++) {
        u64 acc[LIMBS] = {};
        #pragma unroll
        for (unsigned int k = 2 * c; k < 2 * c + 2; k++) {
            u64 scalar[LIMBS], value[LIMBS], term[LIMBS], sum[LIMBS];
            load4(low_scalars + k * LIMBS, scalar);
            load4(hi + ((unsigned long long)k * hi_len + row) * LIMBS, value);
            fr_mul(scalar, value, term); fr_add(acc, term, sum); load4(sum, acc);
        }
        store4(out + ((unsigned long long)c * hi_len + row) * LIMBS, acc);
    }
}
