#define HF_KIND_U64 0u
#define HF_KIND_U128 1u
#define HF_KIND_TWOS_I128 2u
#define HF_KIND_FIELD 3u
#define HF_KIND_U32 4u
#define HF_KIND_BIT 5u

__device__ __forceinline__ void hf_promote_narrow(const u64 *__restrict__ entry,
                                                 unsigned int kind,
                                                 unsigned int bit,
                                                 u64 *out) {
    bool wide = kind == HF_KIND_U128 || kind == HF_KIND_TWOS_I128;
    u64 lo = entry[0];
    if (kind == HF_KIND_U32) lo &= 0xFFFFFFFFull;
    if (kind == HF_KIND_BIT) lo = (lo >> bit) & 1ull;
    u64 hi = wide ? entry[1] : 0ull;
    bool negative = kind == HF_KIND_TWOS_I128 && (hi >> 63) != 0ull;
    if (negative) {
        u64 next = ~lo + 1ull;
        hi = ~hi + (next == 0ull ? 1ull : 0ull);
        lo = next;
    }
    u64 raw[LIMBS] = {lo, hi, 0, 0};
    fr_to_mont(raw, out);
    if (negative) {
        u64 zero[LIMBS] = {0, 0, 0, 0};
        u64 neg[LIMBS];
        fr_sub(zero, out, neg);
        for (int l = 0; l < LIMBS; l++) out[l] = neg[l];
    }
}

__device__ __forceinline__ void hf_load_value(const u64 *__restrict__ words,
                                              unsigned int kind,
                                              unsigned int stride,
                                              unsigned int offset,
                                              unsigned int bit,
                                              unsigned long long index,
                                              u64 *out) {
    if (kind == HF_KIND_FIELD) {
        load4(words + index * LIMBS, out);
        return;
    }
    hf_promote_narrow(words + (unsigned long long)offset + index * (unsigned long long)stride,
                      kind, bit, out);
}

__device__ __forceinline__ void hf_accumulate_native(const u64 *entry,
                                                     const u64 *weight,
                                                     unsigned int kind,
                                                     unsigned int bit,
                                                     u64 *folded) {
    u64 words[2] = {entry[0], 0};
    if (kind == HF_KIND_BIT) {
        if ((words[0] >> bit) & 1ULL) unr_add_field(folded, weight);
        return;
    }
    if (kind == HF_KIND_U32) words[0] &= 0xFFFFFFFFULL;
    bool wide = kind == HF_KIND_U128 || kind == HF_KIND_TWOS_I128;
    if (wide) words[1] = entry[1];
    if (kind == HF_KIND_TWOS_I128 && (words[1] >> 63)) {
        u64 next = ~words[0] + 1ULL;
        words[1] = ~words[1] + (next == 0ULL ? 1ULL : 0ULL);
        words[0] = next;
        u64 zero[LIMBS] = {0, 0, 0, 0}, negative[LIMBS];
        fr_sub(zero, weight, negative);
        unr_mul_words(negative, words, 2, folded);
    } else {
        unr_mul_words(weight, words, wide ? 2u : 1u, folded);
    }
}

__device__ __forceinline__ void hf_fold(const u64 *column, const u64 *weights,
                                        unsigned int sum_len, unsigned int first,
                                        unsigned int step, unsigned long long base,
                                        unsigned int sum_stride, unsigned int kind,
                                        unsigned int entry_stride,
                                        unsigned int entry_offset,
                                        unsigned int entry_bit, u64 *acc) {
    for (int l = 0; l < LIMBS; l++) acc[l] = 0;
    if (kind == HF_KIND_FIELD) {
        for (unsigned long long b = first; b < sum_len; b += step) {
            u64 weight[LIMBS], value[LIMBS], term[LIMBS], sum[LIMBS];
            load4(weights + b * LIMBS, weight);
            load4(column + (base + b * sum_stride) * LIMBS, value);
            fr_mul(weight, value, term);
            fr_add(acc, term, sum);
            for (int l = 0; l < LIMBS; l++) acc[l] = sum[l];
        }
        return;
    }

    // A native term adds at most two 32-bit pieces per slot. Reducing every
    // 4096 terms keeps each u64 slot below 2^45, including full-width u128s.
    constexpr unsigned int NATIVE_BATCH = 4096;
    unsigned long long b = first;
    while (b < sum_len) {
        u64 folded[2 * UNR_SLOTS];
        unr_zero(folded);
        for (unsigned int n = 0; n < NATIVE_BATCH && b < sum_len; n++, b += step) {
            u64 weight[LIMBS];
            load4(weights + b * LIMBS, weight);
            unsigned long long index = base + b * sum_stride;
            const u64 *entry = column + entry_offset + index * entry_stride;
            hf_accumulate_native(entry, weight, kind, entry_bit, folded);
        }
        u64 partial[LIMBS], sum[LIMBS];
        unr_finalize(folded, partial);
        fr_add(acc, partial, sum);
        for (int l = 0; l < LIMBS; l++) acc[l] = sum[l];
    }
}

extern "C" __global__ void hf_combined_u64_kernel(
    const u64 *__restrict__ column0, const u64 *__restrict__ column1,
    const u64 *__restrict__ column2, const u64 *__restrict__ weights,
    u64 *__restrict__ out, unsigned int out_len, unsigned int sum_len,
    unsigned int stride0, unsigned int offset0,
    unsigned int stride1, unsigned int offset1,
    unsigned int stride2, unsigned int offset2,
    u64 c00, u64 c01, u64 c02, u64 c03,
    u64 c10, u64 c11, u64 c12, u64 c13,
    u64 c20, u64 c21, u64 c22, u64 c23,
    u64 bias0, u64 bias1, u64 bias2, u64 bias3) {
    unsigned int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= out_len) return;
    u64 acc[3][LIMBS] = {};
    unsigned long long b = 0;
    while (b < sum_len) {
        u64 folded[3][2 * UNR_SLOTS] = {};
        // One native product per slot per term; reducing every 4096 terms
        // keeps each split accumulator lane below 2^44 for full-width u64s.
        for (unsigned int n = 0; n < 4096 && b < sum_len; n++, b++) {
            unsigned long long index = a + b * out_len;
            u64 weight[LIMBS];
            load4(weights + b * LIMBS, weight);
            u64 v0 = column0[offset0 + index * stride0];
            u64 v1 = column1[offset1 + index * stride1];
            u64 v2 = column2[offset2 + index * stride2];
            unr_mul_words(weight, &v0, 1, folded[0]);
            unr_mul_words(weight, &v1, 1, folded[1]);
            unr_mul_words(weight, &v2, 1, folded[2]);
        }
        #pragma unroll
        for (int c = 0; c < 3; c++) {
            u64 partial[LIMBS], sum[LIMBS];
            unr_finalize(folded[c], partial);
            fr_add(acc[c], partial, sum);
            #pragma unroll
            for (int l = 0; l < LIMBS; l++) acc[c][l] = sum[l];
        }
    }
    const u64 coefficients[3][LIMBS] = {
        {c00, c01, c02, c03}, {c10, c11, c12, c13}, {c20, c21, c22, c23}
    };
    u64 total[LIMBS] = {bias0, bias1, bias2, bias3};
    #pragma unroll
    for (int c = 0; c < 3; c++) {
        u64 term[LIMBS], sum[LIMBS];
        fr_mul(acc[c], coefficients[c], term);
        fr_add(total, term, sum);
        #pragma unroll
        for (int l = 0; l < LIMBS; l++) total[l] = sum[l];
    }
    store4(out + (unsigned long long)a * LIMBS, total);
}

extern "C" __global__ void hf_combined_pairs_kernel(
    const unsigned long long *__restrict__ columns,
    const unsigned int *__restrict__ descriptors,
    const u64 *__restrict__ coefficients,
    const u64 *__restrict__ weights0, const u64 *__restrict__ weights1,
    u64 *__restrict__ out0, u64 *__restrict__ out1,
    unsigned int out_len, unsigned int sum_len,
    u64 bias00, u64 bias01, u64 bias02, u64 bias03,
    u64 bias10, u64 bias11, u64 bias12, u64 bias13) {
    unsigned int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= out_len) return;
    u64 acc0[LIMBS] = {bias00, bias01, bias02, bias03};
    u64 acc1[LIMBS] = {bias10, bias11, bias12, bias13};
    unsigned long long b = 0;
    while (b < sum_len) {
        u64 folded0[2 * PA_SLOTS], folded1[2 * PA_SLOTS];
        pa_zero(folded0);
        pa_zero(folded1);
        // At most eight 32-bit pieces per slot per product keep each lane
        // below 2^47 over this batch, including full-width field operands.
        for (unsigned int n = 0; n < 4096 && b < sum_len; n++, b++) {
            unsigned long long index = a + b * out_len;
            u64 combined[2 * UNR_SLOTS];
            unr_zero(combined);
            for (unsigned int c = 0; c < 4; c++) {
                const u64 *column = reinterpret_cast<const u64 *>(columns[c]);
                const unsigned int *desc = descriptors + 4 * c;
                u64 coefficient[LIMBS];
                load4(coefficients + c * LIMBS, coefficient);
                if (desc[0] == HF_KIND_FIELD) {
                    u64 value[LIMBS], term[LIMBS];
                    load4(column + index * LIMBS, value);
                    fr_mul(coefficient, value, term);
                    unr_add_field(combined, term);
                } else {
                    hf_accumulate_native(column + desc[2] + index * desc[1],
                                         coefficient, desc[0], desc[3], combined);
                }
            }
            u64 value[LIMBS], weight[LIMBS];
            unr_finalize(combined, value);
            load4(weights0 + b * LIMBS, weight);
            pa_fold_mul_accum(value, weight, folded0);
            load4(weights1 + b * LIMBS, weight);
            pa_fold_mul_accum(value, weight, folded1);
        }
        u64 partial[LIMBS], sum[LIMBS];
        pa_finalize(folded0, partial);
        fr_add(acc0, partial, sum);
        for (int l = 0; l < LIMBS; l++) acc0[l] = sum[l];
        pa_finalize(folded1, partial);
        fr_add(acc1, partial, sum);
        for (int l = 0; l < LIMBS; l++) acc1[l] = sum[l];
    }
    store4(out0 + (unsigned long long)a * LIMBS, acc0);
    store4(out1 + (unsigned long long)a * LIMBS, acc1);
}

extern "C" __global__ void hf_half_fold_kernel(const u64 *__restrict__ column,
                                              const u64 *__restrict__ weights,
                                              u64 *__restrict__ out,
                                              u64 scale0, u64 scale1, u64 scale2, u64 scale3,
                                              u64 bias0, u64 bias1, u64 bias2, u64 bias3,
                                              unsigned int out_len,
                                              unsigned int sum_len,
                                              unsigned int out_stride,
                                              unsigned int sum_stride,
                                              unsigned int accumulate,
                                              unsigned int kind,
                                              unsigned int entry_stride,
                                              unsigned int entry_offset,
                                              unsigned int entry_bit) {
    unsigned int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= out_len) return;

    u64 acc[LIMBS];
    unsigned long long base = (unsigned long long)a * (unsigned long long)out_stride;
    hf_fold(column, weights, sum_len, 0, 1, base, sum_stride, kind,
            entry_stride, entry_offset, entry_bit, acc);

    u64 scale[LIMBS] = {scale0, scale1, scale2, scale3};
    u64 scaled[LIMBS], total[LIMBS];
    fr_mul(acc, scale, scaled);
    if (accumulate) {
        u64 previous[LIMBS];
        load4(out + (unsigned long long)a * LIMBS, previous);
        fr_add(previous, scaled, total);
    } else {
        u64 bias[LIMBS] = {bias0, bias1, bias2, bias3};
        fr_add(scaled, bias, total);
    }
    store4(out + (unsigned long long)a * LIMBS, total);
}

extern "C" __global__ void hf_row_fold_kernel(const u64 *__restrict__ column,
                                             const u64 *__restrict__ weights,
                                             u64 *__restrict__ out,
                                             u64 scale0, u64 scale1, u64 scale2, u64 scale3,
                                             u64 bias0, u64 bias1, u64 bias2, u64 bias3,
                                             unsigned int sum_len,
                                             unsigned int accumulate,
                                             unsigned int kind,
                                             unsigned int entry_stride,
                                             unsigned int entry_offset,
                                             unsigned int entry_bit) {
    extern __shared__ u64 scratch[];
    unsigned int a = blockIdx.x;
    unsigned int tid = threadIdx.x;

    u64 acc[LIMBS];
    unsigned long long base = (unsigned long long)a * (unsigned long long)sum_len;
    hf_fold(column, weights, sum_len, tid, blockDim.x, base, 1, kind,
            entry_stride, entry_offset, entry_bit, acc);

    store4(scratch + tid * LIMBS, acc);
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            u64 x[LIMBS], y[LIMBS], s[LIMBS];
            load4(scratch + tid * LIMBS, x);
            load4(scratch + (tid + stride) * LIMBS, y);
            fr_add(x, y, s);
            store4(scratch + tid * LIMBS, s);
        }
        __syncthreads();
    }

    if (tid != 0) return;
    u64 total[LIMBS], scale[LIMBS] = {scale0, scale1, scale2, scale3};
    load4(scratch, total);
    u64 scaled[LIMBS], result[LIMBS];
    fr_mul(total, scale, scaled);
    if (accumulate) {
        u64 previous[LIMBS];
        load4(out + (unsigned long long)a * LIMBS, previous);
        fr_add(previous, scaled, result);
    } else {
        u64 bias[LIMBS] = {bias0, bias1, bias2, bias3};
        fr_add(scaled, bias, result);
    }
    store4(out + (unsigned long long)a * LIMBS, result);
}

extern "C" __global__ void hf_bind_low_to_high_kernel(const u64 *__restrict__ column,
                                                     unsigned int kind,
                                                     unsigned int entry_stride,
                                                     unsigned int entry_offset,
                                                     unsigned int entry_bit,
                                                     u64 c0, u64 c1, u64 c2, u64 c3,
                                                     u64 *__restrict__ out,
                                                     unsigned int half) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= half) return;
    u64 lo[LIMBS], hi[LIMBS], c[LIMBS], d[LIMBS], t[LIMBS], r[LIMBS];
    unsigned long long pair = (unsigned long long)i * 2;
    hf_load_value(column, kind, entry_stride, entry_offset, entry_bit, pair, lo);
    hf_load_value(column, kind, entry_stride, entry_offset, entry_bit, pair + 1, hi);
    c[0] = c0;
    c[1] = c1;
    c[2] = c2;
    c[3] = c3;
    fr_sub(hi, lo, d);
    fr_mul(c, d, t);
    fr_add(lo, t, r);
    store4(out + i * LIMBS, r);
}
