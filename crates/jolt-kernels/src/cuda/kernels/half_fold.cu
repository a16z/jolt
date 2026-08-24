#define HF_KIND_U64 0u
#define HF_KIND_U128 1u
#define HF_KIND_TWOS_I128 2u
#define HF_KIND_FIELD 3u

__device__ __forceinline__ void hf_promote_narrow(const u64 *__restrict__ entry,
                                                 unsigned int kind,
                                                 u64 *out) {
    u64 lo = entry[0];
    u64 hi = kind == HF_KIND_U64 ? 0ull : entry[1];
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
                                              unsigned long long index,
                                              u64 *out) {
    if (kind == HF_KIND_FIELD) {
        load4(words + index * LIMBS, out);
        return;
    }
    hf_promote_narrow(words + (unsigned long long)offset + index * (unsigned long long)stride,
                      kind, out);
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
                                              unsigned int entry_offset) {
    unsigned int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= out_len) return;

    u64 acc[LIMBS] = {0, 0, 0, 0};
    unsigned long long base = (unsigned long long)a * (unsigned long long)out_stride;
    for (unsigned int b = 0; b < sum_len; b++) {
        u64 weight[LIMBS], value[LIMBS], term[LIMBS], sum[LIMBS];
        load4(weights + (unsigned long long)b * LIMBS, weight);
        unsigned long long index = base + (unsigned long long)b * (unsigned long long)sum_stride;
        hf_load_value(column, kind, entry_stride, entry_offset, index, value);
        fr_mul(weight, value, term);
        fr_add(acc, term, sum);
        for (int l = 0; l < LIMBS; l++) acc[l] = sum[l];
    }

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
                                             unsigned int entry_offset) {
    extern __shared__ u64 scratch[];
    unsigned int a = blockIdx.x;
    unsigned int tid = threadIdx.x;

    u64 acc[LIMBS] = {0, 0, 0, 0};
    unsigned long long base = (unsigned long long)a * (unsigned long long)sum_len;
    for (unsigned int b = tid; b < sum_len; b += blockDim.x) {
        u64 weight[LIMBS], value[LIMBS], term[LIMBS], sum[LIMBS];
        load4(weights + (unsigned long long)b * LIMBS, weight);
        hf_load_value(column, kind, entry_stride, entry_offset, base + (unsigned long long)b, value);
        fr_mul(weight, value, term);
        fr_add(acc, term, sum);
        for (int l = 0; l < LIMBS; l++) acc[l] = sum[l];
    }

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
