extern "C" __global__ void hf_half_fold_kernel(const u64 *__restrict__ column,
                                              const u64 *__restrict__ weights,
                                              u64 *__restrict__ out,
                                              u64 scale0, u64 scale1, u64 scale2, u64 scale3,
                                              u64 bias0, u64 bias1, u64 bias2, u64 bias3,
                                              unsigned int out_len,
                                              unsigned int sum_len,
                                              unsigned int out_stride,
                                              unsigned int sum_stride,
                                              unsigned int accumulate) {
    unsigned int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= out_len) return;

    u64 acc[LIMBS] = {0, 0, 0, 0};
    unsigned long long base = (unsigned long long)a * (unsigned long long)out_stride;
    for (unsigned int b = 0; b < sum_len; b++) {
        u64 weight[LIMBS], value[LIMBS], term[LIMBS], sum[LIMBS];
        load4(weights + (unsigned long long)b * LIMBS, weight);
        unsigned long long index = base + (unsigned long long)b * (unsigned long long)sum_stride;
        load4(column + index * LIMBS, value);
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
                                             unsigned int accumulate) {
    extern __shared__ u64 scratch[];
    unsigned int a = blockIdx.x;
    unsigned int tid = threadIdx.x;

    u64 acc[LIMBS] = {0, 0, 0, 0};
    unsigned long long base = (unsigned long long)a * (unsigned long long)sum_len;
    for (unsigned int b = tid; b < sum_len; b += blockDim.x) {
        u64 weight[LIMBS], value[LIMBS], term[LIMBS], sum[LIMBS];
        load4(weights + (unsigned long long)b * LIMBS, weight);
        load4(column + (base + (unsigned long long)b) * LIMBS, value);
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
