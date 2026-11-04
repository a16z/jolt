#define CR_MAX_FACTORS 9

extern "C" __global__ void cr_quotient_kernel(
    const u64 *const *__restrict__ factors, unsigned int factor_count,
    const u64 *__restrict__ e_in, unsigned int in_len,
    const u64 *__restrict__ e_out, unsigned int out_len,
    unsigned int in_bits, unsigned int prefix, unsigned int half,
    u64 tail0, u64 tail1, u64 tail2, u64 tail3, u64 *__restrict__ partials) {
    extern __shared__ u64 scratch[];
    unsigned int tid = threadIdx.x;
    unsigned int y = blockIdx.x * blockDim.x + tid;
    (void)out_len;

    u64 acc[CR_MAX_FACTORS + 1][LIMBS];
#pragma unroll
    for (int lane = 0; lane <= CR_MAX_FACTORS; lane++) {
        load4(FR_ONE, acc[lane]);
    }

    if (y < half) {
        u64 weight[LIMBS], e_i[LIMBS], e_o[LIMBS];
        if (2 * y < prefix) {
            load4(e_in + (unsigned long long)(y & (in_len - 1u)) * LIMBS, e_i);
            load4(e_out + (unsigned long long)(y >> in_bits) * LIMBS, e_o);
            fr_mul(e_i, e_o, weight);
        } else {
            weight[0] = tail0;
            weight[1] = tail1;
            weight[2] = tail2;
            weight[3] = tail3;
        }

        for (unsigned int t = 0; t < factor_count; t++) {
            const u64 *table = factors[t];
            u64 lo[LIMBS], hi[LIMBS], diff[LIMBS], cur[LIMBS];
            load4(table + (unsigned long long)min(2 * y, prefix) * LIMBS, lo);
            load4(table + (unsigned long long)min(2 * y + 1, prefix) * LIMBS, hi);
            if (t == 0) {
                u64 scaled[LIMBS];
                fr_mul(lo, weight, scaled);
                store4(lo, scaled);
                fr_mul(hi, weight, scaled);
                store4(hi, scaled);
            }
            fr_sub(hi, lo, diff);
            store4(cur, hi);

#pragma unroll
            for (int lane = 0; lane <= CR_MAX_FACTORS; lane++) {
                if ((unsigned int)lane > factor_count) continue;
                u64 m[LIMBS], product[LIMBS];
                if ((unsigned int)lane == factor_count) {
                    store4(m, lo);
                } else if ((unsigned int)lane + 1u == factor_count) {
                    store4(m, diff);
                } else {
                    store4(m, cur);
                    u64 next[LIMBS];
                    fr_add(cur, diff, next);
                    store4(cur, next);
                }
                fr_mul(acc[lane], m, product);
                store4(acc[lane], product);
            }
        }
    }

#pragma unroll
    for (int lane = 0; lane <= CR_MAX_FACTORS; lane++) {
        if ((unsigned int)lane > factor_count) continue;
        u64 value[LIMBS] = {0, 0, 0, 0};
        if (y < half) {
#pragma unroll
            for (int l = 0; l < LIMBS; l++) value[l] = acc[lane][l];
        }
        store4(scratch + tid * LIMBS, value);
        __syncthreads();

        for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                u64 a[LIMBS], b[LIMBS], sum[LIMBS];
                load4(scratch + tid * LIMBS, a);
                load4(scratch + (tid + stride) * LIMBS, b);
                fr_add(a, b, sum);
                store4(scratch + tid * LIMBS, sum);
            }
            __syncthreads();
        }

        if (tid == 0) {
            u64 total[LIMBS];
            load4(scratch, total);
            store4(partials + (unsigned long long)((unsigned int)lane * gridDim.x + blockIdx.x) * LIMBS,
                   total);
        }
        __syncthreads();
    }
}

extern "C" __global__ void cr_bind_prefix_kernel(
    const u64 *__restrict__ values, unsigned int prefix, unsigned int count,
    u64 r0, u64 r1, u64 r2, u64 r3, u64 *__restrict__ output) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    u64 lo[LIMBS], hi[LIMBS], diff[LIMBS], scaled[LIMBS], result[LIMBS];
    u64 r[LIMBS] = {r0, r1, r2, r3};
    load4(values + (unsigned long long)min(2 * i, prefix) * LIMBS, lo);
    load4(values + (unsigned long long)min(2 * i + 1, prefix) * LIMBS, hi);
    fr_sub(hi, lo, diff);
    fr_mul(diff, r, scaled);
    fr_add(lo, scaled, result);
    store4(output + (unsigned long long)i * LIMBS, result);
}

extern "C" __global__ void cr_eq_prefix_kernel(
    const u64 *__restrict__ e_in, const u64 *__restrict__ e_out,
    unsigned int in_bits, unsigned int start, unsigned int count,
    u64 tail0, u64 tail1, u64 tail2, u64 tail3, u64 *__restrict__ output) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    u64 value[LIMBS];
    if (i + 1 == count) {
        value[0] = tail0;
        value[1] = tail1;
        value[2] = tail2;
        value[3] = tail3;
    } else {
        unsigned int j = start + i;
        u64 lo[LIMBS], hi[LIMBS];
        load4(e_in + (unsigned long long)(j & ((1u << in_bits) - 1)) * LIMBS, lo);
        load4(e_out + (unsigned long long)(j >> in_bits) * LIMBS, hi);
        fr_mul(lo, hi, value);
    }
    store4(output + (unsigned long long)i * LIMBS, value);
}
