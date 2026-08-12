#define CR_MAX_FACTORS 9

extern "C" __global__ void cr_quotient_kernel(
    const u64 *const *__restrict__ factors, unsigned int factor_count,
    const u64 *__restrict__ e_in, unsigned int in_len,
    const u64 *__restrict__ e_out, unsigned int out_len,
    unsigned int in_bits, unsigned int half, u64 *__restrict__ partials) {
    extern __shared__ u64 scratch[];
    unsigned int tid = threadIdx.x;
    unsigned int y = blockIdx.x * blockDim.x + tid;
    (void)out_len;

    u64 acc[CR_MAX_FACTORS][LIMBS];
#pragma unroll
    for (int lane = 0; lane < CR_MAX_FACTORS; lane++) {
        load4(FR_ONE, acc[lane]);
    }

    if (y < half) {
        u64 weight[LIMBS], e_i[LIMBS], e_o[LIMBS];
        load4(e_in + (unsigned long long)(y & (in_len - 1u)) * LIMBS, e_i);
        load4(e_out + (unsigned long long)(y >> in_bits) * LIMBS, e_o);
        fr_mul(e_i, e_o, weight);

        for (unsigned int t = 0; t < factor_count; t++) {
            const u64 *table = factors[t];
            u64 lo[LIMBS], hi[LIMBS], diff[LIMBS], cur[LIMBS];
            load4(table + (unsigned long long)(2 * y) * LIMBS, lo);
            load4(table + (unsigned long long)(2 * y + 1) * LIMBS, hi);
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
            for (int lane = 0; lane < CR_MAX_FACTORS; lane++) {
                if ((unsigned int)lane >= factor_count) continue;
                u64 m[LIMBS], product[LIMBS];
                if ((unsigned int)lane + 1u == factor_count) {
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
    for (int lane = 0; lane < CR_MAX_FACTORS; lane++) {
        if ((unsigned int)lane >= factor_count) continue;
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
