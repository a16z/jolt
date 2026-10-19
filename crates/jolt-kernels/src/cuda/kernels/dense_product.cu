extern "C" __global__ void lane_sum_total_kernel(const u64 *__restrict__ in,
                                                 u64 *__restrict__ out, unsigned int width) {
    extern __shared__ u64 scratch[];
    unsigned int lane = blockIdx.x;
    unsigned int tid = threadIdx.x;

    u64 acc[LIMBS];
    for (int l = 0; l < LIMBS; l++) acc[l] = 0;
    for (unsigned int i = tid; i < width; i += blockDim.x) {
        u64 value[LIMBS], sum[LIMBS];
        load4(in + ((unsigned long long)lane * width + i) * LIMBS, value);
        fr_add(acc, value, sum);
        for (int l = 0; l < LIMBS; l++) acc[l] = sum[l];
    }

    store4(scratch + tid * LIMBS, acc);
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
        store4(out + (unsigned long long)lane * LIMBS, total);
    }
}

extern "C" __global__ void sop_round_kernel(
    const u64 *const *__restrict__ tables, const unsigned int *__restrict__ descriptors,
    const unsigned int *__restrict__ term_offsets,
    const unsigned int *__restrict__ term_factors, const u64 *__restrict__ coefficients,
    unsigned int terms, unsigned int half, unsigned int lanes, u64 *__restrict__ partials,
    unsigned int first_point, unsigned int infinity_lane) {
    extern __shared__ u64 scratch[];
    unsigned int tid = threadIdx.x;
    unsigned int y = blockIdx.x * blockDim.x + tid;

    for (unsigned int c = 0; c < lanes; c++) {
        u64 acc[LIMBS] = {0, 0, 0, 0};
        bool at_infinity = infinity_lane && (c + 1 == lanes);
        if (y < half) {
            u64 point[LIMBS];
            u64 raw[LIMBS] = {first_point + c, 0, 0, 0};
            fr_to_mont(raw, point);

            for (unsigned int t = 0; t < terms; t++) {
                u64 product[LIMBS];
                load4(FR_ONE, product);
                for (unsigned int f = term_offsets[t]; f < term_offsets[t + 1]; f++) {
                    unsigned int slot = term_factors[f];
                    const unsigned int *entry = descriptors + slot * 4u;
                    u64 lo[LIMBS], hi[LIMBS], diff[LIMBS], scaled[LIMBS], value[LIMBS];
                    unsigned long long pair = (unsigned long long)y * 2;
                    hf_load_value(tables[slot], entry[0], entry[1], entry[2], entry[3], pair, lo);
                    hf_load_value(tables[slot], entry[0], entry[1], entry[2], entry[3], pair + 1, hi);
                    fr_sub(hi, lo, diff);
                    if (at_infinity) {
                        for (int l = 0; l < LIMBS; l++) value[l] = diff[l];
                    } else {
                        fr_mul(point, diff, scaled);
                        fr_add(lo, scaled, value);
                    }
                    u64 next[LIMBS];
                    fr_mul(product, value, next);
                    for (int l = 0; l < LIMBS; l++) product[l] = next[l];
                }
                u64 weighted[LIMBS], sum[LIMBS];
                fr_mul(product, coefficients + t * LIMBS, weighted);
                fr_add(acc, weighted, sum);
                for (int l = 0; l < LIMBS; l++) acc[l] = sum[l];
            }
        }
        store4(scratch + tid * LIMBS, acc);
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
            store4(partials + (c * gridDim.x + blockIdx.x) * LIMBS, total);
        }
        __syncthreads();
    }
}
