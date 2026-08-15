extern "C" __global__ void dense_product_round_kernel(
    const u64 *const *__restrict__ tables, unsigned int table_count,
    unsigned int half, unsigned int lanes, u64 *__restrict__ partials,
    const u64 *__restrict__ lt_lo, const u64 *__restrict__ lt_hi,
    const u64 *__restrict__ eq_hi, const u64 *__restrict__ lt_shift,
    unsigned int lt_lo_bits, unsigned int lt_lo_mask, unsigned int has_lt,
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

            u64 product[LIMBS];
            load4(FR_ONE, product);
            for (unsigned int t = 0; t < table_count; t++) {
                const u64 *table = tables[t];
                u64 lo[LIMBS], hi[LIMBS], diff[LIMBS], scaled[LIMBS], value[LIMBS];
                load4(table + (2 * y) * LIMBS, lo);
                load4(table + (2 * y + 1) * LIMBS, hi);
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
            if (has_lt) {
                u64 lo[LIMBS], hi[LIMBS], diff[LIMBS], scaled[LIMBS], value[LIMBS];
                lt_split_get(lt_lo, lt_hi, eq_hi, lt_shift, lt_lo_bits, lt_lo_mask, 2 * y, lo);
                lt_split_get(lt_lo, lt_hi, eq_hi, lt_shift, lt_lo_bits, lt_lo_mask, 2 * y + 1, hi);
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
            for (int l = 0; l < LIMBS; l++) acc[l] = product[l];
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

extern "C" __global__ void lane_sum_reduce_kernel(const u64 *__restrict__ in,
                                                 u64 *__restrict__ out,
                                                 unsigned int lanes,
                                                 unsigned int in_width,
                                                 unsigned int out_width) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int lane = blockIdx.y;
    if (i >= out_width || lane >= lanes) return;
    u64 acc[LIMBS];
    load4(in + (lane * in_width + i) * LIMBS, acc);
    unsigned int mate = i + out_width;
    if (mate < in_width) {
        u64 other[LIMBS], sum[LIMBS];
        load4(in + (lane * in_width + mate) * LIMBS, other);
        fr_add(acc, other, sum);
        for (int l = 0; l < LIMBS; l++) acc[l] = sum[l];
    }
    store4(out + (lane * out_width + i) * LIMBS, acc);
}

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

extern "C" __global__ void weighted_combine_kernel(const u64 *__restrict__ weights,
                                                  const u64 *__restrict__ coefficient,
                                                  u64 *__restrict__ accumulator,
                                                  unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    u64 w[LIMBS], c[LIMBS], scaled[LIMBS], acc[LIMBS], sum[LIMBS];
    load4(weights + i * LIMBS, w);
    load4(coefficient, c);
    fr_mul(w, c, scaled);
    load4(accumulator + i * LIMBS, acc);
    fr_add(acc, scaled, sum);
    store4(accumulator + i * LIMBS, sum);
}
