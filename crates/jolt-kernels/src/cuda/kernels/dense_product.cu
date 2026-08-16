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

extern "C" __global__ void sopg_round_kernel(
    const u64 *const *__restrict__ tables, const unsigned int *__restrict__ term_offsets,
    const unsigned int *__restrict__ term_factors, const u64 *__restrict__ coefficients,
    unsigned int terms, unsigned int half, const u64 *__restrict__ e_in,
    const u64 *__restrict__ e_out, unsigned int e_in_len, unsigned int in_bits,
    u64 *__restrict__ partials) {
    extern __shared__ u64 scratch[];
    unsigned int tid = threadIdx.x;
    unsigned int y = blockIdx.x * blockDim.x + tid;

    u64 acc[2][LIMBS];
    for (int lane = 0; lane < 2; lane++)
        for (int l = 0; l < LIMBS; l++) acc[lane][l] = 0;

    if (y < half) {
        u64 sum_constant[LIMBS] = {0, 0, 0, 0};
        u64 sum_leading[LIMBS] = {0, 0, 0, 0};

        for (unsigned int t = 0; t < terms; t++) {
            u64 constant[LIMBS], leading[LIMBS];
            load4(FR_ONE, constant);
            load4(FR_ONE, leading);
            for (unsigned int f = term_offsets[t]; f < term_offsets[t + 1]; f++) {
                const u64 *table = tables[term_factors[f]];
                u64 lo[LIMBS], hi[LIMBS], diff[LIMBS], next[LIMBS];
                load4(table + (2 * y) * LIMBS, lo);
                load4(table + (2 * y + 1) * LIMBS, hi);
                fr_sub(hi, lo, diff);
                fr_mul(constant, lo, next);
                for (int l = 0; l < LIMBS; l++) constant[l] = next[l];
                fr_mul(leading, diff, next);
                for (int l = 0; l < LIMBS; l++) leading[l] = next[l];
            }
            u64 weighted[LIMBS], sum[LIMBS];
            fr_mul(constant, coefficients + t * LIMBS, weighted);
            fr_add(sum_constant, weighted, sum);
            for (int l = 0; l < LIMBS; l++) sum_constant[l] = sum[l];
            fr_mul(leading, coefficients + t * LIMBS, weighted);
            fr_add(sum_leading, weighted, sum);
            for (int l = 0; l < LIMBS; l++) sum_leading[l] = sum[l];
        }

        u64 weight[LIMBS];
        if (e_in_len <= 1) {
            load4(e_out + (unsigned long long)y * LIMBS, weight);
        } else {
            u64 inner[LIMBS], outer[LIMBS];
            load4(e_in + (unsigned long long)(y & ((1u << in_bits) - 1u)) * LIMBS, inner);
            load4(e_out + (unsigned long long)(y >> in_bits) * LIMBS, outer);
            fr_mul(inner, outer, weight);
        }
        fr_mul(sum_constant, weight, acc[0]);
        fr_mul(sum_leading, weight, acc[1]);
    }

    for (unsigned int lane = 0; lane < 2; lane++) {
        store4(scratch + tid * LIMBS, acc[lane]);
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
            store4(partials + ((unsigned long long)lane * gridDim.x + blockIdx.x) * LIMBS, total);
        }
        __syncthreads();
    }
}

extern "C" __global__ void sop_round_kernel(
    const u64 *const *__restrict__ tables, const unsigned int *__restrict__ term_offsets,
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
                    const u64 *table = tables[term_factors[f]];
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
