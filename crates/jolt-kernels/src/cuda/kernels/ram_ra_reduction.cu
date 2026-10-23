extern "C" __global__ void ram_ra_gather_h_kernel(const unsigned int *__restrict__ indices,
                                                 const u64 *__restrict__ eq_address,
                                                 u64 *__restrict__ h,
                                                 unsigned int cycles) {
    unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= cycles) return;
    unsigned int address = indices[c];
    if (address == RA_COLD) {
        u64 zero[LIMBS] = {0, 0, 0, 0};
        store4(h + (unsigned long long)c * LIMBS, zero);
        return;
    }
    u64 value[LIMBS];
    load4(eq_address + (unsigned long long)address * LIMBS, value);
    store4(h + (unsigned long long)c * LIMBS, value);
}

extern "C" __global__ void ram_ra_fold_suffix_kernel(const u64 *__restrict__ h,
                                                     const u64 *__restrict__ eq_hi,
                                                     u64 *__restrict__ q,
                                                     unsigned int prefix_size,
                                                     unsigned int suffix_size) {
    unsigned int c_lo = blockIdx.x * blockDim.x + threadIdx.x;
    if (c_lo >= prefix_size) return;
    u64 total[LIMBS] = {0, 0, 0, 0};
    for (unsigned int c_hi = 0; c_hi < suffix_size; c_hi++) {
        u64 value[LIMBS], weight[LIMBS], term[LIMBS];
        load4(h + ((unsigned long long)c_hi * (unsigned long long)prefix_size +
                   (unsigned long long)c_lo) *
                      LIMBS,
              value);
        load4(eq_hi + (unsigned long long)c_hi * LIMBS, weight);
        fr_mul(value, weight, term);
        fr_add(total, term, total);
    }
    store4(q + (unsigned long long)c_lo * LIMBS, total);
}

extern "C" __global__ void ram_ra_fold_prefix_kernel(const u64 *__restrict__ h,
                                                     const u64 *__restrict__ eq_prefix,
                                                     u64 *__restrict__ h_prime,
                                                     unsigned int prefix_size,
                                                     unsigned int suffix_size) {
    unsigned int c_hi = blockIdx.x * blockDim.x + threadIdx.x;
    if (c_hi >= suffix_size) return;
    u64 total[LIMBS] = {0, 0, 0, 0};
    for (unsigned int c_lo = 0; c_lo < prefix_size; c_lo++) {
        u64 value[LIMBS], weight[LIMBS], term[LIMBS];
        load4(h + ((unsigned long long)c_hi * (unsigned long long)prefix_size +
                   (unsigned long long)c_lo) *
                      LIMBS,
              value);
        load4(eq_prefix + (unsigned long long)c_lo * LIMBS, weight);
        fr_mul(value, weight, term);
        fr_add(total, term, total);
    }
    store4(h_prime + (unsigned long long)c_hi * LIMBS, total);
}

extern "C" __global__ void ram_ra_phase1_round_kernel(const u64 *const *__restrict__ p_tables,
                                                      const u64 *const *__restrict__ q_tables,
                                                      const u64 *__restrict__ coefficients,
                                                      unsigned int terms,
                                                      unsigned int half,
                                                      unsigned int lanes,
                                                      u64 *__restrict__ partials) {
    extern __shared__ u64 scratch[];
    unsigned int tid = threadIdx.x;
    unsigned int y = blockIdx.x * blockDim.x + tid;

    for (unsigned int c = 0; c < lanes; c++) {
        u64 acc[LIMBS] = {0, 0, 0, 0};
        if (y < half) {
            u64 point[LIMBS];
            u64 raw[LIMBS] = {c, 0, 0, 0};
            fr_to_mont(raw, point);

            for (unsigned int t = 0; t < terms; t++) {
                u64 p_lo[LIMBS], p_hi[LIMBS], q_lo[LIMBS], q_hi[LIMBS];
                load4(p_tables[t] + (2 * y) * LIMBS, p_lo);
                load4(p_tables[t] + (2 * y + 1) * LIMBS, p_hi);
                load4(q_tables[t] + (2 * y) * LIMBS, q_lo);
                load4(q_tables[t] + (2 * y + 1) * LIMBS, q_hi);

                u64 p_diff[LIMBS], p_scaled[LIMBS], p_at[LIMBS];
                fr_sub(p_hi, p_lo, p_diff);
                fr_mul(point, p_diff, p_scaled);
                fr_add(p_lo, p_scaled, p_at);

                u64 q_diff[LIMBS], q_scaled[LIMBS], q_at[LIMBS];
                fr_sub(q_hi, q_lo, q_diff);
                fr_mul(point, q_diff, q_scaled);
                fr_add(q_lo, q_scaled, q_at);

                u64 product[LIMBS], weight[LIMBS], term[LIMBS];
                fr_mul(p_at, q_at, product);
                load4(coefficients + (unsigned long long)t * LIMBS, weight);
                fr_mul(product, weight, term);
                fr_add(acc, term, acc);
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
