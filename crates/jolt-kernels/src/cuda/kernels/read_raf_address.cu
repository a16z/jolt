__device__ __forceinline__ void irr_extension(const u64 *__restrict__ evals,
                                              unsigned int b,
                                              unsigned int half,
                                              unsigned int c,
                                              u64 *out) {
    u64 lo[LIMBS], hi[LIMBS];
    load4(evals + (unsigned long long)b * LIMBS, lo);
    load4(evals + ((unsigned long long)b + (unsigned long long)half) * LIMBS, hi);
    if (c == 0) { store4(out, lo); return; }
    if (c == 1) { store4(out, hi); return; }
    u64 doubled[LIMBS];
    fr_add(hi, hi, doubled);
    fr_sub(doubled, lo, out);
}

extern "C" __global__ void irr_address_message_kernel(
    const u64 *const *__restrict__ prefix_tables,
    const unsigned int *__restrict__ prefix_ids,
    const unsigned int *__restrict__ suffix_slots,
    const unsigned int *__restrict__ scales,
    const unsigned int *__restrict__ term_offsets,
    const unsigned int *__restrict__ term_counts,
    const u64 *const *__restrict__ suffix_tables,
    const unsigned int *__restrict__ suffix_bases,
    unsigned int table_count,
    const u64 *const *__restrict__ raf_tables,
    unsigned int raf_count,
    unsigned int half,
    u64 *__restrict__ partials) {
    extern __shared__ u64 scratch[];
    unsigned int tid = threadIdx.x;
    unsigned int b = blockIdx.x * blockDim.x + tid;
    unsigned int lanes = 3u * (1u + raf_count);

    for (unsigned int lane = 0; lane < lanes; lane++) {
        unsigned int c = lane / (1u + raf_count);
        unsigned int slot = lane % (1u + raf_count);
        u64 acc[LIMBS] = {0, 0, 0, 0};

        if (b < half) {
            if (slot == 0) {
                for (unsigned int t = 0; t < table_count; t++) {
                    unsigned int count = term_counts[t];
                    unsigned int base = term_offsets[t];
                    u64 sum[LIMBS] = {0, 0, 0, 0};
                    for (unsigned int k = 0; k < count; k++) {
                        unsigned int term = base + k;
                        u64 value[LIMBS];
                        irr_extension(suffix_tables[suffix_bases[t] + suffix_slots[term]],
                                      b, half, c, value);
                        unsigned int prefix = prefix_ids[term];
                        if (prefix != 0xFFFFFFFFu) {
                            u64 p[LIMBS], product[LIMBS];
                            irr_extension(prefix_tables[prefix], b, half, c, p);
                            fr_mul(value, p, product);
                            store4(value, product);
                        }
                        if (scales[term] != 0u) {
                            u64 s[LIMBS], scaled[LIMBS];
                            cmb_scale(scales[term], s);
                            fr_mul(value, s, scaled);
                            store4(value, scaled);
                        }
                        u64 next[LIMBS];
                        fr_add(sum, value, next);
                        store4(sum, next);
                    }
                    u64 total[LIMBS];
                    fr_add(acc, sum, total);
                    store4(acc, total);
                }
            } else {
                unsigned int r = (slot - 1u) * 3u;
                u64 prefix[LIMBS], shift[LIMBS], value[LIMBS], product[LIMBS], sum[LIMBS];
                irr_extension(raf_tables[r], b, half, c, prefix);
                irr_extension(raf_tables[r + 1], b, half, c, shift);
                irr_extension(raf_tables[r + 2], b, half, c, value);
                fr_mul(prefix, shift, product);
                fr_add(product, value, sum);
                store4(acc, sum);
            }
        }

        store4(scratch + tid * LIMBS, acc);
        __syncthreads();
        for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                u64 x[LIMBS], y[LIMBS], sum[LIMBS];
                load4(scratch + tid * LIMBS, x);
                load4(scratch + (tid + stride) * LIMBS, y);
                fr_add(x, y, sum);
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
