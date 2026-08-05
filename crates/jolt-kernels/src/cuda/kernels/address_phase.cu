#define AP_CHUNK_LEN 8
#define AP_CHUNK_SIZE 256
#define AP_RAF_LANES 6
#define AP_MAX_SUFFIXES 4
#define AP_NO_TABLE 0xFFFFFFFFu
#define AP_SKIP 0xFFFFFFFFu

__device__ __forceinline__ u128 ap_index(const unsigned long long *__restrict__ bits,
                                         unsigned int j) {
    return ((u128)bits[2 * j + 1] << 64) | (u128)bits[2 * j];
}

__device__ __forceinline__ unsigned int ap_chunk(u128 index, unsigned int suffix_len) {
    return (unsigned int)((index >> suffix_len) & (u128)(AP_CHUNK_SIZE - 1));
}

extern "C" __global__ void ap_raf_keys_kernel(const unsigned long long *__restrict__ lookup_index,
                                              unsigned int suffix_len,
                                              unsigned int *__restrict__ keys,
                                              unsigned int n) {
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    keys[j] = ap_chunk(ap_index(lookup_index, j), suffix_len);
}

extern "C" __global__ void ap_table_keys_kernel(const unsigned long long *__restrict__ lookup_index,
                                                const unsigned int *__restrict__ table_index,
                                                const unsigned int *__restrict__ table_slots,
                                                unsigned int table_count,
                                                unsigned int suffix_len,
                                                unsigned int *__restrict__ keys,
                                                unsigned int n) {
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    unsigned int table = table_index[j];
    if (table == AP_NO_TABLE || table >= table_count) {
        keys[j] = AP_SKIP;
        return;
    }
    unsigned int slot = table_slots[table];
    if (slot == AP_SKIP) {
        keys[j] = AP_SKIP;
        return;
    }
    keys[j] = slot * AP_CHUNK_SIZE + ap_chunk(ap_index(lookup_index, j), suffix_len);
}

extern "C" __global__ void ap_histogram_kernel(const unsigned int *__restrict__ keys,
                                               unsigned int *__restrict__ counts,
                                               unsigned int n) {
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    unsigned int key = keys[j];
    if (key == AP_SKIP) return;
    atomicAdd(&counts[key], 1u);
}

extern "C" __global__ void ap_scatter_kernel(const unsigned int *__restrict__ keys,
                                             const unsigned int *__restrict__ offsets,
                                             unsigned int *__restrict__ cursors,
                                             unsigned int *__restrict__ order,
                                             unsigned int n) {
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    unsigned int key = keys[j];
    if (key == AP_SKIP) return;
    unsigned int slot = atomicAdd(&cursors[key], 1u);
    order[offsets[key] + slot] = j;
}

__device__ __forceinline__ void ap_block_reduce(u64 *scratch, u64 *acc) {
    unsigned int tid = threadIdx.x;
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
}

extern "C" __global__ void ap_raf_reduce_kernel(const unsigned int *__restrict__ order,
                                                const unsigned int *__restrict__ offsets,
                                                const unsigned int *__restrict__ counts,
                                                const unsigned long long *__restrict__ lookup_index,
                                                const unsigned char *__restrict__ raf_flags,
                                                const u64 *__restrict__ u_evals,
                                                unsigned int suffix_len,
                                                unsigned int upper_suffix_bits,
                                                unsigned int canonical,
                                                u64 *__restrict__ buckets) {
    extern __shared__ u64 scratch[];
    unsigned int bucket = blockIdx.x;
    unsigned int start = offsets[bucket];
    unsigned int count = counts[bucket];

    u64 acc[AP_RAF_LANES][LIMBS];
    for (int lane = 0; lane < AP_RAF_LANES; lane++) {
        for (int l = 0; l < LIMBS; l++) acc[lane][l] = 0;
    }

    for (unsigned int i = threadIdx.x; i < count; i += blockDim.x) {
        unsigned int j = order[start + i];
        u64 u[LIMBS];
        load4(u_evals + (unsigned long long)j * LIMBS, u);
        u128 index = ap_index(lookup_index, j);
        u128 suffix_bits = index & sfx_mask(suffix_len);

        if (canonical != 0u && raf_flags[j] != 0u) {
            u128 ones = ((u128)1 << upper_suffix_bits) - 1;
            if (upper_suffix_bits == 0u ||
                (suffix_bits >> (suffix_len - upper_suffix_bits)) == ones) {
                fr_add(acc[5], u, acc[5]);
            }
        }

        if (raf_flags[j] == 0u) {
            fr_add(acc[0], u, acc[0]);
            sfx_bits whole = sfx_new(suffix_bits, suffix_len);
            sfx_bits left, right;
            sfx_uninterleave(whole, &left, &right);
            unsigned long long left_value = sfx_u64(left);
            if (left_value != 0ULL) {
                u64 mont[LIMBS], term[LIMBS];
                u64 raw[LIMBS] = {left_value, 0, 0, 0};
                fr_to_mont(raw, mont);
                fr_mul(u, mont, term);
                fr_add(acc[2], term, acc[2]);
            }
            unsigned long long right_value = sfx_u64(right);
            if (right_value != 0ULL) {
                u64 mont[LIMBS], term[LIMBS];
                u64 raw[LIMBS] = {right_value, 0, 0, 0};
                fr_to_mont(raw, mont);
                fr_mul(u, mont, term);
                fr_add(acc[3], term, acc[3]);
            }
        } else {
            fr_add(acc[1], u, acc[1]);
            if (suffix_bits != 0) {
                u64 mont[LIMBS], term[LIMBS];
                u64 raw[LIMBS] = {(unsigned long long)suffix_bits,
                                  (unsigned long long)(suffix_bits >> 64), 0, 0};
                fr_to_mont(raw, mont);
                fr_mul(u, mont, term);
                fr_add(acc[4], term, acc[4]);
            }
        }
    }

    for (int lane = 0; lane < AP_RAF_LANES; lane++) {
        ap_block_reduce(scratch, acc[lane]);
        if (threadIdx.x == 0) {
            u64 total[LIMBS];
            load4(scratch, total);
            store4(buckets + ((unsigned long long)lane * AP_CHUNK_SIZE + bucket) * LIMBS, total);
        }
        __syncthreads();
    }
}

extern "C" __global__ void ap_suffix_reduce_kernel(
    const unsigned int *__restrict__ order,
    const unsigned int *__restrict__ offsets,
    const unsigned int *__restrict__ counts,
    const unsigned long long *__restrict__ lookup_index,
    const u64 *__restrict__ u_evals,
    const unsigned int *__restrict__ suffix_ids,
    const unsigned int *__restrict__ suffix_offsets,
    const unsigned int *__restrict__ suffix_counts,
    unsigned int suffix_len,
    u64 *__restrict__ buckets) {
    extern __shared__ u64 scratch[];
    unsigned int slot = blockIdx.x / AP_CHUNK_SIZE;
    unsigned int bucket = blockIdx.x % AP_CHUNK_SIZE;
    unsigned int start = offsets[blockIdx.x];
    unsigned int count = counts[blockIdx.x];
    unsigned int families = suffix_counts[slot];
    unsigned int family_base = suffix_offsets[slot];

    u64 acc[AP_MAX_SUFFIXES][LIMBS];
    for (int s = 0; s < AP_MAX_SUFFIXES; s++) {
        for (int l = 0; l < LIMBS; l++) acc[s][l] = 0;
    }

    for (unsigned int i = threadIdx.x; i < count; i += blockDim.x) {
        unsigned int j = order[start + i];
        u64 u[LIMBS];
        load4(u_evals + (unsigned long long)j * LIMBS, u);
        u128 index = ap_index(lookup_index, j);
        sfx_bits suffix = sfx_new(index & sfx_mask(suffix_len), suffix_len);

        for (unsigned int s = 0; s < families; s++) {
            unsigned long long value = sfx_eval(suffix_ids[family_base + s], suffix);
            if (value == 0ULL) continue;
            u64 term[LIMBS];
            if (value == 1ULL) {
                fr_add(acc[s], u, acc[s]);
                continue;
            }
            u64 mont[LIMBS];
            u64 raw[LIMBS] = {value, 0, 0, 0};
            fr_to_mont(raw, mont);
            fr_mul(u, mont, term);
            fr_add(acc[s], term, acc[s]);
        }
    }

    for (unsigned int s = 0; s < families; s++) {
        ap_block_reduce(scratch, acc[s]);
        if (threadIdx.x == 0) {
            u64 total[LIMBS];
            load4(scratch, total);
            store4(buckets + ((unsigned long long)(family_base + s) * AP_CHUNK_SIZE + bucket) * LIMBS,
                   total);
        }
        __syncthreads();
    }
}

extern "C" __global__ void ap_scale_shift_kernel(u64 *__restrict__ buckets,
                                                 const u64 *__restrict__ half_scale,
                                                 const u64 *__restrict__ full_scale) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= AP_CHUNK_SIZE) return;
    u64 scale[LIMBS], value[LIMBS], scaled[LIMBS];
    load4(half_scale, scale);
    load4(buckets + (unsigned long long)i * LIMBS, value);
    fr_mul(value, scale, scaled);
    store4(buckets + (unsigned long long)i * LIMBS, scaled);
    load4(full_scale, scale);
    load4(buckets + ((unsigned long long)AP_CHUNK_SIZE + i) * LIMBS, value);
    fr_mul(value, scale, scaled);
    store4(buckets + ((unsigned long long)AP_CHUNK_SIZE + i) * LIMBS, scaled);
}

extern "C" __global__ void ap_condense_kernel(const unsigned long long *__restrict__ lookup_index,
                                              u64 *__restrict__ u_evals,
                                              const u64 *__restrict__ v_prev,
                                              unsigned int suffix_len,
                                              unsigned int n) {
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    unsigned int chunk = ap_chunk(ap_index(lookup_index, j), suffix_len);
    u64 u[LIMBS], v[LIMBS], product[LIMBS];
    load4(u_evals + (unsigned long long)j * LIMBS, u);
    load4(v_prev + (unsigned long long)chunk * LIMBS, v);
    fr_mul(u, v, product);
    store4(u_evals + (unsigned long long)j * LIMBS, product);
}

__device__ __forceinline__ void ap_mul_pow2(const u64 *a, unsigned int k, u64 *out) {
    u64 value[LIMBS];
    load4(a, value);
    for (unsigned int i = 0; i < k; i++) {
        u64 doubled[LIMBS];
        fr_add(value, value, doubled);
        store4(value, doubled);
    }
    store4(out, value);
}

extern "C" __global__ void ap_prefix_tables_kernel(const u64 *__restrict__ checkpoints,
                                                   unsigned int suffix_len,
                                                   u64 *__restrict__ out,
                                                   unsigned int prefix_count) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= AP_CHUNK_SIZE) return;
    sfx_bits b = sfx_new((u128)x, AP_CHUNK_LEN);
    for (unsigned int prefix = 0; prefix < prefix_count; prefix++) {
        u64 value[LIMBS];
        pfx_eval(prefix, checkpoints, b, suffix_len, value);
        store4(out + ((unsigned long long)prefix * AP_CHUNK_SIZE + x) * LIMBS, value);
    }
}

extern "C" __global__ void ap_raf_prefix_kernel(const u64 *__restrict__ checkpoints,
                                                unsigned int chunk_upper_bits,
                                                unsigned int canonical,
                                                u64 *__restrict__ out) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= AP_CHUNK_SIZE) return;

    u64 left_cp[LIMBS], right_cp[LIMBS], identity_cp[LIMBS], upper_cp[LIMBS];
    load4(checkpoints, left_cp);
    load4(checkpoints + LIMBS, right_cp);
    load4(checkpoints + 2 * LIMBS, identity_cp);
    load4(checkpoints + 3 * LIMBS, upper_cp);

    sfx_bits chunk = sfx_new((u128)x, AP_CHUNK_LEN);
    sfx_bits left, right;
    sfx_uninterleave(chunk, &left, &right);

    u64 scaled[LIMBS], addend[LIMBS], value[LIMBS], raw[LIMBS];

    ap_mul_pow2(left_cp, AP_CHUNK_LEN / 2, scaled);
    raw[0] = sfx_u64(left); raw[1] = 0; raw[2] = 0; raw[3] = 0;
    fr_to_mont(raw, addend);
    fr_add(scaled, addend, value);
    store4(out + (unsigned long long)x * LIMBS, value);

    ap_mul_pow2(right_cp, AP_CHUNK_LEN / 2, scaled);
    raw[0] = sfx_u64(right); raw[1] = 0; raw[2] = 0; raw[3] = 0;
    fr_to_mont(raw, addend);
    fr_add(scaled, addend, value);
    store4(out + ((unsigned long long)AP_CHUNK_SIZE + x) * LIMBS, value);

    ap_mul_pow2(identity_cp, AP_CHUNK_LEN, scaled);
    raw[0] = (unsigned long long)x; raw[1] = 0; raw[2] = 0; raw[3] = 0;
    fr_to_mont(raw, addend);
    fr_add(scaled, addend, value);
    store4(out + ((unsigned long long)2 * AP_CHUNK_SIZE + x) * LIMBS, value);

    if (canonical != 0u) {
        if (chunk_upper_bits == 0u ||
            (x >> (AP_CHUNK_LEN - chunk_upper_bits)) == ((1u << chunk_upper_bits) - 1u)) {
            store4(out + ((unsigned long long)3 * AP_CHUNK_SIZE + x) * LIMBS, upper_cp);
        } else {
            u64 zero[LIMBS] = {0, 0, 0, 0};
            store4(out + ((unsigned long long)3 * AP_CHUNK_SIZE + x) * LIMBS, zero);
        }
    }
}

extern "C" __global__ void ap_bind_strided_kernel(const u64 *__restrict__ in,
                                                  const u64 *__restrict__ challenge,
                                                  u64 *__restrict__ out,
                                                  unsigned int half,
                                                  unsigned int stride,
                                                  unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned int column = i / half;
    unsigned int b = i % half;
    const u64 *base = in + (unsigned long long)column * stride * LIMBS;
    u64 lo[LIMBS], hi[LIMBS], c[LIMBS], d[LIMBS], t[LIMBS], r[LIMBS];
    load4(base + (unsigned long long)b * LIMBS, lo);
    load4(base + ((unsigned long long)b + half) * LIMBS, hi);
    load4(challenge, c);
    fr_sub(hi, lo, d);
    fr_mul(c, d, t);
    fr_add(lo, t, r);
    store4(out + (unsigned long long)i * LIMBS, r);
}

__device__ __forceinline__ void ap_extension(const u64 *__restrict__ column,
                                             unsigned int b,
                                             unsigned int half,
                                             unsigned int c,
                                             u64 *out) {
    u64 lo[LIMBS], hi[LIMBS];
    load4(column + (unsigned long long)b * LIMBS, lo);
    load4(column + ((unsigned long long)b + half) * LIMBS, hi);
    if (c == 0) { store4(out, lo); return; }
    if (c == 1) { store4(out, hi); return; }
    u64 doubled[LIMBS];
    fr_add(hi, hi, doubled);
    fr_sub(doubled, lo, out);
}

extern "C" __global__ void ap_round_message_kernel(
    const u64 *__restrict__ prefixes,
    const unsigned int *__restrict__ prefix_ids,
    const unsigned int *__restrict__ suffix_slots,
    const unsigned int *__restrict__ scales,
    const unsigned int *__restrict__ term_offsets,
    const unsigned int *__restrict__ term_counts,
    const u64 *__restrict__ suffixes,
    const unsigned int *__restrict__ suffix_bases,
    unsigned int table_count,
    const u64 *__restrict__ raf_prefix,
    const u64 *__restrict__ raf_shift_half,
    const u64 *__restrict__ raf_shift_full,
    const u64 *__restrict__ raf_left,
    const u64 *__restrict__ raf_right,
    const u64 *__restrict__ raf_identity,
    unsigned int raf_count,
    unsigned int stride,
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
                        const u64 *column =
                            suffixes + (unsigned long long)(suffix_bases[t] + suffix_slots[term]) *
                                           stride * LIMBS;
                        u64 value[LIMBS];
                        ap_extension(column, b, half, c, value);
                        unsigned int prefix = prefix_ids[term];
                        if (prefix != 0xFFFFFFFFu) {
                            u64 p[LIMBS], product[LIMBS];
                            ap_extension(prefixes + (unsigned long long)prefix * stride * LIMBS,
                                         b, half, c, p);
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
                const u64 *prefix_column;
                const u64 *shift_column;
                const u64 *value_column;
                if (slot == 1u) {
                    prefix_column = raf_prefix;
                    shift_column = raf_shift_half;
                    value_column = raf_left;
                } else if (slot == 2u) {
                    prefix_column = raf_prefix + (unsigned long long)stride * LIMBS;
                    shift_column = raf_shift_half;
                    value_column = raf_right;
                } else {
                    prefix_column = raf_prefix + (unsigned long long)2 * stride * LIMBS;
                    shift_column = raf_shift_full;
                    value_column = raf_identity;
                }
                u64 p[LIMBS], shift[LIMBS], value[LIMBS], product[LIMBS], sum[LIMBS];
                ap_extension(prefix_column, b, half, c, p);
                ap_extension(shift_column, b, half, c, shift);
                ap_extension(value_column, b, half, c, value);
                fr_mul(p, shift, product);
                fr_add(product, value, sum);
                store4(acc, sum);
            }
        }

        store4(scratch + tid * LIMBS, acc);
        __syncthreads();
        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                u64 x[LIMBS], y[LIMBS], sum[LIMBS];
                load4(scratch + tid * LIMBS, x);
                load4(scratch + (tid + s) * LIMBS, y);
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
