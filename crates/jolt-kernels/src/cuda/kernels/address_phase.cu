#define AP_CHUNK_LEN 8
#define AP_CHUNK_SIZE 256
#define AP_RAF_LANES 6
#define AP_MAX_SUFFIXES 4
#define AP_HINT_POINTS 2
#define AP_PREFIX_COUNT 46
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

__device__ __forceinline__ void ap_block_reduce_folded(u64 *scratch, const u64 *acc) {
    unsigned int tid = threadIdx.x;
    for (int i = 0; i < 2 * UNR_SLOTS; i++) scratch[tid * (2 * UNR_SLOTS) + i] = acc[i];
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            unr_add_folded(scratch + tid * (2 * UNR_SLOTS),
                           scratch + (tid + stride) * (2 * UNR_SLOTS));
        }
        __syncthreads();
    }
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

extern "C" __global__ void ap_raf_reduce_chunked_kernel(
    const unsigned int *__restrict__ order, const unsigned int *__restrict__ offsets,
    const unsigned int *__restrict__ counts, const unsigned long long *__restrict__ lookup_index,
    const unsigned char *__restrict__ raf_flags, const u64 *__restrict__ u_evals,
    unsigned int suffix_len, unsigned int upper_suffix_bits, unsigned int canonical,
    unsigned int chunks, u64 *__restrict__ slots) {
    extern __shared__ u64 scratch[];
    unsigned int bucket = blockIdx.x / chunks;
    unsigned int chunk = blockIdx.x % chunks;
    unsigned int start = offsets[bucket];
    unsigned int count = counts[bucket];
    if (chunk * blockDim.x >= count) return;

    u64 acc[AP_RAF_LANES][2 * UNR_SLOTS];
    for (int lane = 0; lane < AP_RAF_LANES; lane++) unr_zero(acc[lane]);

    for (unsigned int i = chunk * blockDim.x + threadIdx.x; i < count; i += chunks * blockDim.x) {
        unsigned int j = order[start + i];
        u64 u[LIMBS];
        load4(u_evals + (unsigned long long)j * LIMBS, u);
        u128 index = ap_index(lookup_index, j);
        u128 suffix_bits = index & sfx_mask(suffix_len);

        if (canonical != 0u && raf_flags[j] != 0u) {
            u128 ones = ((u128)1 << upper_suffix_bits) - 1;
            if (upper_suffix_bits == 0u ||
                (suffix_bits >> (suffix_len - upper_suffix_bits)) == ones) {
                unr_add_field(acc[5], u);
            }
        }

        if (raf_flags[j] == 0u) {
            unr_add_field(acc[0], u);
            sfx_bits whole = sfx_new(suffix_bits, suffix_len);
            sfx_bits left, right;
            sfx_uninterleave(whole, &left, &right);
            unsigned long long left_value = sfx_u64(left);
            unr_mul_words(u, &left_value, 1u, acc[2]);
            unsigned long long right_value = sfx_u64(right);
            unr_mul_words(u, &right_value, 1u, acc[3]);
        } else {
            unr_add_field(acc[1], u);
            unsigned long long identity[2] = {(unsigned long long)suffix_bits,
                                              (unsigned long long)(suffix_bits >> 64)};
            unr_mul_words(u, identity, 2u, acc[4]);
        }
    }

    for (int lane = 0; lane < AP_RAF_LANES; lane++) {
        ap_block_reduce_folded(scratch, acc[lane]);
        if (threadIdx.x == 0) {
            u64 *target =
                slots +
                (((unsigned long long)lane * AP_CHUNK_SIZE + bucket) * chunks + chunk) *
                    (2 * UNR_SLOTS);
            for (int i = 0; i < 2 * UNR_SLOTS; i++) target[i] = scratch[i];
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
    u64 *__restrict__ slots) {
    extern __shared__ u64 scratch[];
    unsigned int slot = blockIdx.x / AP_CHUNK_SIZE;
    unsigned int bucket = blockIdx.x % AP_CHUNK_SIZE;
    unsigned int start = offsets[blockIdx.x];
    unsigned int count = counts[blockIdx.x];
    unsigned int families = suffix_counts[slot];
    unsigned int family_base = suffix_offsets[slot];
    if (blockIdx.y * blockDim.x >= count) return;

    u64 acc[AP_MAX_SUFFIXES][2 * UNR_SLOTS];
    for (int s = 0; s < AP_MAX_SUFFIXES; s++) unr_zero(acc[s]);

    for (unsigned int i = blockIdx.y * blockDim.x + threadIdx.x; i < count;
         i += gridDim.y * blockDim.x) {
        unsigned int j = order[start + i];
        u64 u[LIMBS];
        load4(u_evals + (unsigned long long)j * LIMBS, u);
        u128 index = ap_index(lookup_index, j);
        sfx_bits suffix = sfx_new(index & sfx_mask(suffix_len), suffix_len);

        for (unsigned int s = 0; s < families; s++) {
            unsigned long long value = sfx_eval(suffix_ids[family_base + s], suffix);
            if (value == 0ULL) continue;
            if (value == 1ULL) {
                unr_add_field(acc[s], u);
                continue;
            }
            unr_mul_words(u, &value, 1u, acc[s]);
        }
    }

    for (unsigned int s = 0; s < families; s++) {
        ap_block_reduce_folded(scratch, acc[s]);
        if (threadIdx.x == 0) {
            u64 *target =
                slots +
                ((((unsigned long long)(family_base + s) * AP_CHUNK_SIZE + bucket) * gridDim.y) +
                 blockIdx.y) *
                    (2 * UNR_SLOTS);
            for (int i = 0; i < 2 * UNR_SLOTS; i++) target[i] = scratch[i];
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

extern "C" __global__ void ap_bind_lanes_kernel(const u64 *const *__restrict__ in_ptrs,
                                               u64 *const *__restrict__ out_ptrs,
                                               const unsigned int *__restrict__ counts,
                                               u64 c0, u64 c1, u64 c2, u64 c3,
                                               unsigned int half,
                                               unsigned int stride,
                                               unsigned int max_count) {
    unsigned int lane = blockIdx.y;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= max_count || i >= counts[lane]) return;
    const u64 *in = in_ptrs[lane];
    u64 *out = out_ptrs[lane];
    unsigned int column = i / half;
    unsigned int b = i % half;
    const u64 *base = in + (unsigned long long)column * stride * LIMBS;
    u64 lo[LIMBS], hi[LIMBS], d[LIMBS], t[LIMBS], r[LIMBS];
    u64 c[LIMBS] = {c0, c1, c2, c3};
    load4(base + (unsigned long long)b * LIMBS, lo);
    load4(base + ((unsigned long long)b + half) * LIMBS, hi);
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

extern "C" __global__ void ap_round_message_hinted_kernel(
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
    u64 *__restrict__ slots) {
    extern __shared__ u64 scratch[];
    unsigned int tid = threadIdx.x;
    unsigned int b = blockIdx.x * blockDim.x + tid;
    unsigned int lanes = AP_HINT_POINTS * (1u + raf_count);

    for (unsigned int lane = 0; lane < lanes; lane++) {
        unsigned int slot = lane % (1u + raf_count);
        unsigned int c = (lane / (1u + raf_count)) == 0u ? 0u : 2u;
        u64 folded[2 * UNR_SLOTS];
        unr_zero(folded);

        if (b < half) {
            if (slot == 0) {
                for (unsigned int t = 0; t < table_count; t++) {
                    unsigned int count = term_counts[t];
                    unsigned int base = term_offsets[t];
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
                            unsigned long long slot =
                                ((unsigned long long)(c == 0u ? 0u : 1u) * AP_PREFIX_COUNT +
                                 prefix) * half + b;
                            load4(prefixes + slot * LIMBS, p);
                            fr_mul(value, p, product);
                            store4(value, product);
                        }
                        if (scales[term] != 0u) {
                            u64 s[LIMBS], scaled[LIMBS];
                            cmb_scale(scales[term], s);
                            fr_mul(value, s, scaled);
                            store4(value, scaled);
                        }
                        unr_add_field(folded, value);
                    }
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
                unr_add_field(folded, sum);
            }
        }

        ap_block_reduce_folded(scratch, folded);
        if (tid == 0) {
            u64 *target =
                slots + ((unsigned long long)lane * gridDim.x + blockIdx.x) * (2 * UNR_SLOTS);
            for (int i = 0; i < 2 * UNR_SLOTS; i++) target[i] = scratch[i];
        }
        __syncthreads();
    }
}

extern "C" __global__ void ap_combined_val_kernel(
    const unsigned int *__restrict__ table_index,
    const unsigned char *__restrict__ raf_flags,
    const u64 *__restrict__ table_values,
    const u64 *__restrict__ raf_interleaved,
    const u64 *__restrict__ raf_identity,
    unsigned int table_count,
    u64 *__restrict__ out,
    unsigned int n) {
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;

    u64 value[LIMBS] = {0, 0, 0, 0};
    unsigned int table = table_index[j];
    if (table != AP_NO_TABLE && table < table_count) {
        load4(table_values + (unsigned long long)table * LIMBS, value);
    }

    u64 raf[LIMBS];
    if (raf_flags[j] != 0u) {
        load4(raf_identity, raf);
    } else {
        load4(raf_interleaved, raf);
    }

    u64 sum[LIMBS];
    fr_add(value, raf, sum);
    store4(out + (unsigned long long)j * LIMBS, sum);
}

extern "C" __global__ void ap_ra_kernel(const unsigned long long *__restrict__ lookup_index,
                                       const u64 *const *__restrict__ v_tables,
                                       unsigned int phase_offset,
                                       unsigned int phases_per_ra,
                                       unsigned int total_phases,
                                       u64 *__restrict__ out,
                                       unsigned int n) {
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;

    u128 index = ap_index(lookup_index, j);
    u64 acc[LIMBS];
    load4(FR_ONE, acc);
    for (unsigned int q = 0; q < phases_per_ra; q++) {
        unsigned int phase = phase_offset + q;
        unsigned int suffix_len = (total_phases - 1u - phase) * AP_CHUNK_LEN;
        unsigned int chunk = ap_chunk(index, suffix_len);
        u64 value[LIMBS], product[LIMBS];
        load4(v_tables[phase] + (unsigned long long)chunk * LIMBS, value);
        fr_mul(acc, value, product);
        store4(acc, product);
    }
    store4(out + (unsigned long long)j * LIMBS, acc);
}

extern "C" __global__ void ap_flag_sums_kernel(const unsigned int *__restrict__ order,
                                               const unsigned int *__restrict__ offsets,
                                               const unsigned int *__restrict__ counts,
                                               const u64 *__restrict__ eq_cycle,
                                               u64 *__restrict__ out) {
    extern __shared__ u64 scratch[];
    unsigned int bucket = blockIdx.x;
    unsigned int start = offsets[bucket];
    unsigned int count = counts[bucket];

    u64 acc[LIMBS] = {0, 0, 0, 0};
    for (unsigned int i = threadIdx.x; i < count; i += blockDim.x) {
        unsigned int j = order[start + i];
        u64 eq[LIMBS];
        load4(eq_cycle + (unsigned long long)j * LIMBS, eq);
        fr_add(acc, eq, acc);
    }

    ap_block_reduce(scratch, acc);
    if (threadIdx.x == 0) {
        u64 total[LIMBS];
        load4(scratch, total);
        store4(out + (unsigned long long)bucket * LIMBS, total);
    }
}

extern "C" __global__ void ap_flag_keys_kernel(const unsigned int *__restrict__ table_index,
                                               unsigned int table_count,
                                               unsigned int *__restrict__ keys,
                                               unsigned int n) {
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    unsigned int table = table_index[j];
    keys[j] = (table == AP_NO_TABLE || table >= table_count) ? AP_SKIP : table;
}

extern "C" __global__ void ap_raf_flag_sum_kernel(const unsigned char *__restrict__ raf_flags,
                                                 const u64 *__restrict__ eq_cycle,
                                                 u64 *__restrict__ partials,
                                                 unsigned int n) {
    extern __shared__ u64 scratch[];
    unsigned int tid = threadIdx.x;
    unsigned int j = blockIdx.x * blockDim.x + tid;

    u64 acc[LIMBS] = {0, 0, 0, 0};
    if (j < n && raf_flags[j] != 0u) {
        load4(eq_cycle + (unsigned long long)j * LIMBS, acc);
    }

    ap_block_reduce(scratch, acc);
    if (tid == 0) {
        u64 total[LIMBS];
        load4(scratch, total);
        store4(partials + (unsigned long long)blockIdx.x * LIMBS, total);
    }
}
