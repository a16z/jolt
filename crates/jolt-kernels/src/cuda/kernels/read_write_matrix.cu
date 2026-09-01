#define RWM_MAX_COEFFS 2

extern "C" __global__ void rwm_square_lut_kernel(const u64 *__restrict__ values, unsigned int values_len,
                                                const u64 *__restrict__ challenge,
                                                u64 *__restrict__ out) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= values_len * values_len) return;
    u64 r[LIMBS], odd[LIMBS], even[LIMBS], diff[LIMBS], scaled[LIMBS], bound[LIMBS];
    load4(challenge, r);
    load4(values + (unsigned long long)(index / values_len) * LIMBS, odd);
    load4(values + (unsigned long long)(index % values_len) * LIMBS, even);
    fr_sub(odd, even, diff);
    fr_mul(r, diff, scaled);
    fr_add(even, scaled, bound);
    store4(out + (unsigned long long)index * LIMBS, bound);
}

extern "C" __global__ void rwm_deref_coeffs_kernel(const unsigned short *__restrict__ coeff_index,
                                                  const u64 *__restrict__ lut_0,
                                                  const u64 *__restrict__ lut_1,
                                                  unsigned int slots, u64 *__restrict__ out) {
    unsigned int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= slots) return;
    const u64 *lut = (slot % RWM_MAX_COEFFS) ? lut_1 : lut_0;
    u64 value[LIMBS];
    load4(lut + (unsigned long long)coeff_index[slot] * LIMBS, value);
    store4(out + (unsigned long long)slot * LIMBS, value);
}

__device__ __forceinline__ void rwm_load_coeff(const u64 *__restrict__ coeffs,
                                              const unsigned short *__restrict__ coeff_index,
                                              const u64 *__restrict__ lut, unsigned int indexed,
                                              unsigned long long slot, u64 *out) {
    if (indexed) {
        load4(lut + (unsigned long long)coeff_index[slot] * LIMBS, out);
    } else {
        load4(coeffs + slot * LIMBS, out);
    }
}

extern "C" __global__ void rwm_segment_flags_kernel(const unsigned int *__restrict__ rows,
                                                   unsigned int entries,
                                                   unsigned int *__restrict__ flags) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= entries) return;
    unsigned int pair = rows[index] >> 1;
    flags[index] = (index == 0 || (rows[index - 1] >> 1) != pair) ? 1u : 0u;
}

extern "C" __global__ void rwm_segment_bounds_kernel(const unsigned int *__restrict__ rows,
                                                    const unsigned int *__restrict__ flags,
                                                    const unsigned int *__restrict__ ranks,
                                                    unsigned int entries,
                                                    unsigned int *__restrict__ seg_start,
                                                    unsigned int *__restrict__ seg_even_end,
                                                    unsigned int *__restrict__ seg_end,
                                                    unsigned int *__restrict__ seg_pair) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= entries) return;
    if (!flags[index]) return;
    unsigned int seg = ranks[index];
    unsigned int pair = rows[index] >> 1;
    seg_start[seg] = index;
    seg_pair[seg] = pair;

    unsigned int end = index;
    while (end < entries && (rows[end] >> 1) == pair) end++;
    seg_end[seg] = end;

    unsigned int even_end = index;
    while (even_end < end && (rows[even_end] & 1u) == 0u) even_end++;
    seg_even_end[seg] = even_end;
}

extern "C" __global__ void rwm_count_kernel(const unsigned int *__restrict__ cols,
                                           const unsigned int *__restrict__ seg_start,
                                           const unsigned int *__restrict__ seg_even_end,
                                           const unsigned int *__restrict__ seg_end,
                                           unsigned int segments,
                                           unsigned int *__restrict__ counts) {
    unsigned int seg = blockIdx.x * blockDim.x + threadIdx.x;
    if (seg >= segments) return;

    unsigned int i = seg_start[seg];
    unsigned int even_end = seg_even_end[seg];
    unsigned int j = even_end;
    unsigned int odd_end = seg_end[seg];

    unsigned int count = 0;
    while (i < even_end && j < odd_end) {
        unsigned int a = cols[i];
        unsigned int b = cols[j];
        if (a == b) {
            i++;
            j++;
        } else if (a < b) {
            i++;
        } else {
            j++;
        }
        count++;
    }
    counts[seg] = count + (even_end - i) + (odd_end - j);
}

extern "C" __global__ void rwm_merge_kernel(
    const unsigned int *__restrict__ cols, const u64 *__restrict__ val_coeff,
    const u64 *__restrict__ prev_val, const u64 *__restrict__ next_val,
    const u64 *__restrict__ coeffs, unsigned int coeff_width,
    const unsigned short *__restrict__ coeff_index, unsigned int indexed,
    unsigned int lut_bits_0, unsigned int lut_bits_1,
    const unsigned int *__restrict__ seg_start, const unsigned int *__restrict__ seg_even_end,
    const unsigned int *__restrict__ seg_end, const unsigned int *__restrict__ seg_pair,
    const unsigned int *__restrict__ offsets, unsigned int segments,
    const u64 *__restrict__ challenge, unsigned int *__restrict__ out_rows,
    unsigned int *__restrict__ out_cols, u64 *__restrict__ out_val,
    u64 *__restrict__ out_prev, u64 *__restrict__ out_next, u64 *__restrict__ out_coeffs,
    unsigned short *__restrict__ out_coeff_index) {
    unsigned int seg = blockIdx.x * blockDim.x + threadIdx.x;
    if (seg >= segments) return;

    u64 r[LIMBS], one[LIMBS];
    load4(challenge, r);
    load4(FR_ONE, one);

    unsigned int i = seg_start[seg];
    unsigned int even_end = seg_even_end[seg];
    unsigned int j = even_end;
    unsigned int odd_end = seg_end[seg];
    unsigned int out_row = seg_pair[seg];
    unsigned int k = offsets[seg];

    while (i < even_end || j < odd_end) {
        bool take_even;
        bool take_odd;
        if (i < even_end && j < odd_end) {
            unsigned int a = cols[i];
            unsigned int b = cols[j];
            take_even = a <= b;
            take_odd = b <= a;
        } else if (i < even_end) {
            take_even = true;
            take_odd = false;
        } else {
            take_even = false;
            take_odd = true;
        }

        unsigned int col = take_even ? cols[i] : cols[j];
        u64 even_val[LIMBS], odd_val[LIMBS];
        u64 prev_out, next_out;

        if (take_even && take_odd) {
            load4(val_coeff + (unsigned long long)i * LIMBS, even_val);
            load4(val_coeff + (unsigned long long)j * LIMBS, odd_val);
            prev_out = prev_val[i];
            next_out = next_val[j];
        } else if (take_even) {
            load4(val_coeff + (unsigned long long)i * LIMBS, even_val);
            u64 raw[LIMBS] = {next_val[i], 0, 0, 0};
            fr_to_mont(raw, odd_val);
            prev_out = prev_val[i];
            next_out = next_val[i];
        } else {
            u64 raw[LIMBS] = {prev_val[j], 0, 0, 0};
            fr_to_mont(raw, even_val);
            load4(val_coeff + (unsigned long long)j * LIMBS, odd_val);
            prev_out = prev_val[j];
            next_out = next_val[j];
        }

        u64 diff[LIMBS], scaled[LIMBS], bound_val[LIMBS];
        fr_sub(odd_val, even_val, diff);
        fr_mul(r, diff, scaled);
        fr_add(even_val, scaled, bound_val);

        out_rows[k] = out_row;
        out_cols[k] = col;
        store4(out_val + (unsigned long long)k * LIMBS, bound_val);
        out_prev[k] = prev_out;
        out_next[k] = next_out;

        for (unsigned int lane = 0; lane < coeff_width; lane++) {
            if (indexed) {
                unsigned int bits = lane ? lut_bits_1 : lut_bits_0;
                unsigned short even_index =
                    take_even ? coeff_index[(unsigned long long)i * coeff_width + lane] : 0;
                unsigned short odd_index =
                    take_odd ? coeff_index[(unsigned long long)j * coeff_width + lane] : 0;
                out_coeff_index[(unsigned long long)k * coeff_width + lane] =
                    (unsigned short)((unsigned int)odd_index << bits) | even_index;
                continue;
            }
            u64 bound[LIMBS];
            if (take_even && take_odd) {
                u64 e[LIMBS], o[LIMBS], d[LIMBS], s[LIMBS];
                load4(coeffs + ((unsigned long long)i * coeff_width + lane) * LIMBS, e);
                load4(coeffs + ((unsigned long long)j * coeff_width + lane) * LIMBS, o);
                fr_sub(o, e, d);
                fr_mul(r, d, s);
                fr_add(e, s, bound);
            } else if (take_even) {
                u64 e[LIMBS], one_minus[LIMBS];
                load4(coeffs + ((unsigned long long)i * coeff_width + lane) * LIMBS, e);
                fr_sub(one, r, one_minus);
                fr_mul(one_minus, e, bound);
            } else {
                u64 o[LIMBS];
                load4(coeffs + ((unsigned long long)j * coeff_width + lane) * LIMBS, o);
                fr_mul(r, o, bound);
            }
            store4(out_coeffs + ((unsigned long long)k * coeff_width + lane) * LIMBS, bound);
        }

        if (take_even) i++;
        if (take_odd) j++;
        k++;
    }
}

extern "C" __global__ void rwm_message_kernel(
    const unsigned int *__restrict__ cols, const u64 *__restrict__ val_coeff,
    const u64 *__restrict__ prev_val, const u64 *__restrict__ next_val,
    const u64 *__restrict__ coeffs, unsigned int coeff_width,
    const unsigned short *__restrict__ coeff_index, unsigned int indexed,
    const u64 *__restrict__ lut_0, const u64 *__restrict__ lut_1,
    const unsigned int *__restrict__ seg_start, const unsigned int *__restrict__ seg_even_end,
    const unsigned int *__restrict__ seg_end, const unsigned int *__restrict__ seg_pair,
    unsigned int segments, const u64 *__restrict__ inc, const u64 *__restrict__ e_in,
    unsigned int e_in_len, const u64 *__restrict__ e_out, unsigned int num_x_in_bits,
    const u64 *__restrict__ wa_scale, u64 *__restrict__ partials) {
    extern __shared__ u64 scratch[];
    unsigned int tid = threadIdx.x;
    unsigned int seg = blockIdx.x * blockDim.x + tid;

    u64 acc[2][LIMBS];
    for (int lane = 0; lane < 2; lane++)
        for (int l = 0; l < LIMBS; l++) acc[lane][l] = 0;

    if (seg < segments) {
        unsigned int i = seg_start[seg];
        unsigned int even_end = seg_even_end[seg];
        unsigned int j = even_end;
        unsigned int odd_end = seg_end[seg];
        unsigned int pair = seg_pair[seg];
        unsigned int ra_lane = 0;
        unsigned int wa_lane = (coeff_width > 1) ? 1 : 0;
        const u64 *ra_lut = lut_0;
        const u64 *wa_lut = wa_lane ? lut_1 : lut_0;

        u64 inc_0[LIMBS], inc_inf[LIMBS];
        load4(inc + (unsigned long long)(2 * pair) * LIMBS, inc_0);
        {
            u64 inc_1[LIMBS];
            load4(inc + (unsigned long long)(2 * pair + 1) * LIMBS, inc_1);
            fr_sub(inc_1, inc_0, inc_inf);
        }

        u64 scale[LIMBS];
        load4(wa_scale, scale);

        u64 inner[2][2 * PA_SLOTS];
        for (int lane = 0; lane < 2; lane++) pa_zero(inner[lane]);

        while (i < even_end || j < odd_end) {
            bool take_even;
            bool take_odd;
            if (i < even_end && j < odd_end) {
                unsigned int a = cols[i];
                unsigned int b = cols[j];
                take_even = a <= b;
                take_odd = b <= a;
            } else if (i < even_end) {
                take_even = true;
                take_odd = false;
            } else {
                take_even = false;
                take_odd = true;
            }

            u64 val_0[LIMBS], val_inf[LIMBS];
            u64 ra_0[LIMBS], ra_inf[LIMBS], wa_0[LIMBS], wa_inf[LIMBS];

            if (take_even && take_odd) {
                u64 even_val[LIMBS], odd_val[LIMBS];
                load4(val_coeff + (unsigned long long)i * LIMBS, even_val);
                load4(val_coeff + (unsigned long long)j * LIMBS, odd_val);
                for (int l = 0; l < LIMBS; l++) val_0[l] = even_val[l];
                fr_sub(odd_val, even_val, val_inf);

                u64 e[LIMBS], o[LIMBS];
                rwm_load_coeff(coeffs, coeff_index, ra_lut, indexed,
                              (unsigned long long)i * coeff_width + ra_lane, e);
                rwm_load_coeff(coeffs, coeff_index, ra_lut, indexed,
                              (unsigned long long)j * coeff_width + ra_lane, o);
                for (int l = 0; l < LIMBS; l++) ra_0[l] = e[l];
                fr_sub(o, e, ra_inf);
                rwm_load_coeff(coeffs, coeff_index, wa_lut, indexed,
                              (unsigned long long)i * coeff_width + wa_lane, e);
                rwm_load_coeff(coeffs, coeff_index, wa_lut, indexed,
                              (unsigned long long)j * coeff_width + wa_lane, o);
                for (int l = 0; l < LIMBS; l++) wa_0[l] = e[l];
                fr_sub(o, e, wa_inf);
            } else if (take_even) {
                u64 even_val[LIMBS], odd_val[LIMBS];
                load4(val_coeff + (unsigned long long)i * LIMBS, even_val);
                u64 raw[LIMBS] = {next_val[i], 0, 0, 0};
                fr_to_mont(raw, odd_val);
                for (int l = 0; l < LIMBS; l++) val_0[l] = even_val[l];
                fr_sub(odd_val, even_val, val_inf);

                u64 zero[LIMBS] = {0, 0, 0, 0};
                u64 e[LIMBS];
                rwm_load_coeff(coeffs, coeff_index, ra_lut, indexed,
                              (unsigned long long)i * coeff_width + ra_lane, e);
                for (int l = 0; l < LIMBS; l++) ra_0[l] = e[l];
                fr_sub(zero, e, ra_inf);
                rwm_load_coeff(coeffs, coeff_index, wa_lut, indexed,
                              (unsigned long long)i * coeff_width + wa_lane, e);
                for (int l = 0; l < LIMBS; l++) wa_0[l] = e[l];
                fr_sub(zero, e, wa_inf);
            } else {
                u64 even_val[LIMBS], odd_val[LIMBS];
                u64 raw[LIMBS] = {prev_val[j], 0, 0, 0};
                fr_to_mont(raw, even_val);
                load4(val_coeff + (unsigned long long)j * LIMBS, odd_val);
                for (int l = 0; l < LIMBS; l++) val_0[l] = even_val[l];
                fr_sub(odd_val, even_val, val_inf);

                for (int l = 0; l < LIMBS; l++) {
                    ra_0[l] = 0;
                    wa_0[l] = 0;
                }
                rwm_load_coeff(coeffs, coeff_index, ra_lut, indexed,
                              (unsigned long long)j * coeff_width + ra_lane, ra_inf);
                rwm_load_coeff(coeffs, coeff_index, wa_lut, indexed,
                              (unsigned long long)j * coeff_width + wa_lane, wa_inf);
            }

            if (coeff_width == 1) {
                u64 scaled[LIMBS];
                fr_mul(scale, wa_0, scaled);
                for (int l = 0; l < LIMBS; l++) wa_0[l] = scaled[l];
                fr_mul(scale, wa_inf, scaled);
                for (int l = 0; l < LIMBS; l++) wa_inf[l] = scaled[l];
            }

            u64 sum[LIMBS];
            pa_fold_mul_accum(ra_0, val_0, inner[0]);
            fr_add(val_0, inc_0, sum);
            pa_fold_mul_accum(wa_0, sum, inner[0]);

            pa_fold_mul_accum(ra_inf, val_inf, inner[1]);
            fr_add(val_inf, inc_inf, sum);
            pa_fold_mul_accum(wa_inf, sum, inner[1]);

            if (take_even) i++;
            if (take_odd) j++;
        }

        u64 weight[LIMBS];
        if (e_in_len <= 1) {
            load4(FR_ONE, weight);
        } else {
            unsigned int x_in = pair & ((1u << num_x_in_bits) - 1u);
            load4(e_in + (unsigned long long)x_in * LIMBS, weight);
        }
        u64 e_out_eval[LIMBS];
        load4(e_out + (unsigned long long)(pair >> num_x_in_bits) * LIMBS, e_out_eval);
        u64 combined[LIMBS];
        fr_mul(weight, e_out_eval, combined);

        for (int lane = 0; lane < 2; lane++) {
            u64 reduced[LIMBS];
            pa_finalize(inner[lane], reduced);
            fr_mul(reduced, combined, acc[lane]);
        }
    }

    for (int lane = 0; lane < 2; lane++) {
        store4(scratch + tid * LIMBS, acc[lane]);
        __syncthreads();
        for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                u64 a[LIMBS], b[LIMBS], s[LIMBS];
                load4(scratch + tid * LIMBS, a);
                load4(scratch + (tid + stride) * LIMBS, b);
                fr_add(a, b, s);
                store4(scratch + tid * LIMBS, s);
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
