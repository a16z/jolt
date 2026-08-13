extern "C" __global__ void amm_segment_flags_kernel(const unsigned int *__restrict__ cols,
                                                   unsigned int entries,
                                                   unsigned int *__restrict__ flags) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= entries) return;
    unsigned int pair = cols[index] >> 1;
    flags[index] = (index == 0 || (cols[index - 1] >> 1) != pair) ? 1u : 0u;
}

extern "C" __global__ void amm_segment_bounds_kernel(const unsigned int *__restrict__ cols,
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
    unsigned int pair = cols[index] >> 1;
    seg_start[seg] = index;
    seg_pair[seg] = pair;

    unsigned int end = index;
    while (end < entries && (cols[end] >> 1) == pair) end++;
    seg_end[seg] = end;

    unsigned int even_end = index;
    while (even_end < end && (cols[even_end] & 1u) == 0u) even_end++;
    seg_even_end[seg] = even_end;
}

extern "C" __global__ void amm_count_kernel(const unsigned int *__restrict__ rows,
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
        unsigned int a = rows[i];
        unsigned int b = rows[j];
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

extern "C" __global__ void amm_merge_kernel(
    const unsigned int *__restrict__ rows, const u64 *__restrict__ val_coeff,
    const u64 *__restrict__ prev_val, const u64 *__restrict__ next_val,
    const u64 *__restrict__ coeffs, unsigned int coeff_width,
    const unsigned int *__restrict__ seg_start, const unsigned int *__restrict__ seg_even_end,
    const unsigned int *__restrict__ seg_end, const unsigned int *__restrict__ seg_pair,
    const unsigned int *__restrict__ offsets, unsigned int segments,
    const u64 *__restrict__ challenge, const u64 *__restrict__ val_init,
    unsigned int *__restrict__ out_rows, unsigned int *__restrict__ out_cols,
    u64 *__restrict__ out_val, u64 *__restrict__ out_prev, u64 *__restrict__ out_next,
    u64 *__restrict__ out_coeffs) {
    unsigned int seg = blockIdx.x * blockDim.x + threadIdx.x;
    if (seg >= segments) return;

    u64 r[LIMBS], one[LIMBS], one_minus_r[LIMBS];
    load4(challenge, r);
    load4(FR_ONE, one);
    fr_sub(one, r, one_minus_r);

    unsigned int i = seg_start[seg];
    unsigned int even_end = seg_even_end[seg];
    unsigned int j = even_end;
    unsigned int odd_end = seg_end[seg];
    unsigned int pair = seg_pair[seg];
    unsigned int k = offsets[seg];

    u64 even_cp[LIMBS], odd_cp[LIMBS];
    load4(val_init + (unsigned long long)(2 * pair) * LIMBS, even_cp);
    load4(val_init + (unsigned long long)(2 * pair + 1) * LIMBS, odd_cp);

    while (i < even_end || j < odd_end) {
        bool in_main = (i < even_end) && (j < odd_end);
        bool take_even;
        bool take_odd;
        if (in_main) {
            unsigned int a = rows[i];
            unsigned int b = rows[j];
            take_even = a <= b;
            take_odd = b <= a;
        } else if (i < even_end) {
            take_even = true;
            take_odd = false;
        } else {
            take_even = false;
            take_odd = true;
        }

        unsigned int out_row = take_even ? rows[i] : rows[j];
        u64 lo_val[LIMBS], hi_val[LIMBS], lo_prev[LIMBS], hi_prev[LIMBS];
        u64 lo_next[LIMBS], hi_next[LIMBS];

        if (take_even && take_odd) {
            load4(val_coeff + (unsigned long long)i * LIMBS, lo_val);
            load4(val_coeff + (unsigned long long)j * LIMBS, hi_val);
            load4(prev_val + (unsigned long long)i * LIMBS, lo_prev);
            load4(prev_val + (unsigned long long)j * LIMBS, hi_prev);
            load4(next_val + (unsigned long long)i * LIMBS, lo_next);
            load4(next_val + (unsigned long long)j * LIMBS, hi_next);
        } else if (take_even) {
            load4(val_coeff + (unsigned long long)i * LIMBS, lo_val);
            load4(prev_val + (unsigned long long)i * LIMBS, lo_prev);
            load4(next_val + (unsigned long long)i * LIMBS, lo_next);
            for (int l = 0; l < LIMBS; l++) {
                hi_val[l] = odd_cp[l];
                hi_prev[l] = odd_cp[l];
                hi_next[l] = odd_cp[l];
            }
        } else {
            load4(val_coeff + (unsigned long long)j * LIMBS, hi_val);
            load4(prev_val + (unsigned long long)j * LIMBS, hi_prev);
            load4(next_val + (unsigned long long)j * LIMBS, hi_next);
            for (int l = 0; l < LIMBS; l++) {
                lo_val[l] = even_cp[l];
                lo_prev[l] = even_cp[l];
                lo_next[l] = even_cp[l];
            }
        }

        u64 diff[LIMBS], scaled[LIMBS], bound[LIMBS];
        fr_sub(hi_val, lo_val, diff);
        fr_mul(r, diff, scaled);
        fr_add(lo_val, scaled, bound);
        store4(out_val + (unsigned long long)k * LIMBS, bound);

        fr_sub(hi_prev, lo_prev, diff);
        fr_mul(r, diff, scaled);
        fr_add(lo_prev, scaled, bound);
        store4(out_prev + (unsigned long long)k * LIMBS, bound);

        fr_sub(hi_next, lo_next, diff);
        fr_mul(r, diff, scaled);
        fr_add(lo_next, scaled, bound);
        store4(out_next + (unsigned long long)k * LIMBS, bound);

        out_rows[k] = out_row;
        out_cols[k] = pair;

        for (unsigned int lane = 0; lane < coeff_width; lane++) {
            u64 c[LIMBS];
            if (take_even && take_odd) {
                u64 e[LIMBS], o[LIMBS], d[LIMBS], s[LIMBS];
                load4(coeffs + ((unsigned long long)i * coeff_width + lane) * LIMBS, e);
                load4(coeffs + ((unsigned long long)j * coeff_width + lane) * LIMBS, o);
                fr_sub(o, e, d);
                fr_mul(r, d, s);
                fr_add(e, s, c);
            } else if (take_even) {
                u64 e[LIMBS];
                load4(coeffs + ((unsigned long long)i * coeff_width + lane) * LIMBS, e);
                fr_mul(one_minus_r, e, c);
            } else {
                u64 o[LIMBS];
                load4(coeffs + ((unsigned long long)j * coeff_width + lane) * LIMBS, o);
                fr_mul(r, o, c);
            }
            store4(out_coeffs + ((unsigned long long)k * coeff_width + lane) * LIMBS, c);
        }

        if (in_main) {
            if (take_even) load4(next_val + (unsigned long long)i * LIMBS, even_cp);
            if (take_odd) load4(next_val + (unsigned long long)j * LIMBS, odd_cp);
        }

        if (take_even) i++;
        if (take_odd) j++;
        k++;
    }
}
