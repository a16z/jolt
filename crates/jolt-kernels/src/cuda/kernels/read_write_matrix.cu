#define RWM_MAX_COEFFS 2

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
    const unsigned int *__restrict__ seg_start, const unsigned int *__restrict__ seg_even_end,
    const unsigned int *__restrict__ seg_end, const unsigned int *__restrict__ seg_pair,
    const unsigned int *__restrict__ offsets, unsigned int segments,
    const u64 *__restrict__ challenge, unsigned int *__restrict__ out_rows,
    unsigned int *__restrict__ out_cols, u64 *__restrict__ out_val,
    u64 *__restrict__ out_prev, u64 *__restrict__ out_next, u64 *__restrict__ out_coeffs) {
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
