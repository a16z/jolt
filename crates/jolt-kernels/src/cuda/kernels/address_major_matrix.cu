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

extern "C" __global__ void amm_materialize_kernel(
    const unsigned int *__restrict__ rows, const unsigned int *__restrict__ cols,
    const u64 *__restrict__ val_coeff, const u64 *__restrict__ next_val,
    const u64 *__restrict__ coeffs, unsigned int coeff_width, unsigned int entries,
    unsigned int t_prime, u64 *__restrict__ ra, u64 *__restrict__ wa, u64 *__restrict__ val) {
    unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= entries) return;

    unsigned int row = rows[n];
    if (row >= t_prime) return;

    unsigned int col = cols[n];
    unsigned long long base = (unsigned long long)col * t_prime;
    unsigned int ra_lane = 0;
    unsigned int wa_lane = (coeff_width > 1) ? 1 : 0;

    u64 coeff[LIMBS];
    load4(coeffs + ((unsigned long long)n * coeff_width + ra_lane) * LIMBS, coeff);
    store4(ra + (base + row) * LIMBS, coeff);
    load4(coeffs + ((unsigned long long)n * coeff_width + wa_lane) * LIMBS, coeff);
    store4(wa + (base + row) * LIMBS, coeff);
    load4(val_coeff + (unsigned long long)n * LIMBS, coeff);
    store4(val + (base + row) * LIMBS, coeff);

    unsigned int fill_end = t_prime;
    if (n + 1 < entries && cols[n + 1] == col) {
        unsigned int next_row = rows[n + 1];
        fill_end = (next_row < t_prime) ? next_row : t_prime;
    }
    if (row + 1 < fill_end) {
        u64 carried[LIMBS];
        load4(next_val + (unsigned long long)n * LIMBS, carried);
        for (unsigned int r = row + 1; r < fill_end; r++) {
            store4(val + (base + r) * LIMBS, carried);
        }
    }
}

extern "C" __global__ void amm_message_kernel(
    const unsigned int *__restrict__ rows, const u64 *__restrict__ val_coeff,
    const u64 *__restrict__ next_val, const u64 *__restrict__ coeffs, unsigned int coeff_width,
    const unsigned int *__restrict__ seg_start, const unsigned int *__restrict__ seg_even_end,
    const unsigned int *__restrict__ seg_end, const unsigned int *__restrict__ seg_pair,
    unsigned int segments, const u64 *__restrict__ val_init, const u64 *__restrict__ inc,
    const u64 *__restrict__ eq, u64 *__restrict__ partials) {
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

        u64 even_cp[LIMBS], odd_cp[LIMBS];
        load4(val_init + (unsigned long long)(2 * pair) * LIMBS, even_cp);
        load4(val_init + (unsigned long long)(2 * pair + 1) * LIMBS, odd_cp);

        u64 inner[2][2 * PA_SLOTS];
        for (int lane = 0; lane < 2; lane++) pa_zero(inner[lane]);

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

            unsigned int row = take_even ? rows[i] : rows[j];
            u64 inc_eval[LIMBS], eq_eval[LIMBS];
            load4(inc + (unsigned long long)row * LIMBS, inc_eval);
            load4(eq + (unsigned long long)row * LIMBS, eq_eval);

            u64 ra_0[LIMBS], ra_2[LIMBS], wa_0[LIMBS], wa_2[LIMBS];
            u64 val_0[LIMBS], val_2[LIMBS];

            if (take_even && take_odd) {
                u64 e[LIMBS], o[LIMBS], twice[LIMBS];
                load4(coeffs + ((unsigned long long)i * coeff_width + ra_lane) * LIMBS, e);
                load4(coeffs + ((unsigned long long)j * coeff_width + ra_lane) * LIMBS, o);
                for (int l = 0; l < LIMBS; l++) ra_0[l] = e[l];
                fr_add(o, o, twice);
                fr_sub(twice, e, ra_2);
                load4(coeffs + ((unsigned long long)i * coeff_width + wa_lane) * LIMBS, e);
                load4(coeffs + ((unsigned long long)j * coeff_width + wa_lane) * LIMBS, o);
                for (int l = 0; l < LIMBS; l++) wa_0[l] = e[l];
                fr_add(o, o, twice);
                fr_sub(twice, e, wa_2);
                load4(val_coeff + (unsigned long long)i * LIMBS, e);
                load4(val_coeff + (unsigned long long)j * LIMBS, o);
                for (int l = 0; l < LIMBS; l++) val_0[l] = e[l];
                fr_add(o, o, twice);
                fr_sub(twice, e, val_2);
            } else if (take_even) {
                u64 e[LIMBS], zero[LIMBS] = {0, 0, 0, 0}, twice[LIMBS];
                load4(coeffs + ((unsigned long long)i * coeff_width + ra_lane) * LIMBS, e);
                for (int l = 0; l < LIMBS; l++) ra_0[l] = e[l];
                fr_sub(zero, e, ra_2);
                load4(coeffs + ((unsigned long long)i * coeff_width + wa_lane) * LIMBS, e);
                for (int l = 0; l < LIMBS; l++) wa_0[l] = e[l];
                fr_sub(zero, e, wa_2);
                load4(val_coeff + (unsigned long long)i * LIMBS, e);
                for (int l = 0; l < LIMBS; l++) val_0[l] = e[l];
                fr_add(odd_cp, odd_cp, twice);
                fr_sub(twice, e, val_2);
            } else {
                u64 o[LIMBS], twice[LIMBS];
                load4(coeffs + ((unsigned long long)j * coeff_width + ra_lane) * LIMBS, o);
                fr_add(o, o, ra_2);
                load4(coeffs + ((unsigned long long)j * coeff_width + wa_lane) * LIMBS, o);
                fr_add(o, o, wa_2);
                load4(val_coeff + (unsigned long long)j * LIMBS, o);
                for (int l = 0; l < LIMBS; l++) val_0[l] = even_cp[l];
                fr_add(o, o, twice);
                fr_sub(twice, even_cp, val_2);
            }

            u64 read[LIMBS], write[LIMBS], sum[LIMBS], bracket[LIMBS];
            if (take_even) {
                fr_mul(ra_0, val_0, read);
                fr_add(val_0, inc_eval, sum);
                fr_mul(wa_0, sum, write);
                fr_add(read, write, bracket);
                pa_fold_mul_accum(eq_eval, bracket, inner[0]);
            }
            fr_mul(ra_2, val_2, read);
            fr_add(val_2, inc_eval, sum);
            fr_mul(wa_2, sum, write);
            fr_add(read, write, bracket);
            pa_fold_mul_accum(eq_eval, bracket, inner[1]);

            if (in_main) {
                if (take_even) load4(next_val + (unsigned long long)i * LIMBS, even_cp);
                if (take_odd) load4(next_val + (unsigned long long)j * LIMBS, odd_cp);
            }

            if (take_even) i++;
            if (take_odd) j++;
        }

        for (int lane = 0; lane < 2; lane++) pa_finalize(inner[lane], acc[lane]);
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
