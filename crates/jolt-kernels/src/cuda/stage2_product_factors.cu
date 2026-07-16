extern "C" __global__ void stage2_product_factors(
    u64 *__restrict__ left_out,
    u64 *__restrict__ right_out,
    const u64 *__restrict__ left_input,
    const u64 *__restrict__ sblo,
    const unsigned char *__restrict__ jump,
    const u64 *__restrict__ ri_abs_lo,
    const u64 *__restrict__ ri_abs_hi,
    const unsigned char *__restrict__ ri_neg,
    const unsigned char *__restrict__ should_branch,
    const unsigned char *__restrict__ not_next_noop,
    const u64 *__restrict__ w0,
    const u64 *__restrict__ w1,
    const u64 *__restrict__ w2,
    unsigned long n
) {
    unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    u64 wc0[4], wc1[4], wc2[4];
    load4(w0, wc0);
    load4(w1, wc1);
    load4(w2, wc2);

    // left = w0*mont(left_input) + w1*mont(sblo) + (jump ? w2 : 0)
    u64 col[4], mont[4], term[4], left[4], t[4];
    {
        u64 raw[4] = {left_input[i], 0, 0, 0};
        raf_to_mont(raw, mont);
        fr_mul(wc0, mont, left);
    }
    {
        u64 raw[4] = {sblo[i], 0, 0, 0};
        raf_to_mont(raw, mont);
        fr_mul(wc1, mont, term);
        fr_add(left, term, t);
        for (int k = 0; k < 4; k++) left[k] = t[k];
    }
    if (jump[i]) {
        fr_add(left, wc2, t);
        for (int k = 0; k < 4; k++) left[k] = t[k];
    }
    store4(left_out + i * 4, left);

    // right = w0*mont(|ri|)*(sign) + (should_branch ? w1 : 0) + (not_next_noop ? w2 : 0)
    u64 right[4];
    {
        u64 raw[4] = {ri_abs_lo[i], ri_abs_hi[i], 0, 0};
        raf_to_mont(raw, mont);
        if (ri_neg[i]) {
            u64 zero[4] = {0, 0, 0, 0};
            fr_sub(zero, mont, col);
            for (int k = 0; k < 4; k++) mont[k] = col[k];
        }
        fr_mul(wc0, mont, right);
    }
    if (should_branch[i]) {
        fr_add(right, wc1, t);
        for (int k = 0; k < 4; k++) right[k] = t[k];
    }
    if (not_next_noop[i]) {
        fr_add(right, wc2, t);
        for (int k = 0; k < 4; k++) right[k] = t[k];
    }
    store4(right_out + i * 4, right);
}
