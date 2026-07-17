extern "C" __global__ void instruction_lookup_q(
    u64 *__restrict__ q_out,
    const u64 *__restrict__ lookup_output,
    const u64 *__restrict__ left_lookup_operand,
    const u64 *__restrict__ right_lookup_operand_lo,
    const u64 *__restrict__ right_lookup_operand_hi,
    const u64 *__restrict__ left_instruction_input,
    const u64 *__restrict__ right_instruction_input_abs_lo,
    const u64 *__restrict__ right_instruction_input_abs_hi,
    const unsigned char *__restrict__ right_instruction_input_neg,
    const u64 *__restrict__ eq_suffix,
    const u64 *__restrict__ g1,
    const u64 *__restrict__ g2,
    const u64 *__restrict__ g3,
    const u64 *__restrict__ g4,
    unsigned long prefix_len,
    unsigned long suffix_len
) {
    unsigned long x_lo = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    if (x_lo >= prefix_len) return;

    u64 wg1[4], wg2[4], wg3[4], wg4[4];
    load4(g1, wg1);
    load4(g2, wg2);
    load4(g3, wg3);
    load4(g4, wg4);

    u64 q[4] = {0, 0, 0, 0};
    u64 mont[4], term[4], combined[4], tmp[4];

    for (unsigned long x_hi = 0; x_hi < suffix_len; x_hi++) {
        unsigned long idx = x_lo + x_hi * prefix_len;

        {
            u64 raw[4] = {lookup_output[idx], 0, 0, 0};
            raf_to_mont(raw, combined);
        }
        {
            u64 raw[4] = {left_lookup_operand[idx], 0, 0, 0};
            raf_to_mont(raw, mont);
            fr_mul(wg1, mont, term);
            fr_add(combined, term, tmp);
            for (int k = 0; k < 4; k++) combined[k] = tmp[k];
        }
        {
            u64 raw[4] = {right_lookup_operand_lo[idx], right_lookup_operand_hi[idx], 0, 0};
            raf_to_mont(raw, mont);
            fr_mul(wg2, mont, term);
            fr_add(combined, term, tmp);
            for (int k = 0; k < 4; k++) combined[k] = tmp[k];
        }
        {
            u64 raw[4] = {left_instruction_input[idx], 0, 0, 0};
            raf_to_mont(raw, mont);
            fr_mul(wg3, mont, term);
            fr_add(combined, term, tmp);
            for (int k = 0; k < 4; k++) combined[k] = tmp[k];
        }
        {
            u64 raw[4] = {right_instruction_input_abs_lo[idx], right_instruction_input_abs_hi[idx], 0, 0};
            raf_to_mont(raw, mont);
            if (right_instruction_input_neg[idx]) {
                u64 zero[4] = {0, 0, 0, 0};
                fr_sub(zero, mont, tmp);
                for (int k = 0; k < 4; k++) mont[k] = tmp[k];
            }
            fr_mul(wg4, mont, term);
            fr_add(combined, term, tmp);
            for (int k = 0; k < 4; k++) combined[k] = tmp[k];
        }

        load4(eq_suffix + x_hi * 4, mont);
        fr_mul(mont, combined, term);
        fr_add(q, term, tmp);
        for (int k = 0; k < 4; k++) q[k] = tmp[k];
    }

    store4(q_out + x_lo * 4, q);
}
