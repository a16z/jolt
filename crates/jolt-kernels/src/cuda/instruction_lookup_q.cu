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
}
