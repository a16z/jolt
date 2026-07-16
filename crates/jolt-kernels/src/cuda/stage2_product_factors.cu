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
}
