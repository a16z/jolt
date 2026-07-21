extern "C" __global__ void prefix_suffix_round_pairs3(
    u64 *__restrict__ out,
    const u64 *__restrict__ prefix0,
    const u64 *__restrict__ q0_0,
    const u64 *__restrict__ q1_0,
    const u64 *__restrict__ prefix1,
    const u64 *__restrict__ q0_1,
    const u64 *__restrict__ q1_1,
    const u64 *__restrict__ prefix2,
    const u64 *__restrict__ q0_2,
    const u64 *__restrict__ q1_2,
    unsigned int has_prefix,
    unsigned long half,
    unsigned long len
) {
    (void)out;
    (void)prefix0; (void)q0_0; (void)q1_0;
    (void)prefix1; (void)q0_1; (void)q1_1;
    (void)prefix2; (void)q0_2; (void)q1_2;
    (void)has_prefix;
    (void)half;
    (void)len;
}
