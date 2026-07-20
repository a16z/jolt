extern "C" __global__ void readraf_combined(
    u64 *__restrict__ out,
    const unsigned int *__restrict__ table_index,
    const unsigned char *__restrict__ is_interleaved,
    const u64 *__restrict__ table_values,
    const u64 *__restrict__ raf_pair,
    unsigned long m
) {
    (void)out;
    (void)table_index;
    (void)is_interleaved;
    (void)table_values;
    (void)raf_pair;
    (void)m;
}
