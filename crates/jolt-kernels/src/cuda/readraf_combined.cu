extern "C" __global__ void readraf_combined(
    u64 *__restrict__ out,
    const unsigned int *__restrict__ table_index,
    const unsigned char *__restrict__ is_interleaved,
    const u64 *__restrict__ table_values,
    const u64 *__restrict__ raf_pair,
    unsigned long m
) {
    unsigned long c = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= m) return;

    u64 table_value[4];
    unsigned int ti = table_index[c];
    if (ti == 0xffffffffu) {
        table_value[0] = 0; table_value[1] = 0; table_value[2] = 0; table_value[3] = 0;
    } else {
        load4(table_values + (unsigned long)ti * 4, table_value);
    }

    u64 raf_value[4];
    load4(raf_pair + (is_interleaved[c] ? 4UL : 0UL), raf_value);

    u64 result[4];
    fr_add(table_value, raf_value, result);
    store4(out + c * 4, result);
}
