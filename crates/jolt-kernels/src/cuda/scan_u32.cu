extern "C" __global__ void scan_u32_block(
    unsigned int *__restrict__ out,
    unsigned int *__restrict__ block_sums,
    const unsigned int *__restrict__ in,
    unsigned long n
) {
}

extern "C" __global__ void scan_u32_add_offsets(
    unsigned int *__restrict__ out,
    const unsigned int *__restrict__ block_offsets,
    unsigned long n
) {
}
