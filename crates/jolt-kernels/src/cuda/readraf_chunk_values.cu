extern "C" __global__ void readraf_chunk_values(
    unsigned short *__restrict__ out,
    const u64 *__restrict__ lookup_index_lo,
    const u64 *__restrict__ lookup_index_hi,
    unsigned long num_chunks,
    unsigned long chunk_bits,
    unsigned long m
) {
    unsigned long c = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= m) return;

    u128 lookup_index = ((u128)lookup_index_hi[c] << 64) | (u128)lookup_index_lo[c];
    u128 mask = ((u128)1 << chunk_bits) - 1;

    for (unsigned long chunk = 0; chunk < num_chunks; chunk++) {
        unsigned long shift = 128UL - (chunk + 1) * chunk_bits;
        out[chunk * m + c] = (unsigned short)((lookup_index >> shift) & mask);
    }
}
