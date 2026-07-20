extern "C" __global__ void readraf_chunk_values(
    unsigned short *__restrict__ out,
    const u64 *__restrict__ lookup_index_lo,
    const u64 *__restrict__ lookup_index_hi,
    unsigned long num_chunks,
    unsigned long chunk_bits,
    unsigned long m
) {
    (void)out;
    (void)lookup_index_lo;
    (void)lookup_index_hi;
    (void)num_chunks;
    (void)chunk_bits;
    (void)m;
}
