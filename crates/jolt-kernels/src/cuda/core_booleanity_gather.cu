extern "C" __global__ void core_booleanity_gather(
    u64 *__restrict__ out,
    const u64 *__restrict__ tables,
    const u64 *__restrict__ present_mask,
    const unsigned char *__restrict__ values,
    unsigned long num_polys,
    unsigned long chunk_domain,
    unsigned long rows,
    unsigned long poly_stride
) {
    unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long total = num_polys * rows;
    if (i < total) {
        unsigned long poly = i / rows;
        unsigned long row = i % rows;
        u64 mask = present_mask[row];
        if ((mask >> poly) & 1ull) {
            unsigned long value = values[row * poly_stride + poly];
            const u64 *entry = tables + (poly * chunk_domain + value) * 4;
            for (int k = 0; k < 4; k++) out[i * 4 + k] = entry[k];
        } else {
            for (int k = 0; k < 4; k++) out[i * 4 + k] = 0;
        }
    }
}
