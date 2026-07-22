extern "C" __global__ void gather8_materialize(
    u64 *__restrict__ out,
    const u64 *__restrict__ g0,
    const u64 *__restrict__ g1,
    const u64 *__restrict__ g2,
    const u64 *__restrict__ g3,
    const u64 *__restrict__ g4,
    const u64 *__restrict__ g5,
    const u64 *__restrict__ g6,
    const u64 *__restrict__ g7,
    const short *__restrict__ indices,
    unsigned long num_chunks,
    unsigned long table_len,
    unsigned long new_len
) {
    unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long total = num_chunks * new_len;
    if (i < total) {
        unsigned long chunk = i / new_len;
        unsigned long index = i % new_len;
        const u64 *groups[8] = { g0, g1, g2, g3, g4, g5, g6, g7 };
        const short *row = indices + chunk * (new_len * 8) + index * 8;
        u64 acc[4];
        for (int k = 0; k < 4; k++) acc[k] = 0;
        for (int offset = 0; offset < 8; offset++) {
            short idx = row[offset];
            if (idx >= 0) {
                const u64 *table = groups[offset] + chunk * table_len * 4;
                u64 entry[4], sum[4];
                load4(table + (unsigned long)idx * 4, entry);
                fr_add(acc, entry, sum);
                for (int k = 0; k < 4; k++) acc[k] = sum[k];
            }
        }
        store4(out + i * 4, acc);
    }
}
