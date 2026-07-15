extern "C" __global__ void read_suffix_scatter(
    u64 *__restrict__ worker_banks,
    const u64 *__restrict__ weight,
    const u64 *__restrict__ lookup_index_lo,
    const u64 *__restrict__ lookup_index_hi,
    const unsigned int *__restrict__ cycle_list,
    const unsigned int *__restrict__ suffix_variants,
    unsigned long suffix_count,
    unsigned long suffix_len,
    unsigned long poly_len,
    unsigned long m,
    unsigned long num_workers
) {
    unsigned long worker = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    if (worker >= num_workers) return;

    u128 suffix_mask = (suffix_len >= 128)
        ? (u128)-1
        : (((u128)1 << suffix_len) - 1);
    unsigned long index_mask = poly_len - 1UL;

    for (unsigned long j = worker; j < m; j += num_workers) {
        unsigned long c = (unsigned long)cycle_list[j];
        u128 lookup_index = ((u128)lookup_index_hi[c] << 64) | (u128)lookup_index_lo[c];
        unsigned long index = ((unsigned long)(lookup_index >> suffix_len)) & index_mask;
        u128 suffix_bits = lookup_index & suffix_mask;

        u64 w[4];
        load4(weight + c * 4, w);

        for (unsigned long s = 0; s < suffix_count; s++) {
            u64 mle = suffix_mle_eval(suffix_bits, (unsigned int)suffix_len, suffix_variants[s]);
            if (mle == 0) continue;
            u64 *slot = worker_banks + ((s * poly_len + index) * num_workers + worker) * 4UL;
            if (mle == 1) {
                raf_add_inplace(slot, w);
            } else {
                u64 raw[4] = {mle, 0, 0, 0};
                u64 mont[4], contrib[4];
                raf_to_mont(raw, mont);
                fr_mul(w, mont, contrib);
                raf_add_inplace(slot, contrib);
            }
        }
    }
}
