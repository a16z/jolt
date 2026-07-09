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
    (void)worker_banks;
    (void)weight;
    (void)lookup_index_lo;
    (void)lookup_index_hi;
    (void)cycle_list;
    (void)suffix_variants;
    (void)suffix_count;
    (void)suffix_len;
    (void)poly_len;
    (void)m;
    (void)num_workers;
}
