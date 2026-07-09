extern "C" __global__ void raf_weight_phase_update(
    u64 *__restrict__ weight,
    const u64 *__restrict__ eq_table,
    const u64 *__restrict__ lookup_index_lo,
    const u64 *__restrict__ lookup_index_hi,
    unsigned long shift,
    unsigned long mask,
    unsigned long trace_len
) {
    (void)weight;
    (void)eq_table;
    (void)lookup_index_lo;
    (void)lookup_index_hi;
    (void)shift;
    (void)mask;
    (void)trace_len;
}
