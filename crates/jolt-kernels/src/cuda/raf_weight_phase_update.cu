extern "C" __global__ void raf_weight_phase_update(
    u64 *__restrict__ weight,
    const u64 *__restrict__ eq_table,
    const u64 *__restrict__ lookup_index_lo,
    const u64 *__restrict__ lookup_index_hi,
    unsigned long shift,
    unsigned long mask,
    unsigned long trace_len
) {
    unsigned long c = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= trace_len) return;

    u128 lookup_index = ((u128)lookup_index_hi[c] << 64) | (u128)lookup_index_lo[c];
    unsigned long slot = ((unsigned long)(lookup_index >> shift)) & mask;

    u64 w[4], e[4], out[4];
    load4(weight + c * 4, w);
    load4(eq_table + slot * 4, e);
    fr_mul(w, e, out);
    store4(weight + c * 4, out);
}
