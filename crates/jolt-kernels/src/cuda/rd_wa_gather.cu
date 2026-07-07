extern "C" __global__ void rd_wa_gather(
    u64 *__restrict__ out,
    const u64 *__restrict__ address_eq,
    const short *__restrict__ addresses,
    unsigned long trace_len,
    unsigned long register_count
) {
    unsigned long c = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    if (c < trace_len) {
        // STUB: body implemented after review. Intended:
        //   out[c] = (addresses[c] >= 0 && addresses[c] < register_count)
        //            ? address_eq[addresses[c]] : 0
        for (int k = 0; k < 4; k++) out[c * 4 + k] = 0;
    }
}
