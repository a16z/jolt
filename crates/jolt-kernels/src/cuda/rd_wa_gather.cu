extern "C" __global__ void rd_wa_gather(
    u64 *__restrict__ out,
    const u64 *__restrict__ address_eq,
    const short *__restrict__ addresses,
    unsigned long trace_len,
    unsigned long register_count
) {
    unsigned long c = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    if (c < trace_len) {
        short addr = addresses[c];
        if (addr >= 0 && (unsigned long)addr < register_count) {
            u64 v[4];
            load4(address_eq + (unsigned long)addr * 4, v);
            store4(out + c * 4, v);
        } else {
            for (int k = 0; k < 4; k++) out[c * 4 + k] = 0;
        }
    }
}
