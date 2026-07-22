extern "C" __global__ void add_scalar_at(
    u64 *__restrict__ a,
    const u64 *__restrict__ scalar,
    unsigned long index
) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        u64 cur[4], s[4], out[4];
        load4(a + index * 4, cur);
        load4(scalar, s);
        fr_add(cur, s, out);
        store4(a + index * 4, out);
    }
}
