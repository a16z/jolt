extern "C" __global__ void lt_double(
    u64 *__restrict__ out,
    const u64 *__restrict__ in,
    const u64 *__restrict__ challenge,
    unsigned long size_in
) {
    unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size_in) {
        u64 left[4], c[4], right[4], new_left[4], t[4];
        load4(in + i * 4, left);
        load4(challenge, c);
        fr_mul(left, c, right);
        fr_sub(c, right, t);
        fr_add(left, t, new_left);
        store4(out + i * 4, new_left);
        store4(out + (size_in + i) * 4, right);
    }
}
