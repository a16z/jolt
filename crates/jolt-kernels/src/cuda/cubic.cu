__device__ __forceinline__ void cubic_coeffs(
    const u64 *eq0, const u64 *eq1,
    const u64 *az0, const u64 *az1,
    const u64 *bz0, const u64 *bz1,
    u64 *c
) {
    u64 eqd[4], azd[4], bzd[4];
    fr_sub(eq1, eq0, eqd);
    fr_sub(az1, az0, azd);
    fr_sub(bz1, bz0, bzd);

    u64 az0bz0[4], azdbz0[4], az0bzd[4], azdbzd[4];
    fr_mul(az0, bz0, az0bz0);
    fr_mul(azd, bz0, azdbz0);
    fr_mul(az0, bzd, az0bzd);
    fr_mul(azd, bzd, azdbzd);

    u64 t[4], s[4];

    fr_mul(eq0, az0bz0, c + 0);

    fr_mul(eqd, az0bz0, s);
    fr_mul(eq0, azdbz0, t);
    fr_add(s, t, s);
    fr_mul(eq0, az0bzd, t);
    fr_add(s, t, c + 4);

    fr_mul(eqd, azdbz0, s);
    fr_mul(eqd, az0bzd, t);
    fr_add(s, t, s);
    fr_mul(eq0, azdbzd, t);
    fr_add(s, t, c + 8);

    fr_mul(eqd, azdbzd, c + 12);
}

extern "C" __global__ void cubic_pairs(
    u64 *__restrict__ out,
    const u64 *__restrict__ eq,
    const u64 *__restrict__ az,
    const u64 *__restrict__ bz,
    unsigned long pairs
) {
    extern __shared__ u64 sdata[];
    u64 *acc = sdata + threadIdx.x * 16;

    unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < pairs) {
        cubic_coeffs(
            eq + (i * 2) * 4, eq + (i * 2 + 1) * 4,
            az + (i * 2) * 4, az + (i * 2 + 1) * 4,
            bz + (i * 2) * 4, bz + (i * 2 + 1) * 4,
            acc
        );
    } else {
        for (int k = 0; k < 16; k++) acc[k] = 0;
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            cubic_tuple_add(acc, sdata + (threadIdx.x + s) * 16);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        for (int k = 0; k < 16; k++) out[blockIdx.x * 16 + k] = acc[k];
    }
}

extern "C" __global__ void cubic_tuple_reduce(
    u64 *__restrict__ out,
    const u64 *__restrict__ in,
    unsigned long n
) {
    extern __shared__ u64 sdata[];
    u64 *acc = sdata + threadIdx.x * 16;

    unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        for (int k = 0; k < 16; k++) acc[k] = in[i * 16 + k];
    } else {
        for (int k = 0; k < 16; k++) acc[k] = 0;
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            cubic_tuple_add(acc, sdata + (threadIdx.x + s) * 16);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        for (int k = 0; k < 16; k++) out[blockIdx.x * 16 + k] = acc[k];
    }
}
