extern "C" __global__ void prefix_suffix_round_pairs3(
    u64 *__restrict__ out,
    const u64 *__restrict__ prefix0,
    const u64 *__restrict__ q0_0,
    const u64 *__restrict__ q1_0,
    const u64 *__restrict__ prefix1,
    const u64 *__restrict__ q0_1,
    const u64 *__restrict__ q1_1,
    const u64 *__restrict__ prefix2,
    const u64 *__restrict__ q0_2,
    const u64 *__restrict__ q1_2,
    unsigned int has_prefix,
    unsigned long half,
    unsigned long len
) {
    const u64 *prefix, *q0, *q1;
    if (blockIdx.y == 0) {
        prefix = prefix0; q0 = q0_0; q1 = q1_0;
    } else if (blockIdx.y == 1) {
        prefix = prefix1; q0 = q0_1; q1 = q1_1;
    } else {
        prefix = prefix2; q0 = q0_2; q1 = q1_2;
    }

    extern __shared__ u64 sdata[];
    u64 *acc = sdata + threadIdx.x * (3 * 4);
    for (int k = 0; k < 3 * 4; k++) acc[k] = 0;

    unsigned long row = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    if (row < half) {
        u64 q0r[4], q1r[4], q0h[4], q1h[4];
        load4(q0 + row * 4, q0r);
        load4(q1 + row * 4, q1r);
        load4(q0 + (row + half) * 4, q0h);
        load4(q1 + (row + half) * 4, q1h);

        u64 p0[4], p2[4];
        if (has_prefix) {
            u64 low[4], high[4], t2[4];
            load4(prefix + row * 4, low);
            load4(prefix + (row + half) * 4, high);
            for (int k = 0; k < 4; k++) p0[k] = low[k];
            fr_add(high, high, t2);
            fr_sub(t2, low, p2);
        } else {
            pc_u64_to_mont(1, p0);
            for (int k = 0; k < 4; k++) p2[k] = p0[k];
        }

        u64 prod[4];
        fr_mul(p0, q0r, prod);
        fr_add(prod, q1r, acc + 0 * 4);
        fr_mul(p2, q0r, prod);
        fr_add(prod, q1r, acc + 1 * 4);
        fr_mul(p2, q0h, prod);
        fr_add(prod, q1h, acc + 2 * 4);
    }
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            u64 *other = sdata + (threadIdx.x + stride) * (3 * 4);
            for (int e = 0; e < 3; e++) {
                u64 s[4];
                fr_add(acc + e * 4, other + e * 4, s);
                for (int k = 0; k < 4; k++) acc[e * 4 + k] = s[k];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        unsigned long block = (unsigned long)blockIdx.y * gridDim.x + blockIdx.x;
        for (int k = 0; k < 3 * 4; k++) out[block * (3 * 4) + k] = acc[k];
    }
}
