__device__ __forceinline__ void so_split_eq(const u64 *__restrict__ e_in,
                                            const u64 *__restrict__ e_out, unsigned int in_bits,
                                            unsigned long long index, u64 *out) {
    u64 inner[LIMBS], outer[LIMBS];
    load4(e_in + (index & ((1ull << in_bits) - 1ull)) * LIMBS, inner);
    load4(e_out + (index >> in_bits) * LIMBS, outer);
    fr_mul(inner, outer, out);
}

__device__ __forceinline__ void so_block_reduce(u64 *scratch, unsigned int lanes,
                                                const u64 acc[][LIMBS], u64 *partials) {
    unsigned int tid = threadIdx.x;
    for (unsigned int lane = 0; lane < lanes; lane++) {
        store4(scratch + tid * LIMBS, acc[lane]);
        __syncthreads();
        for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                u64 a[LIMBS], b[LIMBS], s[LIMBS];
                load4(scratch + tid * LIMBS, a);
                load4(scratch + (tid + stride) * LIMBS, b);
                fr_add(a, b, s);
                store4(scratch + tid * LIMBS, s);
            }
            __syncthreads();
        }
        if (tid == 0) {
            u64 total[LIMBS];
            load4(scratch, total);
            store4(partials + ((unsigned long long)lane * gridDim.x + blockIdx.x) * LIMBS, total);
        }
        __syncthreads();
    }
}

extern "C" __global__ void gruen_pair_message_kernel(const u64 *__restrict__ az,
                                             const u64 *__restrict__ bz,
                                             const u64 *__restrict__ e_in,
                                             const u64 *__restrict__ e_out, unsigned int e_in_len,
                                             unsigned int in_bits, unsigned int half,
                                             unsigned int strip, u64 *__restrict__ partials) {
    extern __shared__ u64 scratch[];
    unsigned long long base =
        ((unsigned long long)blockIdx.x * blockDim.x + threadIdx.x) * (unsigned long long)strip;

    u64 acc[3][LIMBS];
    for (int lane = 0; lane < 3; lane++)
        for (int l = 0; l < LIMBS; l++) acc[lane][l] = 0;

    for (unsigned int step = 0; step < strip; step++) {
        unsigned long long pair = base + step;
        if (pair >= half) break;

        u64 az0[LIMBS], az1[LIMBS], bz0[LIMBS], bz1[LIMBS];
        load4(az + (2 * pair) * LIMBS, az0);
        load4(az + (2 * pair + 1) * LIMBS, az1);
        load4(bz + (2 * pair) * LIMBS, bz0);
        load4(bz + (2 * pair + 1) * LIMBS, bz1);

        u64 weight[LIMBS];
        if (e_in_len <= 1) {
            load4(e_out + pair * LIMBS, weight);
        } else {
            so_split_eq(e_in, e_out, in_bits, pair, weight);
        }

        u64 product[LIMBS], weighted[LIMBS], sum[LIMBS];
        fr_mul(az0, bz0, product);
        fr_mul(product, weight, weighted);
        fr_add(acc[0], weighted, sum);
        for (int l = 0; l < LIMBS; l++) acc[0][l] = sum[l];

        fr_mul(az1, bz1, product);
        fr_mul(product, weight, weighted);
        fr_add(acc[2], weighted, sum);
        for (int l = 0; l < LIMBS; l++) acc[2][l] = sum[l];

        u64 az_inf[LIMBS], bz_inf[LIMBS];
        fr_sub(az1, az0, az_inf);
        fr_sub(bz1, bz0, bz_inf);
        fr_mul(az_inf, bz_inf, product);
        fr_mul(product, weight, weighted);
        fr_add(acc[1], weighted, sum);
        for (int l = 0; l < LIMBS; l++) acc[1][l] = sum[l];
    }

    so_block_reduce(scratch, 3, acc, partials);
}
