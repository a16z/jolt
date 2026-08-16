extern "C" __global__ void brr_coefficient_kernel(
    const u64 *__restrict__ e_in, const u64 *__restrict__ e_out,
    const u64 *__restrict__ weights, const u64 *__restrict__ entry, u64 *__restrict__ out,
    unsigned int stages, unsigned int e_in_len, unsigned int e_out_len, unsigned int in_bits,
    unsigned int len) {
    unsigned long long j = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= len) return;

    unsigned long long lo = j & ((1ull << in_bits) - 1ull);
    unsigned long long hi = j >> in_bits;

    u64 acc[LIMBS];
    for (int l = 0; l < LIMBS; l++) acc[l] = 0;
    for (unsigned int s = 0; s < stages; s++) {
        u64 inner[LIMBS], outer[LIMBS], eq[LIMBS], weight[LIMBS], term[LIMBS], sum[LIMBS];
        load4(e_in + ((unsigned long long)s * e_in_len + lo) * LIMBS, inner);
        load4(e_out + ((unsigned long long)s * e_out_len + hi) * LIMBS, outer);
        fr_mul(inner, outer, eq);
        load4(weights + (unsigned long long)s * LIMBS, weight);
        fr_mul(eq, weight, term);
        fr_add(acc, term, sum);
        for (int l = 0; l < LIMBS; l++) acc[l] = sum[l];
    }
    if (j == 0) {
        u64 boundary[LIMBS], sum[LIMBS];
        load4(entry, boundary);
        fr_add(acc, boundary, sum);
        for (int l = 0; l < LIMBS; l++) acc[l] = sum[l];
    }
    store4(out + j * LIMBS, acc);
}

extern "C" __global__ void brr_gather_kernel(const unsigned int *__restrict__ packed,
                                             const u64 *__restrict__ tables,
                                             unsigned int addresses, unsigned int slots,
                                             unsigned int chunk_bits, unsigned int polys,
                                             u64 *__restrict__ out, unsigned int len) {
    unsigned int p = blockIdx.y;
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= len) return;

    unsigned int shift = chunk_bits * (polys - 1 - p);
    unsigned int mask = (1u << chunk_bits) - 1u;
    const u64 *poly_tables = tables + (unsigned long long)p * slots * addresses * LIMBS;

    u64 value[LIMBS];
    rrv_sparse_coeff(packed, poly_tables, addresses, slots, shift, mask,
                     (unsigned long long)j * slots, value);
    store4(out + ((unsigned long long)p * len + j) * LIMBS, value);
}

__device__ __forceinline__ void brr_accumulate(unsigned int lanes,
                                               const u64 prod[RRV_MAX_LANES][LIMBS],
                                               u64 acc[RRV_MAX_LANES][LIMBS]) {
    for (unsigned int lane = 0; lane < lanes; lane++) {
        u64 sum[LIMBS];
        fr_add(acc[lane], prod[lane], sum);
        for (int l = 0; l < LIMBS; l++) acc[lane][l] = sum[l];
    }
}

extern "C" __global__ void brr_message_sparse_kernel(
    const unsigned int *__restrict__ packed, const u64 *__restrict__ tables,
    const u64 *__restrict__ coefficient, unsigned int addresses, unsigned int slots,
    unsigned int chunk_bits, unsigned int polys, unsigned int lanes, unsigned int half,
    unsigned int strip, u64 *__restrict__ partials) {
    extern __shared__ u64 scratch[];
    unsigned long long base =
        ((unsigned long long)blockIdx.x * blockDim.x + threadIdx.x) * (unsigned long long)strip;

    u64 acc[RRV_MAX_LANES][LIMBS];
    for (unsigned int lane = 0; lane < lanes; lane++)
        for (int l = 0; l < LIMBS; l++) acc[lane][l] = 0;

    unsigned int mask = (1u << chunk_bits) - 1u;
    for (unsigned int step = 0; step < strip; step++) {
        unsigned long long g = base + step;
        if (g >= half) break;

        unsigned long long cycle_even = (unsigned long long)(2 * g) * slots;
        unsigned long long cycle_odd = cycle_even + slots;

        u64 prod[RRV_MAX_LANES][LIMBS];
        u64 c0[LIMBS], c1[LIMBS];
        load4(coefficient + (unsigned long long)(2 * g) * LIMBS, c0);
        load4(coefficient + (unsigned long long)(2 * g + 1) * LIMBS, c1);
        rrv_fold_pair(c0, c1, lanes, true, prod);

        for (unsigned int p = 0; p < polys; p++) {
            unsigned int shift = chunk_bits * (polys - 1 - p);
            const u64 *poly_tables = tables + (unsigned long long)p * slots * addresses * LIMBS;
            u64 p0[LIMBS], p1[LIMBS];
            rrv_sparse_coeff(packed, poly_tables, addresses, slots, shift, mask, cycle_even, p0);
            rrv_sparse_coeff(packed, poly_tables, addresses, slots, shift, mask, cycle_odd, p1);
            rrv_fold_pair(p0, p1, lanes, false, prod);
        }

        brr_accumulate(lanes, prod, acc);
    }

    lane_block_reduce(scratch, lanes, acc, partials);
}

extern "C" __global__ void brr_message_dense_kernel(
    const u64 *__restrict__ dense, const u64 *__restrict__ coefficient, unsigned int polys,
    unsigned int lanes, unsigned int half, unsigned int strip, u64 *__restrict__ partials) {
    extern __shared__ u64 scratch[];
    unsigned long long base =
        ((unsigned long long)blockIdx.x * blockDim.x + threadIdx.x) * (unsigned long long)strip;

    u64 acc[RRV_MAX_LANES][LIMBS];
    for (unsigned int lane = 0; lane < lanes; lane++)
        for (int l = 0; l < LIMBS; l++) acc[lane][l] = 0;

    for (unsigned int step = 0; step < strip; step++) {
        unsigned long long g = base + step;
        if (g >= half) break;

        u64 prod[RRV_MAX_LANES][LIMBS];
        u64 c0[LIMBS], c1[LIMBS];
        load4(coefficient + (unsigned long long)(2 * g) * LIMBS, c0);
        load4(coefficient + (unsigned long long)(2 * g + 1) * LIMBS, c1);
        rrv_fold_pair(c0, c1, lanes, true, prod);

        for (unsigned int p = 0; p < polys; p++) {
            const u64 *row = dense + (unsigned long long)p * 2 * half * LIMBS;
            u64 p0[LIMBS], p1[LIMBS];
            load4(row + (unsigned long long)(2 * g) * LIMBS, p0);
            load4(row + (unsigned long long)(2 * g + 1) * LIMBS, p1);
            rrv_fold_pair(p0, p1, lanes, false, prod);
        }

        brr_accumulate(lanes, prod, acc);
    }

    lane_block_reduce(scratch, lanes, acc, partials);
}
