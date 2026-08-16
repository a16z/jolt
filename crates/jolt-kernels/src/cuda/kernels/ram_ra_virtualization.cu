#define RRV_MAX_LANES 8
#define RRV_COLD 0xFFFFFFFFu

__device__ __forceinline__ bool rrv_any_hot(const unsigned int *__restrict__ packed,
                                            unsigned long long cycle_base, unsigned int slots) {
    for (unsigned int s = 0; s < slots; s++) {
        if (packed[cycle_base + s] != RRV_COLD) return true;
    }
    return false;
}

__device__ __forceinline__ void rrv_sparse_coeff(const unsigned int *__restrict__ packed,
                                                 const u64 *__restrict__ poly_tables,
                                                 unsigned int addresses, unsigned int slots,
                                                 unsigned int shift, unsigned int mask,
                                                 unsigned long long cycle_base, u64 *out) {
    for (int l = 0; l < LIMBS; l++) out[l] = 0;
    for (unsigned int s = 0; s < slots; s++) {
        unsigned int packed_address = packed[cycle_base + s];
        if (packed_address == RRV_COLD) continue;
        unsigned int a = (packed_address >> shift) & mask;
        u64 entry[LIMBS], sum[LIMBS];
        load4(poly_tables + ((unsigned long long)s * addresses + a) * LIMBS, entry);
        fr_add(out, entry, sum);
        for (int l = 0; l < LIMBS; l++) out[l] = sum[l];
    }
}

extern "C" __global__ void rrv_gather_kernel(const unsigned int *__restrict__ packed,
                                             const u64 *__restrict__ tables,
                                             u64 *const *__restrict__ out, unsigned int addresses,
                                             unsigned int slots, unsigned int chunk_bits,
                                             unsigned int committed, unsigned int len) {
    unsigned int p = blockIdx.y;
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= len) return;

    unsigned int shift = chunk_bits * (committed - 1 - p);
    unsigned int mask = (1u << chunk_bits) - 1u;
    const u64 *poly_tables = tables + (unsigned long long)p * slots * addresses * LIMBS;

    u64 value[LIMBS];
    rrv_sparse_coeff(packed, poly_tables, addresses, slots, shift, mask,
                     (unsigned long long)j * slots, value);
    store4(out[p] + (unsigned long long)j * LIMBS, value);
}

__device__ __forceinline__ void rrv_fold_pair(const u64 *p0, const u64 *p1, unsigned int lanes,
                                              bool first, u64 prod[RRV_MAX_LANES][LIMBS]) {
    u64 delta[LIMBS], cur[LIMBS], scaled[LIMBS];
    fr_sub(p1, p0, delta);
    for (int l = 0; l < LIMBS; l++) cur[l] = p1[l];

    for (unsigned int k = 0; k + 1 < lanes; k++) {
        if (k > 0) {
            u64 next[LIMBS];
            fr_add(cur, delta, next);
            for (int l = 0; l < LIMBS; l++) cur[l] = next[l];
        }
        if (first) {
            for (int l = 0; l < LIMBS; l++) prod[k][l] = cur[l];
        } else {
            fr_mul(prod[k], cur, scaled);
            for (int l = 0; l < LIMBS; l++) prod[k][l] = scaled[l];
        }
    }

    unsigned int infinity = lanes - 1;
    if (first) {
        for (int l = 0; l < LIMBS; l++) prod[infinity][l] = delta[l];
    } else {
        fr_mul(prod[infinity], delta, scaled);
        for (int l = 0; l < LIMBS; l++) prod[infinity][l] = scaled[l];
    }
}

__device__ __forceinline__ void rrv_pad_degree(unsigned int polys, unsigned int lanes,
                                               u64 prod[RRV_MAX_LANES][LIMBS]) {
    if (polys < lanes) {
        for (int l = 0; l < LIMBS; l++) prod[lanes - 1][l] = 0;
    }
}

extern "C" __global__ void rrv_message_sparse_kernel(
    const unsigned int *__restrict__ packed, const u64 *__restrict__ tables,
    unsigned int addresses, unsigned int slots, unsigned int chunk_bits, unsigned int polys,
    unsigned int lanes, unsigned int half, const u64 *__restrict__ e_in, unsigned int e_in_len,
    const u64 *__restrict__ e_out, unsigned int num_x_in_bits, u64 *__restrict__ partials) {
    extern __shared__ u64 scratch[];
    unsigned long long g = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;

    u64 acc[RRV_MAX_LANES][LIMBS];
    for (unsigned int lane = 0; lane < lanes; lane++)
        for (int l = 0; l < LIMBS; l++) acc[lane][l] = 0;

    if (g < half) {
        unsigned int mask = (1u << chunk_bits) - 1u;
        unsigned long long cycle_even = (unsigned long long)(2 * g) * slots;
        unsigned long long cycle_odd = cycle_even + slots;

        if (rrv_any_hot(packed, cycle_even, slots) || rrv_any_hot(packed, cycle_odd, slots)) {
            u64 prod[RRV_MAX_LANES][LIMBS];
            for (unsigned int p = 0; p < polys; p++) {
                unsigned int shift = chunk_bits * (polys - 1 - p);
                const u64 *poly_tables =
                    tables + (unsigned long long)p * slots * addresses * LIMBS;
                u64 p0[LIMBS], p1[LIMBS];
                rrv_sparse_coeff(packed, poly_tables, addresses, slots, shift, mask, cycle_even,
                                 p0);
                rrv_sparse_coeff(packed, poly_tables, addresses, slots, shift, mask, cycle_odd, p1);
                rrv_fold_pair(p0, p1, lanes, p == 0, prod);
            }
            rrv_pad_degree(polys, lanes, prod);

            u64 combined[LIMBS];
            eq_split_weight(e_in, e_in_len, e_out, num_x_in_bits, g, combined);
            for (unsigned int lane = 0; lane < lanes; lane++) {
                fr_mul(prod[lane], combined, acc[lane]);
            }
        }
    }

    lane_block_reduce(scratch, lanes, acc, partials);
}

extern "C" __global__ void rrv_message_dense_kernel(
    const u64 *const *__restrict__ dense, unsigned int polys, unsigned int lanes,
    unsigned int half, const u64 *__restrict__ e_in, unsigned int e_in_len,
    const u64 *__restrict__ e_out, unsigned int num_x_in_bits, u64 *__restrict__ partials) {
    extern __shared__ u64 scratch[];
    unsigned long long g = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;

    u64 acc[RRV_MAX_LANES][LIMBS];
    for (unsigned int lane = 0; lane < lanes; lane++)
        for (int l = 0; l < LIMBS; l++) acc[lane][l] = 0;

    if (g < half) {
        u64 prod[RRV_MAX_LANES][LIMBS];
        for (unsigned int p = 0; p < polys; p++) {
            const u64 *table = dense[p];
            u64 p0[LIMBS], p1[LIMBS];
            load4(table + (2 * g) * LIMBS, p0);
            load4(table + (2 * g + 1) * LIMBS, p1);
            rrv_fold_pair(p0, p1, lanes, p == 0, prod);
        }
        rrv_pad_degree(polys, lanes, prod);

        u64 combined[LIMBS];
        eq_split_weight(e_in, e_in_len, e_out, num_x_in_bits, g, combined);
        for (unsigned int lane = 0; lane < lanes; lane++) {
            fr_mul(prod[lane], combined, acc[lane]);
        }
    }

    lane_block_reduce(scratch, lanes, acc, partials);
}
