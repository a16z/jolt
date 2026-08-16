#define BRC_COLD 0xFFFFFFFFu
#define BRC_INSTRUCTION 0u
#define BRC_BYTECODE 1u
#define BRC_RAM 2u

__device__ __forceinline__ unsigned int brc_chunk_u128(u64 lo, u64 hi, unsigned int shift,
                                                      unsigned int mask) {
    u64 value;
    if (shift >= 64) {
        value = hi >> (shift - 64);
    } else {
        value = lo >> shift;
        if (shift > 0) value |= hi << (64 - shift);
    }
    return (unsigned int)(value & (u64)mask);
}

__device__ __forceinline__ void brc_family(unsigned int p, unsigned int instruction_polys,
                                           unsigned int bytecode_polys, unsigned int ram_polys,
                                           unsigned int chunk_bits, unsigned int *family,
                                           unsigned int *shift) {
    if (p < instruction_polys) {
        *family = BRC_INSTRUCTION;
        *shift = chunk_bits * (instruction_polys - 1 - p);
        return;
    }
    if (p < instruction_polys + bytecode_polys) {
        unsigned int local = p - instruction_polys;
        *family = BRC_BYTECODE;
        *shift = chunk_bits * (bytecode_polys - 1 - local);
        return;
    }
    unsigned int local = p - instruction_polys - bytecode_polys;
    *family = BRC_RAM;
    *shift = chunk_bits * (ram_polys - 1 - local);
}

__device__ __forceinline__ bool brc_index(const u64 *__restrict__ lookup,
                                          const unsigned int *__restrict__ pc,
                                          const unsigned int *__restrict__ ram,
                                          unsigned int family, unsigned int shift,
                                          unsigned int mask, unsigned long long cycle,
                                          unsigned int *out) {
    if (family == BRC_INSTRUCTION) {
        *out = brc_chunk_u128(lookup[2 * cycle], lookup[2 * cycle + 1], shift, mask);
        return true;
    }
    unsigned int word = (family == BRC_BYTECODE) ? pc[cycle] : ram[cycle];
    if (word == BRC_COLD) return false;
    *out = (word >> shift) & mask;
    return true;
}

__device__ __forceinline__ void brc_sparse_coeff(
    const u64 *__restrict__ lookup, const unsigned int *__restrict__ pc,
    const unsigned int *__restrict__ ram, const u64 *__restrict__ poly_tables,
    unsigned int addresses, unsigned int slots, unsigned int family, unsigned int shift,
    unsigned int mask, unsigned long long cycle_base, u64 *out) {
    for (int l = 0; l < LIMBS; l++) out[l] = 0;
    for (unsigned int s = 0; s < slots; s++) {
        unsigned int a;
        if (!brc_index(lookup, pc, ram, family, shift, mask, cycle_base + s, &a)) continue;
        u64 entry[LIMBS], sum[LIMBS];
        load4(poly_tables + ((unsigned long long)s * addresses + a) * LIMBS, entry);
        fr_add(out, entry, sum);
        for (int l = 0; l < LIMBS; l++) out[l] = sum[l];
    }
}

__device__ __forceinline__ void brc_accumulate(const u64 *h0, const u64 *h1,
                                               const u64 *__restrict__ rho, unsigned int p,
                                               u64 *fold_c, u64 *fold_e) {
    u64 scale[LIMBS], diff[LIMBS], delta[LIMBS];
    load4(rho + (unsigned long long)p * LIMBS, scale);
    fr_sub(h0, scale, diff);
    pa_fold_mul_accum(h0, diff, fold_c);
    fr_sub(h1, h0, delta);
    pa_fold_mul_accum(delta, delta, fold_e);
}

__device__ __forceinline__ void brc_finish(const u64 *fold_c, const u64 *fold_e,
                                           const u64 *__restrict__ e_in, unsigned int e_in_len,
                                           const u64 *__restrict__ e_out,
                                           unsigned int num_x_in_bits, unsigned long long g,
                                           u64 acc[2][LIMBS]) {
    u64 combined[LIMBS], constant[LIMBS], quadratic[LIMBS];
    eq_split_weight(e_in, e_in_len, e_out, num_x_in_bits, g, combined);
    pa_finalize(fold_c, constant);
    pa_finalize(fold_e, quadratic);
    fr_mul(constant, combined, acc[0]);
    fr_mul(quadratic, combined, acc[1]);
}

extern "C" __global__ void brc_tables_init_kernel(const u64 *__restrict__ base,
                                                 const u64 *__restrict__ rho,
                                                 u64 *__restrict__ out, unsigned int polys,
                                                 unsigned int addresses) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= polys * addresses) return;
    unsigned int p = idx / addresses;
    unsigned int a = idx - p * addresses;

    u64 eq[LIMBS], scale[LIMBS], scaled[LIMBS];
    load4(base + (unsigned long long)a * LIMBS, eq);
    load4(rho + (unsigned long long)p * LIMBS, scale);
    fr_mul(eq, scale, scaled);
    store4(out + (unsigned long long)idx * LIMBS, scaled);
}

extern "C" __global__ void brc_gather_kernel(
    const u64 *__restrict__ lookup, const unsigned int *__restrict__ pc,
    const unsigned int *__restrict__ ram, const u64 *__restrict__ tables,
    unsigned int instruction_polys, unsigned int bytecode_polys, unsigned int ram_polys,
    unsigned int addresses, unsigned int slots, unsigned int chunk_bits, u64 *__restrict__ out,
    unsigned int len) {
    unsigned int p = blockIdx.y;
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= len) return;

    unsigned int family, shift;
    brc_family(p, instruction_polys, bytecode_polys, ram_polys, chunk_bits, &family, &shift);
    unsigned int mask = (1u << chunk_bits) - 1u;
    const u64 *poly_tables = tables + (unsigned long long)p * slots * addresses * LIMBS;

    u64 value[LIMBS];
    brc_sparse_coeff(lookup, pc, ram, poly_tables, addresses, slots, family, shift, mask,
                     (unsigned long long)j * slots, value);
    store4(out + ((unsigned long long)p * len + j) * LIMBS, value);
}

extern "C" __global__ void brc_message_sparse_kernel(
    const u64 *__restrict__ lookup, const unsigned int *__restrict__ pc,
    const unsigned int *__restrict__ ram, const u64 *__restrict__ tables,
    const u64 *__restrict__ rho, unsigned int instruction_polys, unsigned int bytecode_polys,
    unsigned int ram_polys, unsigned int addresses, unsigned int slots, unsigned int chunk_bits,
    unsigned int half, const u64 *__restrict__ e_in, unsigned int e_in_len,
    const u64 *__restrict__ e_out, unsigned int num_x_in_bits, u64 *__restrict__ partials) {
    extern __shared__ u64 scratch[];
    unsigned long long g = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;

    u64 acc[2][LIMBS];
    for (int lane = 0; lane < 2; lane++)
        for (int l = 0; l < LIMBS; l++) acc[lane][l] = 0;

    if (g < half) {
        unsigned int polys = instruction_polys + bytecode_polys + ram_polys;
        unsigned int mask = (1u << chunk_bits) - 1u;
        unsigned long long cycle_even = (unsigned long long)(2 * g) * slots;
        unsigned long long cycle_odd = cycle_even + slots;

        u64 fold_c[2 * PA_SLOTS], fold_e[2 * PA_SLOTS];
        pa_zero(fold_c);
        pa_zero(fold_e);

        for (unsigned int p = 0; p < polys; p++) {
            unsigned int family, shift;
            brc_family(p, instruction_polys, bytecode_polys, ram_polys, chunk_bits, &family,
                       &shift);
            const u64 *poly_tables = tables + (unsigned long long)p * slots * addresses * LIMBS;
            u64 h0[LIMBS], h1[LIMBS];
            brc_sparse_coeff(lookup, pc, ram, poly_tables, addresses, slots, family, shift, mask,
                             cycle_even, h0);
            brc_sparse_coeff(lookup, pc, ram, poly_tables, addresses, slots, family, shift, mask,
                             cycle_odd, h1);
            brc_accumulate(h0, h1, rho, p, fold_c, fold_e);
        }

        brc_finish(fold_c, fold_e, e_in, e_in_len, e_out, num_x_in_bits, g, acc);
    }

    lane_block_reduce(scratch, 2, acc, partials);
}

extern "C" __global__ void brc_message_dense_kernel(
    const u64 *__restrict__ dense, const u64 *__restrict__ rho, unsigned int polys,
    unsigned int half, const u64 *__restrict__ e_in, unsigned int e_in_len,
    const u64 *__restrict__ e_out, unsigned int num_x_in_bits, u64 *__restrict__ partials) {
    extern __shared__ u64 scratch[];
    unsigned long long g = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;

    u64 acc[2][LIMBS];
    for (int lane = 0; lane < 2; lane++)
        for (int l = 0; l < LIMBS; l++) acc[lane][l] = 0;

    if (g < half) {
        u64 fold_c[2 * PA_SLOTS], fold_e[2 * PA_SLOTS];
        pa_zero(fold_c);
        pa_zero(fold_e);

        for (unsigned int p = 0; p < polys; p++) {
            const u64 *row = dense + (unsigned long long)p * 2 * half * LIMBS;
            u64 h0[LIMBS], h1[LIMBS];
            load4(row + (unsigned long long)(2 * g) * LIMBS, h0);
            load4(row + (unsigned long long)(2 * g + 1) * LIMBS, h1);
            brc_accumulate(h0, h1, rho, p, fold_c, fold_e);
        }

        brc_finish(fold_c, fold_e, e_in, e_in_len, e_out, num_x_in_bits, g, acc);
    }

    lane_block_reduce(scratch, 2, acc, partials);
}
