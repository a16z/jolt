#define IRV_LANES 4

__device__ __forceinline__ unsigned int irv_chunk(const u64 *__restrict__ packed,
                                                  unsigned long long cycle, unsigned int shift,
                                                  unsigned int mask) {
    u64 word = packed[2 * cycle + (shift >> 6)];
    return (unsigned int)((word >> (shift & 63)) & (u64)mask);
}

extern "C" __global__ void irv_eq_double_kernel(const u64 *__restrict__ in,
                                                const u64 *__restrict__ point,
                                                u64 *__restrict__ out, unsigned int polys,
                                                unsigned int prev_len, unsigned int level,
                                                unsigned int chunk_bits) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= polys * prev_len) return;
    unsigned int p = idx / prev_len;
    unsigned int i = idx - p * prev_len;

    u64 base[LIMBS], r[LIMBS], one[LIMBS], one_minus_r[LIMBS], lo[LIMBS], hi[LIMBS];
    load4(in + (unsigned long long)idx * LIMBS, base);
    load4(point + (unsigned long long)(p * chunk_bits + level) * LIMBS, r);
    load4(FR_ONE, one);
    fr_sub(one, r, one_minus_r);
    fr_mul(base, one_minus_r, lo);
    fr_mul(base, r, hi);

    unsigned long long out_base = (unsigned long long)p * 2 * prev_len + 2 * i;
    store4(out + out_base * LIMBS, lo);
    store4(out + (out_base + 1) * LIMBS, hi);
}

extern "C" __global__ void irv_tables_split_kernel(const u64 *__restrict__ in,
                                                   const u64 *__restrict__ eq_zero,
                                                   const u64 *__restrict__ eq_one,
                                                   u64 *__restrict__ out, unsigned int polys,
                                                   unsigned int len) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= polys * len) return;
    unsigned int p = idx / len;
    unsigned int i = idx - p * len;

    u64 base[LIMBS], zero[LIMBS], one[LIMBS], lo[LIMBS], hi[LIMBS];
    load4(in + (unsigned long long)idx * LIMBS, base);
    load4(eq_zero, zero);
    load4(eq_one, one);
    fr_mul(base, zero, lo);
    fr_mul(base, one, hi);

    unsigned long long out_base = (unsigned long long)p * 2 * len;
    store4(out + (out_base + i) * LIMBS, lo);
    store4(out + (out_base + len + i) * LIMBS, hi);
}

__device__ __forceinline__ void irv_sparse_coeff(const u64 *__restrict__ packed,
                                                 const u64 *__restrict__ poly_tables,
                                                 unsigned int addresses, unsigned int slots,
                                                 unsigned int shift, unsigned int mask,
                                                 unsigned long long cycle_base, u64 *out) {
    for (int l = 0; l < LIMBS; l++) out[l] = 0;
    for (unsigned int s = 0; s < slots; s++) {
        unsigned int a = irv_chunk(packed, cycle_base + s, shift, mask);
        u64 entry[LIMBS], sum[LIMBS];
        load4(poly_tables + ((unsigned long long)s * addresses + a) * LIMBS, entry);
        fr_add(out, entry, sum);
        for (int l = 0; l < LIMBS; l++) out[l] = sum[l];
    }
}

extern "C" __global__ void irv_gather_kernel(const u64 *__restrict__ packed,
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
    irv_sparse_coeff(packed, poly_tables, addresses, slots, shift, mask,
                     (unsigned long long)j * slots, value);
    store4(out[p] + (unsigned long long)j * LIMBS, value);
}

__device__ __forceinline__ void irv_prod2(const u64 *p0, const u64 *p1, const u64 *q0,
                                          const u64 *q1, u64 *r1, u64 *r2, u64 *r3, u64 *r_inf) {
    u64 p_inf[LIMBS], p2[LIMBS], q_inf[LIMBS], q2[LIMBS];
    fr_sub(p1, p0, p_inf);
    fr_add(p_inf, p1, p2);
    fr_sub(q1, q0, q_inf);
    fr_add(q_inf, q1, q2);
    fr_mul(p1, q1, r1);
    fr_mul(p2, q2, r2);
    fr_mul(p_inf, q_inf, r_inf);

    u64 t[LIMBS], doubled[LIMBS];
    fr_add(r2, r_inf, t);
    fr_add(t, t, doubled);
    fr_sub(doubled, r1, r3);
}

__device__ __forceinline__ void irv_block_reduce(u64 *scratch, u64 acc[IRV_LANES][LIMBS],
                                                 u64 *__restrict__ partials) {
    unsigned int tid = threadIdx.x;
    for (int lane = 0; lane < IRV_LANES; lane++) {
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

__device__ __forceinline__ void irv_weight(const u64 *__restrict__ e_in, unsigned int e_in_len,
                                           const u64 *__restrict__ e_out,
                                           unsigned int num_x_in_bits, unsigned long long g,
                                           u64 *combined) {
    u64 weight[LIMBS];
    if (e_in_len <= 1) {
        load4(FR_ONE, weight);
    } else {
        unsigned long long x_in = g & ((1ull << num_x_in_bits) - 1ull);
        load4(e_in + x_in * LIMBS, weight);
    }
    u64 e_out_eval[LIMBS];
    load4(e_out + (g >> num_x_in_bits) * LIMBS, e_out_eval);
    fr_mul(weight, e_out_eval, combined);
}

extern "C" __global__ void irv_message_sparse_kernel(
    const u64 *__restrict__ packed, const u64 *__restrict__ tables, unsigned int virtual_polys,
    unsigned int addresses, unsigned int slots, unsigned int chunk_bits, unsigned int committed,
    unsigned int half, const u64 *__restrict__ e_in, unsigned int e_in_len,
    const u64 *__restrict__ e_out, unsigned int num_x_in_bits, u64 *__restrict__ partials) {
    extern __shared__ u64 scratch[];
    unsigned long long g = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;

    u64 acc[IRV_LANES][LIMBS];
    for (int lane = 0; lane < IRV_LANES; lane++)
        for (int l = 0; l < LIMBS; l++) acc[lane][l] = 0;

    if (g < half) {
        unsigned int mask = (1u << chunk_bits) - 1u;
        unsigned long long cycle_even = (unsigned long long)(2 * g) * slots;
        unsigned long long cycle_odd = cycle_even + slots;

        u64 inner[IRV_LANES][2 * PA_SLOTS];
        for (int lane = 0; lane < IRV_LANES; lane++) pa_zero(inner[lane]);

        for (unsigned int v = 0; v < virtual_polys; v++) {
            unsigned int base = v * IRV_LANES;
            u64 a1[LIMBS], a2[LIMBS], a3[LIMBS], a_inf[LIMBS];
            u64 b1[LIMBS], b2[LIMBS], b3[LIMBS], b_inf[LIMBS];

            for (int side = 0; side < 2; side++) {
                unsigned int first = base + 2 * side;
                u64 x0[LIMBS], x1[LIMBS], y0[LIMBS], y1[LIMBS];
                for (int which = 0; which < 2; which++) {
                    unsigned int p = first + which;
                    unsigned int shift = chunk_bits * (committed - 1 - p);
                    const u64 *poly_tables =
                        tables + (unsigned long long)p * slots * addresses * LIMBS;
                    u64 *even = which == 0 ? x0 : y0;
                    u64 *odd = which == 0 ? x1 : y1;
                    irv_sparse_coeff(packed, poly_tables, addresses, slots, shift, mask,
                                     cycle_even, even);
                    irv_sparse_coeff(packed, poly_tables, addresses, slots, shift, mask, cycle_odd,
                                     odd);
                }
                if (side == 0) {
                    irv_prod2(x0, x1, y0, y1, a1, a2, a3, a_inf);
                } else {
                    irv_prod2(x0, x1, y0, y1, b1, b2, b3, b_inf);
                }
            }

            pa_fold_mul_accum(a1, b1, inner[0]);
            pa_fold_mul_accum(a2, b2, inner[1]);
            pa_fold_mul_accum(a3, b3, inner[2]);
            pa_fold_mul_accum(a_inf, b_inf, inner[3]);
        }

        u64 combined[LIMBS];
        irv_weight(e_in, e_in_len, e_out, num_x_in_bits, g, combined);
        for (int lane = 0; lane < IRV_LANES; lane++) {
            u64 reduced[LIMBS];
            pa_finalize(inner[lane], reduced);
            fr_mul(reduced, combined, acc[lane]);
        }
    }

    irv_block_reduce(scratch, acc, partials);
}

extern "C" __global__ void irv_message_dense_kernel(
    const u64 *const *__restrict__ dense, unsigned int virtual_polys, unsigned int half,
    const u64 *__restrict__ e_in, unsigned int e_in_len, const u64 *__restrict__ e_out,
    unsigned int num_x_in_bits, u64 *__restrict__ partials) {
    extern __shared__ u64 scratch[];
    unsigned long long g = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;

    u64 acc[IRV_LANES][LIMBS];
    for (int lane = 0; lane < IRV_LANES; lane++)
        for (int l = 0; l < LIMBS; l++) acc[lane][l] = 0;

    if (g < half) {
        u64 inner[IRV_LANES][2 * PA_SLOTS];
        for (int lane = 0; lane < IRV_LANES; lane++) pa_zero(inner[lane]);

        for (unsigned int v = 0; v < virtual_polys; v++) {
            unsigned int base = v * IRV_LANES;
            u64 a1[LIMBS], a2[LIMBS], a3[LIMBS], a_inf[LIMBS];
            u64 b1[LIMBS], b2[LIMBS], b3[LIMBS], b_inf[LIMBS];

            for (int side = 0; side < 2; side++) {
                unsigned int first = base + 2 * side;
                u64 x0[LIMBS], x1[LIMBS], y0[LIMBS], y1[LIMBS];
                for (int which = 0; which < 2; which++) {
                    const u64 *table = dense[first + which];
                    u64 *even = which == 0 ? x0 : y0;
                    u64 *odd = which == 0 ? x1 : y1;
                    load4(table + (2 * g) * LIMBS, even);
                    load4(table + (2 * g + 1) * LIMBS, odd);
                }
                if (side == 0) {
                    irv_prod2(x0, x1, y0, y1, a1, a2, a3, a_inf);
                } else {
                    irv_prod2(x0, x1, y0, y1, b1, b2, b3, b_inf);
                }
            }

            pa_fold_mul_accum(a1, b1, inner[0]);
            pa_fold_mul_accum(a2, b2, inner[1]);
            pa_fold_mul_accum(a3, b3, inner[2]);
            pa_fold_mul_accum(a_inf, b_inf, inner[3]);
        }

        u64 combined[LIMBS];
        irv_weight(e_in, e_in_len, e_out, num_x_in_bits, g, combined);
        for (int lane = 0; lane < IRV_LANES; lane++) {
            u64 reduced[LIMBS];
            pa_finalize(inner[lane], reduced);
            fr_mul(reduced, combined, acc[lane]);
        }
    }

    irv_block_reduce(scratch, acc, partials);
}
